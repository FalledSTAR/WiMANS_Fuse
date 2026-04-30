import argparse
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import WiMANSHARDataset, build_single_user_dataframe, build_single_user_label  # noqa: E402
from losses import CAFDLoss, classification_loss  # noqa: E402
from models import VideoWiFiCAFDModel, XFiWiFiStudent  # noqa: E402
from utils import accuracy_top1, load_config, resolve_path, save_checkpoint, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--stage", choices=["v0", "v1"], default="v0")
    return parser.parse_args()


def select_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_loaders(cfg, use_video: bool):
    data_root = resolve_path(PROJECT_ROOT, cfg["data"]["root"])
    annotation = resolve_path(PROJECT_ROOT, cfg["data"]["annotation"])
    dataframe = build_single_user_dataframe(
        annotation,
        wifi_band=cfg["data"]["wifi_band"],
        environment=cfg["data"]["environment"],
        num_users=cfg["data"]["num_users"],
        sample_limit=cfg["data"]["sample_limit"],
    )
    labels = dataframe.apply(build_single_user_label, axis=1)
    train_df, val_df = train_test_split(
        dataframe,
        test_size=float(cfg["data"]["test_size"]),
        shuffle=True,
        random_state=int(cfg["experiment"]["seed"]),
        stratify=labels,
    )

    dataset_kwargs = {
        "data_root": data_root,
        "label_mode": "single_ce",
        "use_video": use_video,
        "target_len": cfg["data"]["target_len"],
        "pad_mode": cfg["data"]["pad_mode"],
        "truncate_mode": cfg["data"]["truncate_mode"],
        "normalize": cfg["data"]["normalize"],
        "video_num_frames": cfg["video"]["num_frames"],
    }
    train_dataset = WiMANSHARDataset(train_df, **dataset_kwargs)
    val_dataset = WiMANSHARDataset(val_df, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model(cfg, stage: str):
    weight_path = resolve_path(PROJECT_ROOT, cfg["model"]["xfi_weight_path"])
    if stage == "v0":
        return XFiWiFiStudent(weight_path=weight_path, num_classes=cfg["model"]["num_classes"])

    return VideoWiFiCAFDModel(
        xfi_weight_path=weight_path,
        num_classes=cfg["model"]["num_classes"],
        s3d_weights=cfg["video"]["s3d_weights"],
        freeze_s3d=cfg["video"]["freeze_s3d"],
        projector_hidden_dim=cfg["projector"]["hidden_dim"],
        projector_out_dim=cfg["projector"]["out_dim"],
        projector_num_heads=cfg["projector"]["num_heads"],
    )


def run_epoch(model, loader, optimizer, device, stage, cfg, cafd_loss_fn=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for batch in loader:
        wifi = batch["wifi"].float().to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if stage == "v0":
            logits = model(wifi)
            loss = classification_loss(logits, labels, "single_ce")
        else:
            video = batch["video"].float().to(device)
            outputs = model(wifi, video)
            logits = outputs["logits"]
            cls_loss = classification_loss(logits, labels, "single_ce")
            cafd_loss = cafd_loss_fn(outputs["wifi_projected"], outputs["video_projected"])
            loss = cls_loss + float(cfg["cafd"]["lambda_cafd"]) * cafd_loss

        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_acc += accuracy_top1(logits.detach(), labels.detach())

    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, stage):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    for batch in loader:
        wifi = batch["wifi"].float().to(device)
        labels = batch["label"].to(device)
        if stage == "v0":
            logits = model(wifi)
        else:
            video = batch["video"].float().to(device)
            logits = model(wifi, video)["logits"]
        loss = classification_loss(logits, labels, "single_ce")
        total_loss += float(loss.item())
        total_acc += accuracy_top1(logits, labels)
    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    train_loader, val_loader = build_loaders(cfg, use_video=args.stage == "v1")
    model = build_model(cfg, args.stage).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["train"]["lr_head"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    cafd_loss_fn = None
    if args.stage == "v1":
        cafd_loss_fn = CAFDLoss(
            temperature=float(cfg["cafd"]["temperature"]),
            alpha=float(cfg["cafd"]["alpha"]),
            beta=float(cfg["cafd"]["beta"]),
        )

    best_acc = -1.0
    output_dir = PROJECT_ROOT / cfg["experiment"]["output_dir"] / args.stage
    for epoch in range(int(cfg["train"]["epochs"])):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, args.stage, cfg, cafd_loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, device, args.stage)
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                str(output_dir / "best.pt"),
                model,
                optimizer,
                extra={"epoch": epoch + 1, "val_acc": best_acc, "stage": args.stage},
            )


if __name__ == "__main__":
    main()
