import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import WiMANSHARDataset, build_single_user_dataframe  # noqa: E402
from losses import CAFDLoss, classification_loss  # noqa: E402
from models import VideoWiFiCAFDModel  # noqa: E402
from utils import load_checkpoint, load_config, resolve_path, save_checkpoint, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--s3d-weights", default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(PROJECT_ROOT, cfg["data"]["root"])
    annotation = resolve_path(PROJECT_ROOT, cfg["data"]["annotation"])
    weight_path = resolve_path(PROJECT_ROOT, cfg["model"]["xfi_weight_path"])

    dataframe = build_single_user_dataframe(
        annotation,
        wifi_band=cfg["data"]["wifi_band"],
        environment=cfg["data"]["environment"],
        num_users=cfg["data"]["num_users"],
        sample_limit=args.limit,
    )
    dataset = WiMANSHARDataset(
        dataframe,
        data_root=data_root,
        label_mode="single_ce",
        use_video=True,
        target_len=cfg["data"]["target_len"],
        pad_mode=cfg["data"]["pad_mode"],
        truncate_mode=cfg["data"]["truncate_mode"],
        normalize=cfg["data"]["normalize"],
        video_num_frames=args.num_frames,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = VideoWiFiCAFDModel(
        xfi_weight_path=weight_path,
        num_classes=cfg["model"]["num_classes"],
        s3d_weights=args.s3d_weights,
        freeze_s3d=True,
        projector_hidden_dim=cfg["projector"]["hidden_dim"],
        projector_out_dim=cfg["projector"]["out_dim"],
        projector_num_heads=cfg["projector"]["num_heads"],
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["train"]["lr_head"]),
    )
    cafd_loss_fn = CAFDLoss(
        temperature=float(cfg["cafd"]["temperature"]),
        alpha=float(cfg["cafd"]["alpha"]),
        beta=float(cfg["cafd"]["beta"]),
    )

    wifi = batch["wifi"].float().to(device)
    video = batch["video"].float().to(device)
    labels = batch["label"].to(device)

    outputs = model(wifi, video)
    cls_loss = classification_loss(outputs["logits"], labels, "single_ce")
    cafd_loss = cafd_loss_fn(outputs["wifi_projected"], outputs["video_projected"])
    total_loss = cls_loss + float(cfg["cafd"]["lambda_cafd"]) * cafd_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    ckpt_path = PROJECT_ROOT / "output" / "smoke_v1" / "checkpoint.pt"
    save_checkpoint(str(ckpt_path), model, optimizer, extra={"loss": float(total_loss.item())})
    load_checkpoint(str(ckpt_path), model, optimizer, map_location=device)

    trainable_teacher_params = sum(p.requires_grad for p in model.video_teacher.parameters())
    print(f"logits_shape={tuple(outputs['logits'].shape)}")
    print(f"video_shape={tuple(video.shape)}")
    print(f"cls_loss={float(cls_loss.item()):.6f}")
    print(f"cafd_loss={float(cafd_loss.item()):.6f}")
    print(f"total_loss={float(total_loss.item()):.6f}")
    print(f"trainable_teacher_params={trainable_teacher_params}")
    print(f"checkpoint={ckpt_path}")
    print("SMOKE_V1_OK")


if __name__ == "__main__":
    main()
