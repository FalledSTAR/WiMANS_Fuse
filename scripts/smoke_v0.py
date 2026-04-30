import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import WiMANSHARDataset, build_single_user_dataframe  # noqa: E402
from losses import classification_loss  # noqa: E402
from models import XFiWiFiStudent  # noqa: E402
from utils import accuracy_top1, load_checkpoint, load_config, resolve_path, save_checkpoint, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["experiment"]["seed"])

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
        use_video=False,
        target_len=cfg["data"]["target_len"],
        pad_mode=cfg["data"]["pad_mode"],
        truncate_mode=cfg["data"]["truncate_mode"],
        normalize=cfg["data"]["normalize"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = XFiWiFiStudent(weight_path=weight_path, num_classes=cfg["model"]["num_classes"])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr_head"]))

    logits = model(batch["wifi"].float())
    loss = classification_loss(logits, batch["label"], "single_ce")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ckpt_path = PROJECT_ROOT / "output" / "smoke_v0" / "checkpoint.pt"
    save_checkpoint(str(ckpt_path), model, optimizer, extra={"loss": float(loss.item())})
    load_checkpoint(str(ckpt_path), model, optimizer, map_location="cpu")

    print(f"logits_shape={tuple(logits.shape)}")
    print(f"loss={float(loss.item()):.6f}")
    print(f"accuracy={accuracy_top1(logits.detach(), batch['label']):.6f}")
    print(f"checkpoint={ckpt_path}")
    print("SMOKE_V0_OK")


if __name__ == "__main__":
    main()
