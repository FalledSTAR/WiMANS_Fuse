import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import build_loaders, build_model, evaluate, select_device  # noqa: E402
from utils import load_checkpoint, load_config, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--stage", choices=["v0", "v1"], default="v0")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    _, val_loader = build_loaders(cfg, use_video=args.stage == "v1")
    model = build_model(cfg, args.stage).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    loss, acc = evaluate(model, val_loader, device, args.stage)
    print(f"test_loss={loss:.6f} test_acc={acc:.6f}")


if __name__ == "__main__":
    main()
