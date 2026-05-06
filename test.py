import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from train import build_loaders, build_model, collect_predictions, select_device  # noqa: E402
from utils import append_csv_rows, load_checkpoint, load_config, seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--stage", choices=["v0", "v1"], default="v0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions-out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    _, val_loader, _, _ = build_loaders(cfg, use_video=args.stage == "v1")
    model = build_model(cfg, args.stage).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    loss, acc, prediction_rows = collect_predictions(model, val_loader, device, args.stage)
    print(f"test_loss={loss:.6f} test_acc={acc:.6f}")
    if args.predictions_out is not None:
        prediction_path = Path(args.predictions_out)
    else:
        checkpoint_path = Path(args.checkpoint).resolve()
        if checkpoint_path.parent.name == "checkpoints":
            prediction_path = checkpoint_path.parent.parent / "splits" / "test_predictions.csv"
        else:
            prediction_path = PROJECT_ROOT / cfg["experiment"]["output_dir"] / "test_predictions.csv"
    if prediction_rows:
        prediction_path.unlink(missing_ok=True)
        append_csv_rows(prediction_path, prediction_rows, list(prediction_rows[0].keys()))
        print(f"predictions_saved={prediction_path}")


if __name__ == "__main__":
    main()
