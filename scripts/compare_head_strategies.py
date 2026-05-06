import argparse
import sys
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import WiMANSHARDataset, build_single_user_dataframe, build_single_user_label  # noqa: E402
from models import XFiWiFiOriginalFC, XFiWiFiStudent  # noqa: E402
from train import collect_predictions, run_epoch, select_device  # noqa: E402
from utils import (  # noqa: E402
    append_csv_rows,
    count_parameters,
    create_run_dir,
    load_config,
    resolve_path,
    save_yaml,
    seed_everything,
    setup_run_logger,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--sample-limit", type=int, default=90)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def build_split(cfg):
    annotation = resolve_path(PROJECT_ROOT, cfg["data"]["annotation"])
    dataframe = build_single_user_dataframe(
        annotation,
        wifi_band=cfg["data"]["wifi_band"],
        environment=cfg["data"]["environment"],
        num_users=cfg["data"]["num_users"],
        sample_limit=cfg["data"]["sample_limit"],
    )
    labels = dataframe.apply(build_single_user_label, axis=1)
    stratify_labels = labels if labels.value_counts().min() >= 2 else None
    return train_test_split(
        dataframe,
        test_size=float(cfg["data"]["test_size"]),
        shuffle=True,
        random_state=int(cfg["experiment"]["seed"]),
        stratify=stratify_labels,
    )


def build_loaders_from_split(cfg, train_df, val_df, seed: int):
    data_root = resolve_path(PROJECT_ROOT, cfg["data"]["root"])
    dataset_kwargs = {
        "data_root": data_root,
        "label_mode": "single_ce",
        "use_video": False,
        "target_len": cfg["data"]["target_len"],
        "pad_mode": cfg["data"]["pad_mode"],
        "truncate_mode": cfg["data"]["truncate_mode"],
        "normalize": cfg["data"]["normalize"],
        "video_num_frames": cfg["video"]["num_frames"],
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        WiMANSHARDataset(train_df, **dataset_kwargs),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        WiMANSHARDataset(val_df, **dataset_kwargs),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_strategy_model(strategy: str, weight_path: Path, num_classes: int):
    if strategy == "xfi_feature_head":
        return XFiWiFiStudent(str(weight_path), num_classes=num_classes)
    if strategy == "original_fc_replace":
        return XFiWiFiOriginalFC(str(weight_path), num_classes=num_classes)
    raise ValueError(f"Unknown strategy: {strategy}")


def run_strategy(strategy, cfg, train_df, val_df, run_dir, logger, device):
    seed_everything(int(cfg["experiment"]["seed"]))
    train_loader, val_loader = build_loaders_from_split(cfg, train_df, val_df, int(cfg["experiment"]["seed"]))
    weight_path = resolve_path(PROJECT_ROOT, cfg["model"]["xfi_weight_path"])
    model = build_strategy_model(strategy, weight_path, int(cfg["model"]["num_classes"]))
    params = count_parameters(model)
    logger.info("strategy=%s parameters=%s", strategy, params)
    logger.info("strategy=%s model_structure:\n%s", strategy, model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["train"]["lr_head"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    best = {"epoch": 0, "val_acc": -1.0, "val_loss": None}
    epoch_rows = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            "v0",
            cfg,
            epoch=epoch,
            logger=logger,
            batch_csv_path=run_dir / "metrics" / f"{strategy}_train_batches.csv",
        )
        val_loss, val_acc, prediction_rows = collect_predictions(model, val_loader, device, "v0")
        row = {
            "strategy": strategy,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "params_total": params["total"],
            "params_trainable": params["trainable"],
        }
        epoch_rows.append(row)
        append_csv_rows(
            run_dir / "metrics" / "head_strategy_epochs.csv",
            [row],
            ["strategy", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "params_total", "params_trainable"],
        )
        logger.info(
            "strategy=%s epoch=%s train_loss=%.6f train_acc=%.6f val_loss=%.6f val_acc=%.6f",
            strategy,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        if val_acc > best["val_acc"]:
            best = {"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss}
            if prediction_rows:
                prediction_path = run_dir / "splits" / f"{strategy}_val_predictions_best.csv"
                prediction_path.unlink(missing_ok=True)
                append_csv_rows(prediction_path, prediction_rows, list(prediction_rows[0].keys()))
                logger.info("strategy=%s saved_best_predictions=%s", strategy, prediction_path)

    final = epoch_rows[-1]
    return {
        "strategy": strategy,
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "final_train_acc": final["train_acc"],
        "final_val_acc": final["val_acc"],
        "params_total": params["total"],
        "params_trainable": params["trainable"],
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["sample_limit"] = args.sample_limit
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_size
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    run_dir = create_run_dir(PROJECT_ROOT, cfg["experiment"]["output_dir"], "head_strategy_compare", "v0")
    logger = setup_run_logger(run_dir / "train.log")
    save_yaml(run_dir / "config.yaml", cfg)
    logger.info("run_dir=%s", run_dir)
    logger.info("device=%s", device)
    logger.info("effective_config:\n%s", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    train_df, val_df = build_split(cfg)
    train_df.to_csv(run_dir / "splits" / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(run_dir / "splits" / "val.csv", index=False, encoding="utf-8-sig")
    logger.info("dataset_split train=%s val=%s", len(train_df), len(val_df))

    rows = []
    for strategy in ("xfi_feature_head", "original_fc_replace"):
        rows.append(run_strategy(strategy, cfg, train_df, val_df, run_dir, logger, device))

    append_csv_rows(
        run_dir / "metrics" / "head_strategy_comparison.csv",
        rows,
        [
            "strategy",
            "best_epoch",
            "best_val_acc",
            "best_val_loss",
            "final_train_acc",
            "final_val_acc",
            "params_total",
            "params_trainable",
        ],
    )
    print(f"comparison_saved={run_dir / 'metrics' / 'head_strategy_comparison.csv'}")
    for row in rows:
        print(
            "strategy={strategy} best_epoch={best_epoch} best_val_acc={best_val_acc:.6f} "
            "final_train_acc={final_train_acc:.6f} final_val_acc={final_val_acc:.6f}".format(**row)
        )


if __name__ == "__main__":
    main()
