import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import ID_TO_ACTIVITY, WiMANSHARDataset, build_single_user_dataframe, build_single_user_label  # noqa: E402
from losses import classification_loss  # noqa: E402
from models.video_teacher import VideoTeacherClassifier, build_video_transform, normalize_video_backbone_name  # noqa: E402
from utils import (  # noqa: E402
    accuracy_top1,
    append_csv_rows,
    count_parameters,
    create_run_dir,
    load_config,
    resolve_path,
    save_checkpoint,
    save_yaml,
    seed_everything,
    setup_run_logger,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "video_teacher.yaml"))
    parser.add_argument("--backbone", "--model", dest="backbone", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-flops", action="store_true")
    return parser.parse_args()


def select_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def apply_overrides(cfg, args):
    if args.backbone is not None:
        cfg["video_teacher"]["backbone"] = args.backbone
    if args.weights is not None:
        cfg["video_teacher"]["weights"] = args.weights
    if args.sample_limit is not None:
        cfg["data"]["sample_limit"] = args.sample_limit
    if args.num_frames is not None:
        cfg["video"]["num_frames"] = args.num_frames
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        cfg["train"]["gradient_accumulation_steps"] = args.grad_accum_steps
    if args.freeze_backbone:
        cfg["video_teacher"]["freeze_backbone"] = True
    if args.no_flops:
        cfg.setdefault("logging", {})["compute_flops"] = False


def build_loaders(cfg):
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
    stratify_labels = labels if labels.value_counts().min() >= 2 and len(labels.unique()) <= int(len(dataframe) * float(cfg["data"]["test_size"])) else None
    train_df, val_df = train_test_split(
        dataframe,
        test_size=float(cfg["data"]["test_size"]),
        shuffle=True,
        random_state=int(cfg["experiment"]["seed"]),
        stratify=stratify_labels,
    )

    video_transform = build_video_transform(cfg["video_teacher"]["backbone"], cfg["video_teacher"]["weights"])
    dataset_kwargs = {
        "data_root": data_root,
        "label_mode": "single_ce",
        "use_wifi": False,
        "use_video": True,
        "target_len": 3000,
        "pad_mode": "left",
        "truncate_mode": "tail",
        "normalize": "none",
        "video_num_frames": cfg["video"]["num_frames"],
        "video_transform": video_transform,
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
    return train_loader, val_loader, train_df, val_df


def build_model(cfg):
    return VideoTeacherClassifier(
        backbone=cfg["video_teacher"]["backbone"],
        weights=cfg["video_teacher"]["weights"],
        num_classes=int(cfg["video_teacher"]["num_classes"]),
        freeze_backbone=bool(cfg["video_teacher"]["freeze_backbone"]),
        dropout=float(cfg["video_teacher"]["dropout"]),
    )


def build_optimizer(model, cfg, logger=None):
    backbone_params = [parameter for parameter in model.backbone_parameters() if parameter.requires_grad]
    head_params = [parameter for parameter in model.head_parameters() if parameter.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(cfg["train"]["lr_backbone"]), "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(cfg["train"]["lr_head"]), "name": "head"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg["train"]["weight_decay"]))
    if logger is not None:
        logger.info(
            "optimizer groups: backbone=%d params lr=%.2e | head=%d params lr=%.2e | weight_decay=%.2e",
            sum(parameter.numel() for parameter in backbone_params),
            float(cfg["train"]["lr_backbone"]),
            sum(parameter.numel() for parameter in head_params),
            float(cfg["train"]["lr_head"]),
            float(cfg["train"]["weight_decay"]),
        )
    return optimizer


def build_scheduler(optimizer, cfg, logger=None):
    scheduler_cfg = cfg.get("train", {}).get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 5)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
    )
    if logger is not None:
        logger.info(
            "scheduler ReduceLROnPlateau: mode=max factor=%.2f patience=%d min_lr=%.2e",
            float(scheduler_cfg.get("factor", 0.5)),
            int(scheduler_cfg.get("patience", 5)),
            float(scheduler_cfg.get("min_lr", 1e-7)),
        )
    return scheduler


def current_lrs(optimizer):
    return {group.get("name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}


def build_video_model_summary(model, cfg):
    summary = {
        "backbone": normalize_video_backbone_name(cfg["video_teacher"]["backbone"]),
        "parameters": count_parameters(model),
        "flops": {"available": False, "error": "disabled by config"},
    }
    if not bool(cfg.get("logging", {}).get("compute_flops", True)):
        return summary

    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        summary["flops"] = {"available": False, "error": "ptflops is not installed"}
        return summary

    try:
        module_cpu = copy.deepcopy(model).cpu().eval()
        frames = int(cfg.get("logging", {}).get("flops_video_frames", min(int(cfg["video"]["num_frames"]), 16)))
        macs, params = get_model_complexity_info(
            module_cpu,
            (3, frames, 224, 224),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        summary["flops"] = {
            "available": True,
            "input_res": [3, frames, 224, 224],
            "macs": int(macs),
            "flops_approx": int(macs * 2),
            "ptflops_params_raw": int(params),
            "note": "ptflops reports MACs; flops_approx is MACs * 2.",
        }
    except Exception as exc:  # pragma: no cover - profiler support differs by architecture
        summary["flops"] = {"available": False, "error": repr(exc)}
    finally:
        if "module_cpu" in locals():
            del module_cpu
    return summary


def prediction_rows_from_batch(batch, logits, loss):
    labels = batch["label"].detach().cpu()
    probs = torch.softmax(logits.detach().cpu(), dim=-1)
    pred_ids = probs.argmax(dim=-1)
    correct = pred_ids.eq(labels.long())
    sample_ids = batch["sample_id"]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]

    rows = []
    for item_idx, sample_id in enumerate(sample_ids):
        true_id = int(labels[item_idx].item())
        pred_id = int(pred_ids[item_idx].item())
        item_probs = probs[item_idx].tolist()
        row = {
            "sample_id": sample_id,
            "true_id": true_id,
            "true_activity": ID_TO_ACTIVITY[true_id],
            "pred_id": pred_id,
            "pred_activity": ID_TO_ACTIVITY[pred_id],
            "correct": int(bool(correct[item_idx].item())),
            "pred_probability": float(item_probs[pred_id]),
            "true_probability": float(item_probs[true_id]),
            "loss": float(loss.item()),
        }
        for class_id, class_name in ID_TO_ACTIVITY.items():
            row[f"prob_{class_id}_{class_name}"] = float(item_probs[class_id])
        rows.append(row)
    return rows


def run_epoch(model, loader, optimizer, scaler, device, cfg, epoch, logger=None, batch_csv_path=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batch_rows = []
    log_interval = max(int(cfg["train"].get("log_interval", 10)), 1)
    accumulation_steps = max(int(cfg["train"].get("gradient_accumulation_steps", 1)), 1)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    grad_clip_norm = cfg["train"].get("grad_clip_norm")
    samples_seen = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader, start=1):
        video = batch["video"].float().to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(video)
            loss = classification_loss(logits, labels, "single_ce")

        window_start = ((batch_idx - 1) // accumulation_steps) * accumulation_steps + 1
        window_end = min(window_start + accumulation_steps - 1, len(loader))
        current_accumulation_steps = window_end - window_start + 1
        scaler.scale(loss / current_accumulation_steps).backward()
        should_step = batch_idx % accumulation_steps == 0 or batch_idx == len(loader)
        if should_step:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        batch_acc = accuracy_top1(logits.detach(), labels.detach())
        total_loss += float(loss.item())
        total_acc += batch_acc
        samples_seen += int(labels.shape[0])
        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "samples_seen": samples_seen,
            "loss": float(loss.item()),
            "accuracy": batch_acc,
            "batch_size": int(labels.shape[0]),
            "accumulation_step": ((batch_idx - 1) % accumulation_steps) + 1,
            "optimizer_step": optimizer_step,
        }
        batch_rows.append(row)

        if logger is not None and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader)):
            logger.info(
                "train epoch=%s batch=%s/%s loss=%.6f acc=%.6f accum=%s/%s opt_step=%s",
                epoch,
                batch_idx,
                len(loader),
                row["loss"],
                row["accuracy"],
                row["accumulation_step"],
                accumulation_steps,
                row["optimizer_step"],
            )

    if batch_csv_path is not None:
        append_csv_rows(
            batch_csv_path,
            batch_rows,
            ["epoch", "batch", "samples_seen", "loss", "accuracy", "batch_size", "accumulation_step", "optimizer_step"],
        )

    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    rows = []
    for batch in loader:
        video = batch["video"].float().to(device)
        labels = batch["label"].to(device)
        logits = model(video)
        loss = classification_loss(logits, labels, "single_ce")
        total_loss += float(loss.item())
        total_acc += accuracy_top1(logits, labels)
        rows.extend(prediction_rows_from_batch(batch, logits, loss))
    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1), rows


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    run_dir = create_run_dir(PROJECT_ROOT, cfg["experiment"]["output_dir"], cfg["experiment"]["name"], "video_teacher")
    logger = setup_run_logger(run_dir / "train.log")
    logger.info("run_dir=%s", run_dir)
    logger.info("device=%s", device)
    logger.info("effective_config:\n%s", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    save_yaml(run_dir / "config.yaml", cfg)

    train_loader, val_loader, train_df, val_df = build_loaders(cfg)
    train_df.to_csv(run_dir / "splits" / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(run_dir / "splits" / "val.csv", index=False, encoding="utf-8-sig")
    logger.info("dataset_split train=%s val=%s", len(train_df), len(val_df))
    logger.info(
        "video_loading=online_mp4 no_npy_preprocess micro_batch_size=%d gradient_accumulation_steps=%d effective_batch_size=%d",
        int(cfg["train"]["batch_size"]),
        max(int(cfg["train"].get("gradient_accumulation_steps", 1)), 1),
        int(cfg["train"]["batch_size"]) * max(int(cfg["train"].get("gradient_accumulation_steps", 1)), 1),
    )

    model = build_model(cfg)
    (run_dir / "model.txt").write_text(str(model), encoding="utf-8")
    logger.info("model_structure:\n%s", model)
    summary = build_video_model_summary(model, cfg)
    save_yaml(run_dir / "model_summary.yaml", summary)
    logger.info("model_summary:\n%s", yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    model = model.to(device)

    optimizer = build_optimizer(model, cfg, logger=logger)
    scheduler = build_scheduler(optimizer, cfg, logger=logger)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    epoch_fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "lr_backbone", "lr_head"]
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        lrs_before = current_lrs(optimizer)
        logger.info(
            "epoch=%d lr_backbone=%.2e lr_head=%.2e",
            epoch,
            lrs_before.get("backbone", float("nan")),
            lrs_before.get("head", float("nan")),
        )
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            cfg,
            epoch,
            logger=logger,
            batch_csv_path=run_dir / "metrics" / "train_batches.csv",
        )
        val_loss, val_acc, prediction_rows = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        lrs_after = current_lrs(optimizer)
        if prediction_rows:
            prediction_path = run_dir / "splits" / f"val_predictions_epoch_{epoch:03d}.csv"
            append_csv_rows(prediction_path, prediction_rows, list(prediction_rows[0].keys()))
            logger.info("saved_val_predictions=%s", prediction_path)

        message = (
            f"epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}"
        )
        print(message)
        logger.info(message)

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": max(best_acc, val_acc),
            "lr_backbone": lrs_after.get("backbone", float("nan")),
            "lr_head": lrs_after.get("head", float("nan")),
        }
        append_csv_rows(run_dir / "metrics" / "epochs.csv", [epoch_row], epoch_fieldnames)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                str(run_dir / "checkpoints" / "best.pt"),
                model,
                optimizer,
                extra={
                    "epoch": epoch,
                    "val_acc": best_acc,
                    "model_type": "video_teacher",
                    "backbone": normalize_video_backbone_name(cfg["video_teacher"]["backbone"]),
                    "feature_dim": model.feature_dim,
                    "num_classes": int(cfg["video_teacher"]["num_classes"]),
                },
            )
            if prediction_rows:
                best_prediction_path = run_dir / "splits" / "val_predictions_best.csv"
                best_prediction_path.unlink(missing_ok=True)
                append_csv_rows(best_prediction_path, prediction_rows, list(prediction_rows[0].keys()))
                logger.info("saved_best_val_predictions=%s", best_prediction_path)
            logger.info("saved_best_checkpoint=%s val_acc=%.6f", run_dir / "checkpoints" / "best.pt", best_acc)

    logger.info("training_finished best_acc=%.6f", best_acc)
    logger.info(
        "run_artifacts config=%s model=%s summary=%s splits=%s metrics=%s",
        run_dir / "config.yaml",
        run_dir / "model.txt",
        run_dir / "model_summary.yaml",
        run_dir / "splits",
        run_dir / "metrics",
    )


if __name__ == "__main__":
    main()
