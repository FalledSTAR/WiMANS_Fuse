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
from models.video_teacher import (  # noqa: E402
    ProjectedVideoTeacherClassifier,
    VideoTeacherClassifier,
    build_video_transform,
    normalize_video_backbone_name,
)
from utils import (  # noqa: E402
    accuracy_for_mode,
    append_csv_rows,
    build_epoch_result,
    build_wimans_result_payload,
    compact_prediction_rows,
    count_parameters,
    create_run_dir,
    load_config,
    resolve_path,
    save_checkpoint,
    save_yaml,
    seed_everything,
    setup_run_logger,
    update_result_payload,
    write_result_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "video_teacher.yaml"))
    parser.add_argument("--backbone", "--model", dest="backbone", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--mode", choices=["classifier", "projector"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--projector-hidden-dim", type=int, default=None)
    parser.add_argument("--projector-out-dim", type=int, default=None)
    parser.add_argument("--projector-num-heads", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--keep-top-k", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-flops", action="store_true")
    return parser.parse_args()


def select_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_label_mode(cfg) -> str:
    return str(cfg.get("data", {}).get("label_mode", "single_ce"))


def get_threshold(cfg) -> float:
    return float(cfg.get("test", {}).get("threshold", 0.5))


def get_bce_pos_weight(cfg):
    return cfg.get("train", {}).get("bce_pos_weight", None)


def get_slot_ce_class_weights(cfg):
    loss_cfg = cfg.get("loss", {})
    weights = loss_cfg.get("slot_ce_class_weights")
    if weights is None:
        empty_weight = loss_cfg.get("slot_ce_empty_weight")
        activity_weight = loss_cfg.get("slot_ce_activity_weight", 1.0)
        if empty_weight is None:
            return None
        return [float(empty_weight), *[float(activity_weight)] * 9]

    if isinstance(weights, dict):
        empty_weight = float(weights.get("empty_slot", weights.get("empty", 1.0)))
        activity_default = float(weights.get("activity", 1.0))
        return [
            empty_weight,
            *[float(weights.get(ID_TO_ACTIVITY[idx], activity_default)) for idx in range(9)],
        ]

    if len(weights) != 10:
        raise ValueError(f"loss.slot_ce_class_weights must contain 10 values, got {len(weights)}")
    return [float(value) for value in weights]


def _activity_vector_to_text(vector) -> str:
    active = [ID_TO_ACTIVITY[idx] for idx, value in enumerate(vector) if int(value) == 1]
    if not active:
        return "empty_slot"
    return "|".join(active)


def _slot_class_to_text(class_id: int) -> str:
    class_id = int(class_id)
    if class_id == 0:
        return "empty_slot"
    return ID_TO_ACTIVITY[class_id - 1]


def _slot_class_to_activity_vector(class_id: int):
    vector = [0] * 9
    class_id = int(class_id)
    if class_id > 0:
        vector[class_id - 1] = 1
    return vector


def apply_overrides(cfg, args):
    if args.backbone is not None:
        cfg["video_teacher"]["backbone"] = args.backbone
    if args.weights is not None:
        cfg["video_teacher"]["weights"] = args.weights
    if args.mode is not None:
        cfg["video_teacher"]["mode"] = args.mode
    if args.checkpoint is not None:
        cfg["video_teacher"]["checkpoint"] = args.checkpoint
    if args.projector_hidden_dim is not None:
        cfg.setdefault("video_projector", {})["hidden_dim"] = args.projector_hidden_dim
    if args.projector_out_dim is not None:
        cfg.setdefault("video_projector", {})["out_dim"] = args.projector_out_dim
    if args.projector_num_heads is not None:
        cfg.setdefault("video_projector", {})["num_heads"] = args.projector_num_heads
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
    if args.keep_top_k is not None:
        cfg["train"]["keep_top_k"] = args.keep_top_k
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
    label_mode = get_label_mode(cfg)
    if label_mode == "single_ce":
        labels = dataframe.apply(build_single_user_label, axis=1)
        stratify_labels = (
            labels
            if labels.value_counts().min() >= 2 and len(labels.unique()) <= int(len(dataframe) * float(cfg["data"]["test_size"]))
            else None
        )
    elif label_mode in {"multi_bce", "multi_slot_ce"}:
        stratify_labels = None
    else:
        raise ValueError(f"Unsupported data.label_mode: {label_mode}")
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
        "label_mode": label_mode,
        "use_wifi": False,
        "use_video": True,
        "target_len": cfg["data"].get("target_len", 3000),
        "pad_mode": cfg["data"].get("pad_mode", "left"),
        "truncate_mode": cfg["data"].get("truncate_mode", "tail"),
        "normalize": cfg["data"].get("normalize", "none"),
        "video_num_frames": cfg["video"]["num_frames"],
        "video_transform": video_transform,
    }
    train_dataset = WiMANSHARDataset(train_df, **dataset_kwargs)
    val_dataset = WiMANSHARDataset(val_df, **dataset_kwargs)
    batch_size = int(cfg["train"]["batch_size"])
    drop_last_train = bool(len(train_dataset) > batch_size and len(train_dataset) % batch_size == 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_df, val_df


def build_model(cfg):
    label_mode = get_label_mode(cfg)
    num_classes = int(cfg["video_teacher"]["num_classes"])
    if label_mode == "multi_slot_ce" and num_classes != 60:
        raise ValueError("data.label_mode='multi_slot_ce' requires video_teacher.num_classes=60 for 6 slots x 10 classes")
    if label_mode == "multi_bce" and num_classes != 54:
        raise ValueError("data.label_mode='multi_bce' requires video_teacher.num_classes=54 for 6 slots x 9 activities")
    mode = str(cfg["video_teacher"].get("mode", "classifier")).lower()
    if mode == "projector":
        checkpoint = cfg["video_teacher"].get("checkpoint")
        if checkpoint is None:
            raise ValueError("video_teacher.checkpoint is required when video_teacher.mode is 'projector'")
        projector_cfg = cfg.get("video_projector", {})
        return ProjectedVideoTeacherClassifier(
            backbone=cfg["video_teacher"]["backbone"],
            weights=cfg["video_teacher"]["weights"],
            checkpoint_path=str(resolve_path(PROJECT_ROOT, checkpoint)),
            num_classes=int(cfg["video_teacher"]["num_classes"]),
            freeze_video_teacher=bool(cfg["video_teacher"].get("freeze_video_teacher", True)),
            projector_hidden_dim=int(projector_cfg.get("hidden_dim", 256)),
            projector_out_dim=int(projector_cfg.get("out_dim", 256)),
            projector_num_heads=int(projector_cfg.get("num_heads", 2)),
            dropout=float(cfg["video_teacher"]["dropout"]),
        )
    if mode != "classifier":
        raise ValueError("video_teacher.mode must be 'classifier' or 'projector'")
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


def checkpoint_rank_key(item):
    return (-float(item["val_acc"]), float(item["val_loss"]), -int(item["epoch"]))


def checkpoint_filename(epoch: int, val_acc: float, val_loss: float) -> str:
    return f"epoch_{epoch:03d}_acc_{val_acc:.6f}_loss_{val_loss:.6f}.pt"


def prediction_filename(epoch: int, val_acc: float, val_loss: float) -> str:
    return f"val_predictions_epoch_{epoch:03d}_acc_{val_acc:.6f}_loss_{val_loss:.6f}.csv"


def rewrite_csv_rows(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    if rows:
        append_csv_rows(path, rows, fieldnames)


def refresh_top_prediction_files(
    run_dir: Path,
    top_records,
    prediction_fieldnames: list,
    compact_prediction_fieldnames: list,
    save_detailed: bool = True,
    save_compact: bool = True,
    logger=None,
):
    top_dir = run_dir / "splits" / "top_predictions"
    top_dir.mkdir(parents=True, exist_ok=True)
    for old_file in top_dir.glob("val_predictions_top*.csv"):
        old_file.unlink()

    manifest_rows = []
    for rank, record in enumerate(top_records, start=1):
        epoch = int(record["epoch"])
        val_acc = float(record["val_acc"])
        prefix = f"top{rank:02d}_epoch_{epoch:03d}_acc_{val_acc:.6f}"
        if save_detailed and record["prediction_rows"]:
            rewrite_csv_rows(top_dir / f"val_predictions_{prefix}.csv", record["prediction_rows"], prediction_fieldnames)
        if save_compact and record["compact_prediction_rows"]:
            rewrite_csv_rows(
                top_dir / f"val_predictions_{prefix}_compact.csv",
                record["compact_prediction_rows"],
                compact_prediction_fieldnames,
            )
        epoch_result = record["epoch_result"]
        manifest_rows.append(
            {
                "rank": rank,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": float(record["val_loss"]),
                "official_slot_acc": epoch_result.get("official_slot_acc"),
                "active_slot_acc": epoch_result.get("active_slot_acc"),
                "sample_exact_acc": epoch_result.get("sample_exact_acc"),
            }
        )

    rewrite_csv_rows(
        top_dir / "top_epochs.csv",
        manifest_rows,
        ["rank", "epoch", "val_acc", "val_loss", "official_slot_acc", "active_slot_acc", "sample_exact_acc"],
    )
    if logger is not None:
        logger.info("refreshed_top_prediction_files=%s count=%d", top_dir, len(top_records))


def write_top_checkpoint_manifest(checkpoint_dir: Path, top_checkpoints):
    manifest_path = checkpoint_dir / "top_k_checkpoints.csv"
    manifest_path.unlink(missing_ok=True)
    rows = []
    for rank, item in enumerate(sorted(top_checkpoints, key=checkpoint_rank_key), start=1):
        rows.append(
            {
                "rank": rank,
                "epoch": item["epoch"],
                "val_acc": item["val_acc"],
                "val_loss": item["val_loss"],
                "checkpoint": item["checkpoint"].name,
                "predictions": item["predictions"].name if item.get("predictions") is not None else "",
            }
        )
    append_csv_rows(
        manifest_path,
        rows,
        ["rank", "epoch", "val_acc", "val_loss", "checkpoint", "predictions"],
    )


def build_video_model_summary(model, cfg):
    summary = {
        "backbone": normalize_video_backbone_name(cfg["video_teacher"]["backbone"]),
        "mode": str(cfg["video_teacher"].get("mode", "classifier")),
        "feature_dim": getattr(model, "feature_dim", None),
        "base_feature_dim": getattr(model, "base_feature_dim", None),
        "projector_out_dim": getattr(model, "projector_out_dim", None),
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


def prediction_rows_from_batch(batch, logits, loss, cfg):
    label_mode = get_label_mode(cfg)
    threshold = get_threshold(cfg)
    labels = batch["label"].detach().cpu()
    sample_ids = batch["sample_id"]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]

    rows = []
    batch_size = int(labels.shape[0])
    if label_mode == "multi_bce":
        probs = torch.sigmoid(logits.detach().cpu()).reshape(batch_size, 6, 9)
        pred_mask = (probs > threshold).long()
        true_mask = labels.reshape(batch_size, 6, 9).long()
        slot_correct = pred_mask.eq(true_mask).all(dim=-1)
        active_slots = true_mask.sum(dim=-1) > 0
        exact_sample = slot_correct.all(dim=-1)
        for item_idx, sample_id in enumerate(sample_ids):
            item_true = true_mask[item_idx]
            item_pred = pred_mask[item_idx]
            item_probs = probs[item_idx]
            item_slot_correct = slot_correct[item_idx]
            item_active = active_slots[item_idx]
            active_total = int(item_active.long().sum().item())
            active_correct = int((item_slot_correct & item_active).long().sum().item())
            row = {
                "sample_id": sample_id,
                "task_mode": "multi_bce",
                "threshold": threshold,
                "loss": float(loss.item()),
                "slot_correct_count": int(item_slot_correct.long().sum().item()),
                "slot_total": 6,
                "official_slot_accuracy": float(item_slot_correct.float().mean().item()),
                "active_slot_correct_count": active_correct,
                "active_slot_total": active_total,
                "active_slot_accuracy": "" if active_total == 0 else float(active_correct / active_total),
                "exact_sample_correct": int(bool(exact_sample[item_idx].item())),
            }
            for slot_idx in range(6):
                true_vec = item_true[slot_idx].tolist()
                pred_vec = item_pred[slot_idx].tolist()
                row[f"slot_{slot_idx + 1}_true_activity"] = _activity_vector_to_text(true_vec)
                row[f"slot_{slot_idx + 1}_pred_activity"] = _activity_vector_to_text(pred_vec)
                row[f"slot_{slot_idx + 1}_correct"] = int(bool(item_slot_correct[slot_idx].item()))
                row[f"slot_{slot_idx + 1}_active"] = int(bool(item_active[slot_idx].item()))
                for class_id, class_name in ID_TO_ACTIVITY.items():
                    row[f"true_s{slot_idx + 1}_c{class_id}"] = int(true_vec[class_id])
                    row[f"pred_s{slot_idx + 1}_c{class_id}"] = int(pred_vec[class_id])
                    row[f"prob_s{slot_idx + 1}_{class_name}"] = float(item_probs[slot_idx, class_id].item())
            rows.append(row)
        return rows

    if label_mode == "multi_slot_ce":
        probs = torch.softmax(logits.detach().cpu().reshape(batch_size, 6, 10), dim=-1)
        pred_classes = probs.argmax(dim=-1).long()
        true_classes = labels.reshape(batch_size, 6).long()
        slot_correct = pred_classes.eq(true_classes)
        active_slots = true_classes > 0
        exact_sample = slot_correct.all(dim=-1)
        for item_idx, sample_id in enumerate(sample_ids):
            item_true = true_classes[item_idx]
            item_pred = pred_classes[item_idx]
            item_probs = probs[item_idx]
            item_slot_correct = slot_correct[item_idx]
            item_active = active_slots[item_idx]
            active_total = int(item_active.long().sum().item())
            active_correct = int((item_slot_correct & item_active).long().sum().item())
            row = {
                "sample_id": sample_id,
                "task_mode": "multi_slot_ce",
                "loss": float(loss.item()),
                "slot_correct_count": int(item_slot_correct.long().sum().item()),
                "slot_total": 6,
                "official_slot_accuracy": float(item_slot_correct.float().mean().item()),
                "active_slot_correct_count": active_correct,
                "active_slot_total": active_total,
                "active_slot_accuracy": "" if active_total == 0 else float(active_correct / active_total),
                "exact_sample_correct": int(bool(exact_sample[item_idx].item())),
            }
            for slot_idx in range(6):
                true_id = int(item_true[slot_idx].item())
                pred_id = int(item_pred[slot_idx].item())
                true_vec = _slot_class_to_activity_vector(true_id)
                pred_vec = _slot_class_to_activity_vector(pred_id)
                row[f"slot_{slot_idx + 1}_true_id"] = true_id
                row[f"slot_{slot_idx + 1}_pred_id"] = pred_id
                row[f"slot_{slot_idx + 1}_true_activity"] = _slot_class_to_text(true_id)
                row[f"slot_{slot_idx + 1}_pred_activity"] = _slot_class_to_text(pred_id)
                row[f"slot_{slot_idx + 1}_correct"] = int(bool(item_slot_correct[slot_idx].item()))
                row[f"slot_{slot_idx + 1}_active"] = int(bool(item_active[slot_idx].item()))
                row[f"prob_s{slot_idx + 1}_empty_slot"] = float(item_probs[slot_idx, 0].item())
                for class_id, class_name in ID_TO_ACTIVITY.items():
                    row[f"true_s{slot_idx + 1}_c{class_id}"] = int(true_vec[class_id])
                    row[f"pred_s{slot_idx + 1}_c{class_id}"] = int(pred_vec[class_id])
                    row[f"prob_s{slot_idx + 1}_{class_name}"] = float(item_probs[slot_idx, class_id + 1].item())
            rows.append(row)
        return rows

    probs = torch.softmax(logits.detach().cpu(), dim=-1)
    pred_ids = probs.argmax(dim=-1)
    correct = pred_ids.eq(labels.long())
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
    label_mode = get_label_mode(cfg)
    threshold = get_threshold(cfg)
    bce_pos_weight = get_bce_pos_weight(cfg)
    slot_class_weights = get_slot_ce_class_weights(cfg)
    samples_seen = 0
    optimizer_step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader, start=1):
        video = batch["video"].float().to(device)
        labels = batch["label"].to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(video)
            loss = classification_loss(
                logits,
                labels,
                label_mode,
                pos_weight=bce_pos_weight,
                slot_class_weights=slot_class_weights,
            )

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

        batch_acc = accuracy_for_mode(logits.detach(), labels.detach(), label_mode, threshold=threshold)
        total_loss += float(loss.item()) * int(labels.shape[0])
        total_acc += batch_acc * int(labels.shape[0])
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

    return total_loss / max(samples_seen, 1), total_acc / max(samples_seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    rows = []
    label_mode = get_label_mode(cfg)
    threshold = get_threshold(cfg)
    bce_pos_weight = get_bce_pos_weight(cfg)
    slot_class_weights = get_slot_ce_class_weights(cfg)
    for batch in loader:
        video = batch["video"].float().to(device)
        labels = batch["label"].to(device)
        logits = model(video)
        loss = classification_loss(
            logits,
            labels,
            label_mode,
            pos_weight=bce_pos_weight,
            slot_class_weights=slot_class_weights,
        )
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_acc += accuracy_for_mode(logits, labels, label_mode, threshold=threshold) * batch_size
        total_samples += batch_size
        rows.extend(prediction_rows_from_batch(batch, logits, loss, cfg))
    return total_loss / max(total_samples, 1), total_acc / max(total_samples, 1), rows


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    run_dir = create_run_dir(PROJECT_ROOT, cfg["experiment"]["output_dir"], cfg["experiment"]["name"], "video_teacher", cfg=cfg)
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
    if str(cfg["video_teacher"].get("mode", "classifier")).lower() == "projector":
        logger.info("loaded_base_video_checkpoint=%s", model.checkpoint_path)
        logger.info("base_video_checkpoint_extra=%s", model.checkpoint_extra)
        logger.info("base_video_checkpoint_load_info=%s", model.checkpoint_load_info)
    (run_dir / "model.txt").write_text(str(model), encoding="utf-8")
    logger.info("model_structure:\n%s", model)
    summary = build_video_model_summary(model, cfg)
    save_yaml(run_dir / "model_summary.yaml", summary)
    logger.info("model_summary:\n%s", yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    model = model.to(device)

    label_mode = get_label_mode(cfg)
    task_prefix = "multi_user" if label_mode in {"multi_bce", "multi_slot_ce"} else "single_person"
    class_names = [ID_TO_ACTIVITY[class_id] for class_id in sorted(ID_TO_ACTIVITY)]
    result_payload = build_wimans_result_payload(
        model_name=f"{normalize_video_backbone_name(cfg['video_teacher']['backbone'])}_{cfg['video_teacher'].get('mode', 'classifier')}",
        task=f"{task_prefix}_video_teacher_{cfg['video_teacher'].get('mode', 'classifier')}",
        cfg=cfg,
        model_summary=summary,
    )
    epoch_results = []

    optimizer = build_optimizer(model, cfg, logger=logger)
    scheduler = build_scheduler(optimizer, cfg, logger=logger)
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = -1.0
    keep_top_k = max(int(cfg["train"].get("keep_top_k", 3)), 1)
    top_checkpoints = []
    top_prediction_records = []
    best_checkpoint_path = None
    logging_cfg = cfg.get("logging", {})
    save_epoch_predictions = bool(logging_cfg.get("save_epoch_predictions", True))
    save_best_detailed_predictions = bool(logging_cfg.get("save_best_detailed_predictions", True))
    save_compact_predictions = bool(logging_cfg.get("save_compact_predictions", True))
    save_topk_predictions = max(int(logging_cfg.get("save_topk_predictions", keep_top_k)), 0)
    epoch_fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_acc",
        "official_slot_acc",
        "active_slot_acc",
        "sample_exact_acc",
        "lr_backbone",
        "lr_head",
    ]
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
        val_loss, val_acc, prediction_rows = evaluate(model, val_loader, device, cfg)
        scheduler.step(val_acc)
        lrs_after = current_lrs(optimizer)
        prediction_fieldnames = list(prediction_rows[0].keys()) if prediction_rows else []
        compact_prediction_rows_epoch = compact_prediction_rows(prediction_rows)
        compact_prediction_fieldnames = list(compact_prediction_rows_epoch[0].keys()) if compact_prediction_rows_epoch else []
        if prediction_rows and save_epoch_predictions:
            prediction_path = run_dir / "splits" / f"val_predictions_epoch_{epoch:03d}.csv"
            append_csv_rows(prediction_path, prediction_rows, prediction_fieldnames)
            logger.info("saved_val_predictions=%s", prediction_path)

        message = (
            f"epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}"
        )
        print(message)
        logger.info(message)

        epoch_result = build_epoch_result(
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            prediction_rows,
            class_names,
            split_df=val_df,
            lrs=lrs_after,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": max(best_acc, val_acc),
            "official_slot_acc": epoch_result.get("official_slot_acc"),
            "active_slot_acc": epoch_result.get("active_slot_acc"),
            "sample_exact_acc": epoch_result.get("sample_exact_acc"),
            "lr_backbone": lrs_after.get("backbone", float("nan")),
            "lr_head": lrs_after.get("head", float("nan")),
        }
        append_csv_rows(run_dir / "metrics" / "epochs.csv", [epoch_row], epoch_fieldnames)
        epoch_results.append(epoch_result)

        if prediction_rows and save_topk_predictions > 0:
            top_prediction_records.append(
                {
                    "epoch": epoch,
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                    "epoch_result": epoch_result,
                    "prediction_rows": list(prediction_rows),
                    "compact_prediction_rows": list(compact_prediction_rows_epoch),
                }
            )
            top_prediction_records = sorted(
                top_prediction_records,
                key=lambda item: (-float(item["val_acc"]), float(item["val_loss"]), -int(item["epoch"])),
            )[:save_topk_predictions]
            refresh_top_prediction_files(
                run_dir,
                top_prediction_records,
                prediction_fieldnames,
                compact_prediction_fieldnames,
                save_detailed=save_best_detailed_predictions,
                save_compact=save_compact_predictions,
                logger=logger,
            )

        candidate = {
            "epoch": epoch,
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
        }
        sorted_top = sorted(top_checkpoints, key=checkpoint_rank_key)
        should_save_top = len(sorted_top) < keep_top_k or checkpoint_rank_key(candidate) < checkpoint_rank_key(sorted_top[-1])
        best_acc = max(best_acc, val_acc)

        if should_save_top:
            checkpoint_path = run_dir / "checkpoints" / checkpoint_filename(epoch, val_acc, val_loss)
            top_prediction_path = None
            if prediction_rows:
                top_prediction_path = run_dir / "splits" / prediction_filename(epoch, val_acc, val_loss)
                top_prediction_path.unlink(missing_ok=True)
                append_csv_rows(top_prediction_path, prediction_rows, prediction_fieldnames)

            save_checkpoint(
                str(checkpoint_path),
                model,
                optimizer,
                extra={
                    "epoch": epoch,
                    "val_acc": float(val_acc),
                    "val_loss": float(val_loss),
                    "model_type": f"video_teacher_{cfg['video_teacher'].get('mode', 'classifier')}",
                    "backbone": normalize_video_backbone_name(cfg["video_teacher"]["backbone"]),
                    "feature_dim": model.feature_dim,
                    "base_feature_dim": getattr(model, "base_feature_dim", None),
                    "projector_out_dim": getattr(model, "projector_out_dim", None),
                    "num_classes": int(cfg["video_teacher"]["num_classes"]),
                },
            )
            candidate["checkpoint"] = checkpoint_path
            candidate["predictions"] = top_prediction_path
            top_checkpoints.append(candidate)
            top_checkpoints = sorted(top_checkpoints, key=checkpoint_rank_key)

            removed_checkpoints = top_checkpoints[keep_top_k:]
            top_checkpoints = top_checkpoints[:keep_top_k]
            for removed in removed_checkpoints:
                removed["checkpoint"].unlink(missing_ok=True)
                if removed.get("predictions") is not None:
                    removed["predictions"].unlink(missing_ok=True)
                logger.info(
                    "removed_checkpoint_outside_top_k=%s val_acc=%.6f val_loss=%.6f",
                    removed["checkpoint"],
                    removed["val_acc"],
                    removed["val_loss"],
                )

            write_top_checkpoint_manifest(run_dir / "checkpoints", top_checkpoints)
            logger.info(
                "saved_top_checkpoint=%s val_acc=%.6f val_loss=%.6f keep_top_k=%d",
                checkpoint_path,
                val_acc,
                val_loss,
                keep_top_k,
            )

            if top_checkpoints[0]["checkpoint"] == checkpoint_path and best_checkpoint_path != checkpoint_path:
                best_checkpoint_path = checkpoint_path
                save_checkpoint(
                    str(run_dir / "checkpoints" / "best.pt"),
                    model,
                    optimizer,
                    extra={
                        "epoch": epoch,
                        "val_acc": float(val_acc),
                        "val_loss": float(val_loss),
                        "model_type": f"video_teacher_{cfg['video_teacher'].get('mode', 'classifier')}",
                        "backbone": normalize_video_backbone_name(cfg["video_teacher"]["backbone"]),
                        "feature_dim": model.feature_dim,
                        "base_feature_dim": getattr(model, "base_feature_dim", None),
                        "projector_out_dim": getattr(model, "projector_out_dim", None),
                        "num_classes": int(cfg["video_teacher"]["num_classes"]),
                    },
                )
                if prediction_rows:
                    if save_best_detailed_predictions:
                        best_prediction_path = run_dir / "splits" / "val_predictions_best.csv"
                        best_prediction_path.unlink(missing_ok=True)
                        append_csv_rows(best_prediction_path, prediction_rows, prediction_fieldnames)
                        logger.info("saved_best_val_predictions=%s", best_prediction_path)
                    if save_compact_predictions and compact_prediction_rows_epoch:
                        compact_prediction_path = run_dir / "splits" / "val_predictions_best_compact.csv"
                        compact_prediction_path.unlink(missing_ok=True)
                        append_csv_rows(compact_prediction_path, compact_prediction_rows_epoch, compact_prediction_fieldnames)
                        logger.info("saved_best_compact_val_predictions=%s", compact_prediction_path)
                logger.info(
                    "saved_best_checkpoint=%s val_acc=%.6f val_loss=%.6f",
                    run_dir / "checkpoints" / "best.pt",
                    val_acc,
                    val_loss,
                )

        update_result_payload(result_payload, epoch_results, top_checkpoints=top_checkpoints)
        write_result_json(run_dir / "result.json", result_payload)

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
