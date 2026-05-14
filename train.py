import argparse
import sys
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import ID_TO_ACTIVITY, WiMANSHARDataset, build_single_user_dataframe, build_single_user_label  # noqa: E402
from losses import CAFDLoss, RSDLoss, classification_loss, logits_kd_loss  # noqa: E402
from models import VideoWiFiCAFDModel, XFiWiFiStudent  # noqa: E402
from utils import (  # noqa: E402
    accuracy_top1,
    append_csv_rows,
    build_epoch_result,
    build_model_summary,
    build_wimans_result_payload,
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
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--stage", choices=["v0", "v1"], default="v0")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--s3d-weights", default=None)
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--projector-target", choices=["video_feature", "projected"], default=None)
    parser.add_argument("--lambda-cafd", type=float, default=None)
    parser.add_argument("--lambda-logits", type=float, default=None)
    parser.add_argument("--kd-temperature", type=float, default=None)
    parser.add_argument("--kd-warmup-epochs", type=int, default=None)
    parser.add_argument("--lambda-rsd", type=float, default=None)
    parser.add_argument("--rsd-kappa", type=float, default=None)
    parser.add_argument("--rsd-warmup-epochs", type=int, default=None)
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
    stratify_labels = labels if labels.value_counts().min() >= 2 and len(labels.unique()) <= int(len(dataframe) * float(cfg["data"]["test_size"])) else None
    train_df, val_df = train_test_split(
        dataframe,
        test_size=float(cfg["data"]["test_size"]),
        shuffle=True,
        random_state=int(cfg["experiment"]["seed"]),
        stratify=stratify_labels,
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
    batch_size = int(cfg["train"]["batch_size"])
    drop_last_train = bool(use_video and len(train_dataset) > batch_size and len(train_dataset) % batch_size == 1)
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


def build_model(cfg, stage: str):
    weight_path = resolve_path(PROJECT_ROOT, cfg["model"]["xfi_weight_path"])
    if stage == "v0":
        return XFiWiFiStudent(weight_path=weight_path, num_classes=cfg["model"]["num_classes"])

    teacher_checkpoint = cfg["video"].get("teacher_checkpoint")
    teacher_checkpoint_path = resolve_path(PROJECT_ROOT, teacher_checkpoint) if teacher_checkpoint else None
    return VideoWiFiCAFDModel(
        xfi_weight_path=weight_path,
        num_classes=cfg["model"]["num_classes"],
        s3d_weights=cfg["video"]["s3d_weights"],
        teacher_checkpoint_path=teacher_checkpoint_path,
        freeze_s3d=cfg["video"]["freeze_s3d"],
        projector_hidden_dim=cfg["projector"]["hidden_dim"],
        projector_out_dim=cfg["projector"]["out_dim"],
        projector_num_heads=cfg["projector"]["num_heads"],
        projector_target=str(cfg["projector"].get("target", "video_feature")),
        freeze_video_projector=bool(cfg["projector"].get("freeze_video_projector", True)),
        use_projector_logits=bool(cfg["projector"].get("use_projector_logits", True)),
        projector_dropout=float(cfg["projector"].get("dropout", 0.2)),
    )


def build_optimizer(model, cfg, stage: str, logger=None) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with separate learning rates for the WiFi backbone
    (feature_extractor) and all other trainable parameters (head, projectors,
    LayerNorm, etc.).

    Config keys used:
        train.lr_backbone  – applied to wifi_student.feature_extractor
        train.lr_head      – applied to every other trainable parameter
        train.weight_decay – shared weight decay
    """
    lr_backbone = float(cfg["train"]["lr_backbone"])
    lr_head = float(cfg["train"]["lr_head"])
    weight_decay = float(cfg["train"]["weight_decay"])

    # Collect backbone parameter ids so we can split the groups cleanly.
    if stage == "v0":
        # XFiWiFiStudent: feature_extractor is the pretrained backbone.
        backbone_params = list(model.feature_extractor.parameters())
    else:
        # VideoWiFiCAFDModel: wifi_student.feature_extractor is the pretrained backbone.
        # S3D teacher is frozen entirely and contributes no gradients.
        backbone_params = list(model.wifi_student.feature_extractor.parameters())

    backbone_ids = {id(p) for p in backbone_params}
    backbone_trainable = [p for p in backbone_params if p.requires_grad]
    other_trainable = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]

    param_groups = []
    if backbone_trainable:
        param_groups.append({"params": backbone_trainable, "lr": lr_backbone, "name": "backbone"})
    if other_trainable:
        param_groups.append({"params": other_trainable, "lr": lr_head, "name": "head_projector"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    if logger is not None:
        logger.info(
            "optimizer groups: backbone=%d params lr=%.2e | head_projector=%d params lr=%.2e | weight_decay=%.2e",
            sum(p.numel() for p in backbone_trainable),
            lr_backbone,
            sum(p.numel() for p in other_trainable),
            lr_head,
            weight_decay,
        )

    return optimizer


def build_scheduler(optimizer, cfg, logger=None) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Build a ReduceLROnPlateau scheduler that monitors val_acc (mode='max').

    Config keys used (all under train.scheduler):
        factor   – LR multiplication factor on plateau  (default 0.5)
        patience – epochs with no improvement to wait   (default 5)
        min_lr   – floor for any param group LR         (default 1e-7)

    When val_acc stops improving for `patience` epochs, every param group's
    LR is multiplied by `factor`, down to a minimum of `min_lr`.
    """
    sched_cfg = cfg.get("train", {}).get("scheduler", {})
    factor = float(sched_cfg.get("factor", 0.5))
    patience = int(sched_cfg.get("patience", 5))
    min_lr = float(sched_cfg.get("min_lr", 1e-7))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",        # we want val_acc to go UP
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        verbose=False,     # we log manually below
    )

    if logger is not None:
        logger.info(
            "scheduler ReduceLROnPlateau: mode=max factor=%.2f patience=%d min_lr=%.2e",
            factor,
            patience,
            min_lr,
        )

    return scheduler


def get_current_lrs(optimizer) -> dict:
    """Return current LR for each named param group as a dict."""
    lrs = {}
    for group in optimizer.param_groups:
        name = group.get("name", f"group_{optimizer.param_groups.index(group)}")
        lrs[name] = group["lr"]
    return lrs


def get_logits_kd_cfg(cfg) -> dict:
    kd_cfg = cfg.get("logits_kd", {})
    return {
        "enabled": bool(kd_cfg.get("enable", False)) and float(kd_cfg.get("lambda_logits", 0.0)) > 0,
        "lambda_logits": float(kd_cfg.get("lambda_logits", 0.0)),
        "temperature": float(kd_cfg.get("temperature", 4.0)),
        "warmup_epochs": int(kd_cfg.get("warmup_epochs", 0)),
        "confidence_threshold": float(kd_cfg.get("confidence_threshold", 0.0)),
    }


def effective_logits_kd_lambda(kd_cfg: dict, epoch: int) -> float:
    target = float(kd_cfg["lambda_logits"])
    warmup_epochs = int(kd_cfg.get("warmup_epochs", 0))
    if not kd_cfg["enabled"] or target <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return target
    return target * min(max(int(epoch), 1) / warmup_epochs, 1.0)


def get_rsd_cfg(cfg) -> dict:
    rsd_cfg = cfg.get("rsd", {})
    return {
        "enabled": bool(rsd_cfg.get("enable", False)) and float(rsd_cfg.get("lambda_rsd", 0.0)) > 0,
        "lambda_rsd": float(rsd_cfg.get("lambda_rsd", 0.0)),
        "kappa": float(rsd_cfg.get("kappa", 0.01)),
        "warmup_epochs": int(rsd_cfg.get("warmup_epochs", 0)),
        "source": str(rsd_cfg.get("source", "distill")),
        "reduction": str(rsd_cfg.get("reduction", "sum")),
    }


def effective_rsd_lambda(rsd_cfg: dict, epoch: int) -> float:
    target = float(rsd_cfg["lambda_rsd"])
    warmup_epochs = int(rsd_cfg.get("warmup_epochs", 0))
    if not rsd_cfg["enabled"] or target <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return target
    return target * min(max(int(epoch), 1) / warmup_epochs, 1.0)


def select_rsd_pair(outputs: dict, source: str):
    if source == "distill":
        return outputs["wifi_distill_feature"], outputs["teacher_distill_feature"]
    if source in {"video_feature", "teacher_feature"}:
        return outputs["wifi_projected"], outputs["video_feature"]
    if source == "projected":
        return outputs["wifi_projected"], outputs["video_projected"]
    raise ValueError(f"Unsupported rsd.source: {source}")


def run_epoch(model, loader, optimizer, device, stage, cfg, epoch: int, cafd_loss_fn=None, rsd_loss_fn=None, logger=None, batch_csv_path=None):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    batch_rows = []
    log_interval = max(int(cfg["train"].get("log_interval", 50)), 1)
    kd_cfg = get_logits_kd_cfg(cfg)
    lambda_logits_effective = effective_logits_kd_lambda(kd_cfg, epoch)
    rsd_cfg = get_rsd_cfg(cfg)
    lambda_rsd_effective = effective_rsd_lambda(rsd_cfg, epoch)
    samples_seen = 0
    for batch_idx, batch in enumerate(loader, start=1):
        wifi = batch["wifi"].float().to(device)
        labels = batch["label"].to(device)
        batch_size = int(labels.shape[0])
        samples_seen += batch_size
        optimizer.zero_grad()

        if stage == "v0":
            logits = model(wifi)
            loss = classification_loss(logits, labels, "single_ce")
            cls_loss_value = float(loss.item())
            cafd_loss_value = None
            logits_kd_loss_value = None
            logits_kd_weighted_value = None
            lambda_logits_value = None
            cafd_weighted_mse_value = None
            cafd_correlation_value = None
            cafd_diagonal_gap_value = None
            cafd_relation_kl_value = None
            cafd_plain_mse_value = None
            rsd_loss_value = None
            rsd_weighted_value = None
            lambda_rsd_value = None
            rsd_invariance_value = None
            rsd_decorrelation_value = None
            rsd_diag_mean_value = None
            rsd_offdiag_abs_mean_value = None
            teacher_acc_value = None
        else:
            video = batch["video"].float().to(device)
            outputs = model(wifi, video)
            logits = outputs["logits"]
            cls_loss = classification_loss(logits, labels, "single_ce")
            cafd_loss, cafd_details = cafd_loss_fn(
                outputs["wifi_distill_feature"],
                outputs["teacher_distill_feature"],
                return_details=True,
            )
            loss = cls_loss + float(cfg["cafd"]["lambda_cafd"]) * cafd_loss
            logits_kd_value = None
            if kd_cfg["enabled"] and lambda_logits_effective > 0:
                logits_kd_value = logits_kd_loss(
                    logits,
                    outputs["teacher_logits"],
                    temperature=kd_cfg["temperature"],
                    confidence_threshold=kd_cfg["confidence_threshold"],
                )
                loss = loss + lambda_logits_effective * logits_kd_value
            rsd_value = None
            rsd_details = None
            if rsd_loss_fn is not None and rsd_cfg["enabled"] and lambda_rsd_effective > 0:
                student_rsd, teacher_rsd = select_rsd_pair(outputs, rsd_cfg["source"])
                rsd_value, rsd_details = rsd_loss_fn(student_rsd, teacher_rsd, return_details=True)
                loss = loss + lambda_rsd_effective * rsd_value
            cls_loss_value = float(cls_loss.item())
            cafd_loss_value = float(cafd_loss.item())
            logits_kd_loss_value = None if logits_kd_value is None else float(logits_kd_value.item())
            logits_kd_weighted_value = None if logits_kd_value is None else float((lambda_logits_effective * logits_kd_value).item())
            lambda_logits_value = lambda_logits_effective
            cafd_weighted_mse_value = float(cafd_details["weighted_mse"].item())
            cafd_correlation_value = float(cafd_details["correlation"].item())
            cafd_diagonal_gap_value = float(cafd_details["diagonal_gap"].item())
            cafd_relation_kl_value = float(cafd_details["relation_kl"].item())
            cafd_plain_mse_value = float(cafd_details["plain_mse"].item())
            rsd_loss_value = None if rsd_value is None else float(rsd_value.item())
            rsd_weighted_value = None if rsd_value is None else float((lambda_rsd_effective * rsd_value).item())
            lambda_rsd_value = lambda_rsd_effective
            rsd_invariance_value = None if rsd_details is None else float(rsd_details["invariance"].item())
            rsd_decorrelation_value = None if rsd_details is None else float(rsd_details["decorrelation"].item())
            rsd_diag_mean_value = None if rsd_details is None else float(rsd_details["diag_mean"].item())
            rsd_offdiag_abs_mean_value = None if rsd_details is None else float(rsd_details["offdiag_abs_mean"].item())
            teacher_acc_value = accuracy_top1(outputs["teacher_logits"].detach(), labels.detach())

        loss.backward()
        optimizer.step()
        batch_acc = accuracy_top1(logits.detach(), labels.detach())
        total_loss += float(loss.item()) * batch_size
        total_correct += batch_acc * batch_size
        total_samples += batch_size

        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "samples_seen": samples_seen,
            "loss": float(loss.item()),
            "classification_loss": cls_loss_value,
            "cafd_loss": cafd_loss_value,
            "cafd_weighted_mse": cafd_weighted_mse_value,
            "cafd_correlation": cafd_correlation_value,
            "cafd_diagonal_gap": cafd_diagonal_gap_value,
            "cafd_relation_kl": cafd_relation_kl_value,
            "cafd_plain_mse": cafd_plain_mse_value,
            "logits_kd_loss": logits_kd_loss_value,
            "logits_kd_weighted_loss": logits_kd_weighted_value,
            "lambda_logits_effective": lambda_logits_value,
            "rsd_loss": rsd_loss_value,
            "rsd_weighted_loss": rsd_weighted_value,
            "lambda_rsd_effective": lambda_rsd_value,
            "rsd_invariance": rsd_invariance_value,
            "rsd_decorrelation": rsd_decorrelation_value,
            "rsd_diag_mean": rsd_diag_mean_value,
            "rsd_offdiag_abs_mean": rsd_offdiag_abs_mean_value,
            "teacher_accuracy": teacher_acc_value,
            "accuracy": batch_acc,
            "batch_size": int(labels.shape[0]),
        }
        batch_rows.append(row)

        if logger is not None and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader)):
            logger.info(
                "train epoch=%s batch=%s/%s loss=%.6f cls_loss=%.6f cafd_loss=%s cafd_weighted_mse=%s cafd_correlation=%s cafd_diagonal_gap=%s cafd_relation_kl=%s cafd_plain_mse=%s logits_kd_loss=%s logits_kd_weighted=%s lambda_logits=%s rsd_loss=%s rsd_weighted=%s lambda_rsd=%s rsd_diag_mean=%s rsd_offdiag_abs_mean=%s teacher_acc=%s acc=%.6f",
                epoch,
                batch_idx,
                len(loader),
                row["loss"],
                row["classification_loss"],
                "None" if row["cafd_loss"] is None else f"{row['cafd_loss']:.6f}",
                "None" if row["cafd_weighted_mse"] is None else f"{row['cafd_weighted_mse']:.6f}",
                "None" if row["cafd_correlation"] is None else f"{row['cafd_correlation']:.6f}",
                "None" if row["cafd_diagonal_gap"] is None else f"{row['cafd_diagonal_gap']:.6f}",
                "None" if row["cafd_relation_kl"] is None else f"{row['cafd_relation_kl']:.6f}",
                "None" if row["cafd_plain_mse"] is None else f"{row['cafd_plain_mse']:.6f}",
                "None" if row["logits_kd_loss"] is None else f"{row['logits_kd_loss']:.6f}",
                "None" if row["logits_kd_weighted_loss"] is None else f"{row['logits_kd_weighted_loss']:.6f}",
                "None" if row["lambda_logits_effective"] is None else f"{row['lambda_logits_effective']:.6f}",
                "None" if row["rsd_loss"] is None else f"{row['rsd_loss']:.6f}",
                "None" if row["rsd_weighted_loss"] is None else f"{row['rsd_weighted_loss']:.6f}",
                "None" if row["lambda_rsd_effective"] is None else f"{row['lambda_rsd_effective']:.6f}",
                "None" if row["rsd_diag_mean"] is None else f"{row['rsd_diag_mean']:.6f}",
                "None" if row["rsd_offdiag_abs_mean"] is None else f"{row['rsd_offdiag_abs_mean']:.6f}",
                "None" if row["teacher_accuracy"] is None else f"{row['teacher_accuracy']:.6f}",
                row["accuracy"],
            )

    if batch_csv_path is not None:
        append_csv_rows(
            batch_csv_path,
            batch_rows,
            [
                "epoch",
                "batch",
                "samples_seen",
                "loss",
                "classification_loss",
                "cafd_loss",
                "cafd_weighted_mse",
                "cafd_correlation",
                "cafd_diagonal_gap",
                "cafd_relation_kl",
                "cafd_plain_mse",
                "logits_kd_loss",
                "logits_kd_weighted_loss",
                "lambda_logits_effective",
                "rsd_loss",
                "rsd_weighted_loss",
                "lambda_rsd_effective",
                "rsd_invariance",
                "rsd_decorrelation",
                "rsd_diag_mean",
                "rsd_offdiag_abs_mean",
                "teacher_accuracy",
                "accuracy",
                "batch_size",
            ],
        )

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, stage):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch in loader:
        wifi = batch["wifi"].float().to(device)
        labels = batch["label"].to(device)
        batch_size = int(labels.shape[0])
        if stage == "v0":
            logits = model(wifi)
        else:
            video = batch["video"].float().to(device)
            logits = model(wifi, video)["logits"]
        loss = classification_loss(logits, labels, "single_ce")
        total_loss += float(loss.item()) * batch_size
        total_correct += accuracy_top1(logits, labels) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def collect_predictions(model, loader, device, stage):
    model.eval()
    rows = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in loader:
        wifi = batch["wifi"].float().to(device)
        labels = batch["label"].to(device)
        batch_size = int(labels.shape[0])
        teacher_logits = None
        if stage == "v0":
            logits = model(wifi)
        else:
            video = batch["video"].float().to(device)
            outputs = model(wifi, video)
            logits = outputs["logits"]
            teacher_logits = outputs.get("teacher_logits")

        loss = classification_loss(logits, labels, "single_ce")
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        correct = pred_ids.eq(labels.long())
        teacher_probs = None
        teacher_pred_ids = None
        teacher_correct = None
        if teacher_logits is not None:
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
            teacher_pred_ids = teacher_probs.argmax(dim=-1)
            teacher_correct = teacher_pred_ids.eq(labels.long())
        total_loss += float(loss.item()) * batch_size
        total_correct += int(correct.long().sum().item())
        total_samples += batch_size

        sample_ids = batch["sample_id"]
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]

        for item_idx, sample_id in enumerate(sample_ids):
            true_id = int(labels[item_idx].detach().cpu().item())
            pred_id = int(pred_ids[item_idx].detach().cpu().item())
            item_probs = probs[item_idx].detach().cpu().tolist()
            row = {
                "sample_id": sample_id,
                "true_id": true_id,
                "true_activity": ID_TO_ACTIVITY[true_id],
                "pred_id": pred_id,
                "pred_activity": ID_TO_ACTIVITY[pred_id],
                "correct": int(bool(correct[item_idx].detach().cpu().item())),
                "pred_probability": float(item_probs[pred_id]),
                "true_probability": float(item_probs[true_id]),
                "loss": float(loss.item()),
            }
            if teacher_probs is not None:
                teacher_pred_id = int(teacher_pred_ids[item_idx].detach().cpu().item())
                teacher_item_probs = teacher_probs[item_idx].detach().cpu().tolist()
                row.update(
                    {
                        "teacher_pred_id": teacher_pred_id,
                        "teacher_pred_activity": ID_TO_ACTIVITY[teacher_pred_id],
                        "teacher_correct": int(bool(teacher_correct[item_idx].detach().cpu().item())),
                        "teacher_pred_probability": float(teacher_item_probs[teacher_pred_id]),
                        "teacher_true_probability": float(teacher_item_probs[true_id]),
                    }
                )
            for class_id, class_name in ID_TO_ACTIVITY.items():
                row[f"prob_{class_id}_{class_name}"] = float(item_probs[class_id])
            if teacher_probs is not None:
                for class_id, class_name in ID_TO_ACTIVITY.items():
                    row[f"teacher_prob_{class_id}_{class_name}"] = float(teacher_item_probs[class_id])
            rows.append(row)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), rows


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.sample_limit is not None:
        cfg["data"]["sample_limit"] = args.sample_limit
    if args.num_frames is not None:
        cfg["video"]["num_frames"] = args.num_frames
    if args.s3d_weights is not None:
        cfg["video"]["s3d_weights"] = args.s3d_weights
    if args.teacher_checkpoint is not None:
        cfg["video"]["teacher_checkpoint"] = args.teacher_checkpoint
    if args.projector_target is not None:
        cfg.setdefault("projector", {})["target"] = args.projector_target
    if args.lambda_cafd is not None:
        cfg.setdefault("cafd", {})["lambda_cafd"] = args.lambda_cafd
    if args.lambda_logits is not None:
        cfg.setdefault("logits_kd", {})["lambda_logits"] = args.lambda_logits
        cfg["logits_kd"]["enable"] = args.lambda_logits > 0
    if args.kd_temperature is not None:
        cfg.setdefault("logits_kd", {})["temperature"] = args.kd_temperature
    if args.kd_warmup_epochs is not None:
        cfg.setdefault("logits_kd", {})["warmup_epochs"] = args.kd_warmup_epochs
    if args.lambda_rsd is not None:
        cfg.setdefault("rsd", {})["lambda_rsd"] = args.lambda_rsd
        cfg["rsd"]["enable"] = args.lambda_rsd > 0
    if args.rsd_kappa is not None:
        cfg.setdefault("rsd", {})["kappa"] = args.rsd_kappa
    if args.rsd_warmup_epochs is not None:
        cfg.setdefault("rsd", {})["warmup_epochs"] = args.rsd_warmup_epochs
    seed_everything(int(cfg["experiment"]["seed"]))
    device = select_device(cfg["train"]["device"])

    run_dir = create_run_dir(
        PROJECT_ROOT,
        cfg["experiment"]["output_dir"],
        cfg["experiment"]["name"],
        args.stage,
    )
    logger = setup_run_logger(run_dir / "train.log")
    logger.info("run_dir=%s", run_dir)
    logger.info("stage=%s", args.stage)
    logger.info("device=%s", device)
    logger.info("effective_config:\n%s", yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    save_yaml(run_dir / "config.yaml", cfg)

    train_loader, val_loader, train_df, val_df = build_loaders(cfg, use_video=args.stage == "v1")
    train_df.to_csv(run_dir / "splits" / "train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(run_dir / "splits" / "val.csv", index=False, encoding="utf-8-sig")
    logger.info("dataset_split train=%s val=%s", len(train_df), len(val_df))
    logger.info(
        "dataloader train_batches=%s val_batches=%s train_drop_last=%s batch_size=%s",
        len(train_loader),
        len(val_loader),
        getattr(train_loader, "drop_last", False),
        int(cfg["train"]["batch_size"]),
    )
    logger.info("saved_train_split=%s", run_dir / "splits" / "train.csv")
    logger.info("saved_val_split=%s", run_dir / "splits" / "val.csv")

    model = build_model(cfg, args.stage)
    if args.stage == "v1" and getattr(model.video_teacher, "checkpoint_path", None):
        logger.info("loaded_video_teacher_checkpoint=%s", model.video_teacher.checkpoint_path)
        logger.info("video_teacher_checkpoint_extra=%s", model.video_teacher.checkpoint_extra)
        logger.info("video_teacher_checkpoint_load_info=%s", model.video_teacher.checkpoint_load_info)
        logger.info("video_projector_checkpoint_load_info=%s", model.video_projector_checkpoint_load_info)
        logger.info("projector_classifier_checkpoint_load_info=%s", model.projector_classifier_checkpoint_load_info)
    if args.stage == "v1":
        logger.info(
            "projector target=%s freeze_video_projector=%s use_projector_logits=%s teacher_logits_source=%s wifi_projector_out_dim=%d video_projector_trainable_params=%d",
            getattr(model, "projector_target", "unknown"),
            getattr(model, "freeze_video_projector", None),
            getattr(model, "use_projector_logits_requested", None),
            getattr(model, "teacher_logits_source", "unknown"),
            model.wifi_projector.fc_out.out_features,
            sum(p.numel() for p in model.video_projector.parameters() if p.requires_grad),
        )
    model_text = str(model)
    (run_dir / "model.txt").write_text(model_text, encoding="utf-8")
    logger.info("model_structure:\n%s", model_text)
    if bool(cfg.get("logging", {}).get("compute_flops", True)):
        model_summary = build_model_summary(model, args.stage, cfg)
    else:
        model_summary = {"stage": args.stage, "flops": {"available": False, "error": "disabled by config"}}
    save_yaml(run_dir / "model_summary.yaml", model_summary)
    logger.info("model_summary:\n%s", yaml.safe_dump(model_summary, allow_unicode=True, sort_keys=False))
    model = model.to(device)

    class_names = [ID_TO_ACTIVITY[class_id] for class_id in sorted(ID_TO_ACTIVITY)]
    kd_cfg = get_logits_kd_cfg(cfg)
    rsd_cfg = get_rsd_cfg(cfg)
    if args.stage == "v0":
        model_name = "xfi_resnet18"
    else:
        model_parts = ["xfi_resnet18", "s3d", "cafd"]
        if kd_cfg["enabled"]:
            model_parts.append("logit_kd")
        if rsd_cfg["enabled"]:
            model_parts.append("rsd")
        model_name = "_with_".join(model_parts)
    result_payload = build_wimans_result_payload(
        model_name=model_name,
        task=f"{args.stage}_single_person_har",
        cfg=cfg,
        model_summary=model_summary,
    )
    epoch_results = []

    optimizer = build_optimizer(model, cfg, args.stage, logger=logger)
    scheduler = build_scheduler(optimizer, cfg, logger=logger)

    cafd_loss_fn = None
    rsd_loss_fn = None
    if args.stage == "v1":
        cafd_loss_fn = CAFDLoss(
            temperature=float(cfg["cafd"]["temperature"]),
        )
        logger.info(
            "cafd enabled=%s lambda=%.4f temperature=%.4f formula=weighted_mse_plus_diagonal_gap correlation=disabled",
            bool(cfg["cafd"].get("enable", True)),
            float(cfg["cafd"]["lambda_cafd"]),
            float(cfg["cafd"]["temperature"]),
        )
        logger.info(
            "logits_kd enabled=%s lambda=%.4f temperature=%.4f warmup_epochs=%d confidence_threshold=%.4f",
            kd_cfg["enabled"],
            kd_cfg["lambda_logits"],
            kd_cfg["temperature"],
            kd_cfg["warmup_epochs"],
            kd_cfg["confidence_threshold"],
        )
        rsd_loss_fn = RSDLoss(
            kappa=rsd_cfg["kappa"],
            reduction=rsd_cfg["reduction"],
        )
        logger.info(
            "rsd enabled=%s source=%s lambda=%.6f kappa=%.6f warmup_epochs=%d reduction=%s",
            rsd_cfg["enabled"],
            rsd_cfg["source"],
            rsd_cfg["lambda_rsd"],
            rsd_cfg["kappa"],
            rsd_cfg["warmup_epochs"],
            rsd_cfg["reduction"],
        )

    best_acc = -1.0
    checkpoint_dir = run_dir / "checkpoints"
    epoch_rows = []
    epoch_csv_fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "lr_backbone", "lr_head_projector"]

    for epoch in range(int(cfg["train"]["epochs"])):
        # Log current LRs before each epoch so we can see when scheduler fires.
        current_lrs = get_current_lrs(optimizer)
        logger.info(
            "epoch=%d lr_backbone=%.2e lr_head_projector=%.2e",
            epoch + 1,
            current_lrs.get("backbone", float("nan")),
            current_lrs.get("head_projector", float("nan")),
        )

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.stage,
            cfg,
            epoch=epoch + 1,
            cafd_loss_fn=cafd_loss_fn,
            rsd_loss_fn=rsd_loss_fn,
            logger=logger,
            batch_csv_path=run_dir / "metrics" / "train_batches.csv",
        )
        val_loss, val_acc, prediction_rows = collect_predictions(model, val_loader, device, args.stage)

        # Step scheduler based on val_acc (higher is better, mode='max').
        scheduler.step(val_acc)

        prediction_fieldnames = list(prediction_rows[0].keys()) if prediction_rows else []
        if prediction_rows:
            prediction_path = run_dir / "splits" / f"val_predictions_epoch_{epoch + 1:03d}.csv"
            append_csv_rows(prediction_path, prediction_rows, prediction_fieldnames)
            logger.info("saved_val_predictions=%s", prediction_path)

        message = (
            f"epoch={epoch + 1} train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f}"
        )
        print(message)
        logger.info(message)

        # Log if scheduler reduced any LR this epoch.
        new_lrs = get_current_lrs(optimizer)
        for group_name, new_lr in new_lrs.items():
            old_lr = current_lrs.get(group_name, new_lr)
            if new_lr < old_lr - 1e-12:
                logger.info(
                    "scheduler reduced lr: %s %.2e -> %.2e",
                    group_name,
                    old_lr,
                    new_lr,
                )

        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": max(best_acc, val_acc),
                "lr_backbone": new_lrs.get("backbone", float("nan")),
                "lr_head_projector": new_lrs.get("head_projector", float("nan")),
            }
        )
        append_csv_rows(
            run_dir / "metrics" / "epochs.csv",
            [epoch_rows[-1]],
            epoch_csv_fieldnames,
        )
        epoch_results.append(
            build_epoch_result(
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                prediction_rows,
                class_names,
                split_df=val_df,
                lrs=new_lrs,
            )
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                str(checkpoint_dir / "best.pt"),
                model,
                optimizer,
                extra={"epoch": epoch + 1, "val_acc": best_acc, "stage": args.stage},
            )
            if prediction_rows:
                best_prediction_path = run_dir / "splits" / "val_predictions_best.csv"
                best_prediction_path.unlink(missing_ok=True)
                append_csv_rows(best_prediction_path, prediction_rows, prediction_fieldnames)
                logger.info("saved_best_val_predictions=%s", best_prediction_path)
            logger.info("saved_best_checkpoint=%s val_acc=%.6f", checkpoint_dir / "best.pt", best_acc)

        update_result_payload(result_payload, epoch_results)
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
