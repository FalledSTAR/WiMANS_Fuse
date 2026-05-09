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
from losses import CAFDLoss, classification_loss  # noqa: E402
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


def run_epoch(model, loader, optimizer, device, stage, cfg, epoch: int, cafd_loss_fn=None, logger=None, batch_csv_path=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batch_rows = []
    log_interval = max(int(cfg["train"].get("log_interval", 50)), 1)
    for batch_idx, batch in enumerate(loader, start=1):
        wifi = batch["wifi"].float().to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if stage == "v0":
            logits = model(wifi)
            loss = classification_loss(logits, labels, "single_ce")
            cls_loss_value = float(loss.item())
            cafd_loss_value = None
        else:
            video = batch["video"].float().to(device)
            outputs = model(wifi, video)
            logits = outputs["logits"]
            cls_loss = classification_loss(logits, labels, "single_ce")
            cafd_loss = cafd_loss_fn(outputs["wifi_projected"], outputs["video_projected"])
            loss = cls_loss + float(cfg["cafd"]["lambda_cafd"]) * cafd_loss
            cls_loss_value = float(cls_loss.item())
            cafd_loss_value = float(cafd_loss.item())

        loss.backward()
        optimizer.step()
        batch_acc = accuracy_top1(logits.detach(), labels.detach())
        total_loss += float(loss.item())
        total_acc += batch_acc

        row = {
            "epoch": epoch,
            "batch": batch_idx,
            "samples_seen": batch_idx * int(cfg["train"]["batch_size"]),
            "loss": float(loss.item()),
            "classification_loss": cls_loss_value,
            "cafd_loss": cafd_loss_value,
            "accuracy": batch_acc,
            "batch_size": int(labels.shape[0]),
        }
        batch_rows.append(row)

        if logger is not None and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader)):
            logger.info(
                "train epoch=%s batch=%s/%s loss=%.6f cls_loss=%.6f cafd_loss=%s acc=%.6f",
                epoch,
                batch_idx,
                len(loader),
                row["loss"],
                row["classification_loss"],
                "None" if row["cafd_loss"] is None else f"{row['cafd_loss']:.6f}",
                row["accuracy"],
            )

    if batch_csv_path is not None:
        append_csv_rows(
            batch_csv_path,
            batch_rows,
            ["epoch", "batch", "samples_seen", "loss", "classification_loss", "cafd_loss", "accuracy", "batch_size"],
        )

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


@torch.no_grad()
def collect_predictions(model, loader, device, stage):
    model.eval()
    rows = []
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
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        correct = pred_ids.eq(labels.long())
        total_loss += float(loss.item())
        total_acc += correct.float().mean().item()

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
            for class_id, class_name in ID_TO_ACTIVITY.items():
                row[f"prob_{class_id}_{class_name}"] = float(item_probs[class_id])
            rows.append(row)

    return total_loss / max(len(loader), 1), total_acc / max(len(loader), 1), rows


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
    logger.info("saved_train_split=%s", run_dir / "splits" / "train.csv")
    logger.info("saved_val_split=%s", run_dir / "splits" / "val.csv")

    model = build_model(cfg, args.stage)
    if args.stage == "v1" and getattr(model.video_teacher, "checkpoint_path", None):
        logger.info("loaded_video_teacher_checkpoint=%s", model.video_teacher.checkpoint_path)
        logger.info("video_teacher_checkpoint_extra=%s", model.video_teacher.checkpoint_extra)
        logger.info("video_teacher_checkpoint_load_info=%s", model.video_teacher.checkpoint_load_info)
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
    result_payload = build_wimans_result_payload(
        model_name="xfi_resnet18" if args.stage == "v0" else "xfi_resnet18_with_s3d_cafd",
        task=f"{args.stage}_single_person_har",
        cfg=cfg,
        model_summary=model_summary,
    )
    epoch_results = []

    optimizer = build_optimizer(model, cfg, args.stage, logger=logger)
    scheduler = build_scheduler(optimizer, cfg, logger=logger)

    cafd_loss_fn = None
    if args.stage == "v1":
        cafd_loss_fn = CAFDLoss(
            temperature=float(cfg["cafd"]["temperature"]),
            alpha=float(cfg["cafd"]["alpha"]),
            beta=float(cfg["cafd"]["beta"]),
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
