import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def _to_builtin(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    return value


def attach_split_metadata(prediction_rows, split_df, fields=("environment",)):
    if split_df is None or not prediction_rows:
        return prediction_rows

    metadata = {}
    for _, row in split_df.iterrows():
        label = str(row.get("label", ""))
        metadata[label] = {field: row.get(field, None) for field in fields if field in row}

    enriched = []
    for row in prediction_rows:
        item = dict(row)
        item.update(metadata.get(str(item.get("sample_id", "")), {}))
        enriched.append(item)
    return enriched


def _labels_from_rows(prediction_rows):
    y_true = [int(row["true_id"]) for row in prediction_rows]
    y_pred = [int(row["pred_id"]) for row in prediction_rows]
    return y_true, y_pred


def _group_accuracy(prediction_rows, group_key):
    groups = {}
    for row in prediction_rows:
        group = row.get(group_key)
        if group is None or group == "":
            continue
        groups.setdefault(str(group), {"correct": 0, "total": 0})
        groups[str(group)]["total"] += 1
        groups[str(group)]["correct"] += int(row.get("correct", 0))

    result = {}
    for group, stats in sorted(groups.items()):
        total = stats["total"]
        correct = stats["correct"]
        result[group] = {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total else 0.0,
        }
    return result


def _multi_label_arrays(rows, num_slots=6, num_classes=9):
    y_true = []
    y_pred = []
    for row in rows:
        for slot_idx in range(num_slots):
            true_vec = [int(row.get(f"true_s{slot_idx + 1}_c{class_idx}", 0)) for class_idx in range(num_classes)]
            pred_vec = [int(row.get(f"pred_s{slot_idx + 1}_c{class_idx}", 0)) for class_idx in range(num_classes)]
            y_true.append(true_vec)
            y_pred.append(pred_vec)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def _multi_class_accuracy(rows, num_slots=6):
    groups = {}
    for row in rows:
        for slot_idx in range(num_slots):
            key = str(row.get(f"slot_{slot_idx + 1}_true_activity", "empty_slot"))
            groups.setdefault(key, {"correct": 0, "total": 0})
            groups[key]["total"] += 1
            groups[key]["correct"] += int(row.get(f"slot_{slot_idx + 1}_correct", 0))

    result = {}
    for group, stats in sorted(groups.items()):
        total = stats["total"]
        correct = stats["correct"]
        result[group] = {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total else 0.0,
        }
    return result


def _multi_environment_accuracy(rows):
    groups = {}
    for row in rows:
        group = row.get("environment")
        if group is None or group == "":
            continue
        groups.setdefault(str(group), {"correct": 0, "total": 0})
        groups[str(group)]["correct"] += int(row.get("slot_correct_count", 0))
        groups[str(group)]["total"] += int(row.get("slot_total", 0))

    result = {}
    for group, stats in sorted(groups.items()):
        total = stats["total"]
        correct = stats["correct"]
        result[group] = {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total else 0.0,
        }
    return result


def summarize_multi_bce_prediction_rows(prediction_rows, class_names, split_df=None):
    rows = attach_split_metadata(prediction_rows, split_df)
    if not rows:
        return {
            "accuracy": 0.0,
            "official_slot_acc": 0.0,
            "sample_exact_acc": 0.0,
            "active_slot_acc": 0.0,
            "classification_report": {},
            "class_accuracy": {},
            "environment_accuracy": {},
            "num_samples": 0,
        }

    slot_correct = sum(int(row.get("slot_correct_count", 0)) for row in rows)
    slot_total = sum(int(row.get("slot_total", 0)) for row in rows)
    active_correct = sum(int(row.get("active_slot_correct_count", 0)) for row in rows)
    active_total = sum(int(row.get("active_slot_total", 0)) for row in rows)
    sample_exact = sum(int(row.get("exact_sample_correct", 0)) for row in rows)
    y_true, y_pred = _multi_label_arrays(rows, num_classes=len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=6,
        zero_division=0,
        output_dict=True,
    )
    official_slot_acc = float(slot_correct / slot_total) if slot_total else 0.0
    return {
        "accuracy": official_slot_acc,
        "official_slot_acc": official_slot_acc,
        "sample_exact_acc": float(sample_exact / len(rows)) if rows else 0.0,
        "active_slot_acc": float(active_correct / active_total) if active_total else 0.0,
        "classification_report": report,
        "class_accuracy": _multi_class_accuracy(rows),
        "environment_accuracy": _multi_environment_accuracy(rows),
        "num_samples": len(rows),
    }


def summarize_prediction_rows(prediction_rows, class_names, split_df=None):
    rows = attach_split_metadata(prediction_rows, split_df)
    if not rows:
        return {
            "accuracy": 0.0,
            "classification_report": {},
            "class_accuracy": {},
            "environment_accuracy": {},
            "num_samples": 0,
        }

    if str(rows[0].get("task_mode", "")) == "multi_bce":
        return summarize_multi_bce_prediction_rows(rows, class_names, split_df=None)

    labels = list(range(len(class_names)))
    y_true, y_pred = _labels_from_rows(rows)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=6,
        zero_division=0,
        output_dict=True,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "class_accuracy": _group_accuracy(rows, "true_activity"),
        "environment_accuracy": _group_accuracy(rows, "environment"),
        "num_samples": len(rows),
    }


def build_epoch_result(epoch, train_loss, train_acc, val_loss, val_acc, prediction_rows, class_names, split_df=None, lrs=None):
    summary = summarize_prediction_rows(prediction_rows, class_names, split_df=split_df)
    result = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "accuracy": float(summary["accuracy"]),
        "num_val_samples": int(summary["num_samples"]),
        "classification_report": summary["classification_report"],
        "class_accuracy": summary["class_accuracy"],
        "environment_accuracy": summary["environment_accuracy"],
    }
    for key in ("official_slot_acc", "sample_exact_acc", "active_slot_acc"):
        if key in summary:
            result[key] = float(summary[key])
    if lrs is not None:
        result["lr"] = {str(key): float(value) for key, value in lrs.items()}
    return result


def best_epoch_result(epoch_results):
    if not epoch_results:
        return None
    return sorted(
        epoch_results,
        key=lambda item: (-float(item["val_acc"]), float(item["val_loss"]), -int(item["epoch"])),
    )[0]


def build_wimans_result_payload(model_name, task, cfg, model_summary=None):
    return {
        "model": model_name,
        "task": task,
        "data": cfg.get("data", {}),
        "nn": cfg.get("train", {}),
        "video": cfg.get("video", {}),
        "projector": cfg.get("projector", {}),
        "cafd": cfg.get("cafd", {}),
        "complexity": (model_summary or {}).get("flops", {}),
        "epochs": [],
        "accuracy": {"avg": 0.0, "std": 0.0},
    }


def update_result_payload(payload, epoch_results, top_checkpoints=None):
    payload["epochs"] = epoch_results
    best = best_epoch_result(epoch_results)
    if best is not None:
        payload["best_epoch"] = best["epoch"]
        payload["best_val_acc"] = best["val_acc"]
        payload["best_val_loss"] = best["val_loss"]
        payload["repeat_0"] = best["classification_report"]
        payload["accuracy"] = {"avg": best["accuracy"], "std": 0.0}
        payload["best_class_accuracy"] = best.get("class_accuracy", {})
        payload["best_environment_accuracy"] = best.get("environment_accuracy", {})
        for key in ("official_slot_acc", "sample_exact_acc", "active_slot_acc"):
            if key in best:
                payload[f"best_{key}"] = best[key]
    if top_checkpoints is not None:
        payload["top_checkpoints"] = [
            {
                "rank": rank,
                "epoch": int(item["epoch"]),
                "val_acc": float(item["val_acc"]),
                "val_loss": float(item["val_loss"]),
                "checkpoint": Path(item["checkpoint"]).name,
                "predictions": Path(item["predictions"]).name if item.get("predictions") is not None else "",
            }
            for rank, item in enumerate(top_checkpoints, start=1)
        ]
    return payload


def write_result_json(path, payload):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(_to_builtin(payload), indent=4, ensure_ascii=False), encoding="utf-8")
