def _is_truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _activity_text(value):
    text = "" if value is None else str(value).strip()
    return "-" if text in {"", "empty_slot"} else text


def _format_float(value, digits=4):
    if value in {None, ""}:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def _compact_multi_bce_rows(rows):
    compact_rows = []
    for row in rows:
        slot_results = []
        wrong_slots = []
        true_slots = []
        pred_slots = []
        compact_row = {
            "sample_id": row.get("sample_id", ""),
            "result": "OK" if _is_truthy(row.get("exact_sample_correct")) else "ERR",
            "slot_correct": f"{row.get('slot_correct_count', '')}/{row.get('slot_total', '')}",
            "active_slot_correct": f"{row.get('active_slot_correct_count', '')}/{row.get('active_slot_total', '')}",
            "official_slot_acc": _format_float(row.get("official_slot_accuracy")),
            "active_slot_acc": _format_float(row.get("active_slot_accuracy")),
            "loss": _format_float(row.get("loss"), digits=6),
        }
        for slot_idx in range(1, 7):
            true_activity = _activity_text(row.get(f"slot_{slot_idx}_true_activity", ""))
            pred_activity = _activity_text(row.get(f"slot_{slot_idx}_pred_activity", ""))
            slot_ok = _is_truthy(row.get(f"slot_{slot_idx}_correct"))
            slot_result = "OK" if slot_ok else "ERR"
            true_slots.append(f"s{slot_idx}:{true_activity}")
            pred_slots.append(f"s{slot_idx}:{pred_activity}")
            slot_results.append(f"s{slot_idx}:{slot_result}")
            if not slot_ok:
                wrong_slots.append(f"s{slot_idx}:{true_activity}->{pred_activity}")
            compact_row[f"s{slot_idx}_true"] = true_activity
            compact_row[f"s{slot_idx}_pred"] = pred_activity
            compact_row[f"s{slot_idx}_result"] = slot_result
        compact_row["true_slots"] = "; ".join(true_slots)
        compact_row["pred_slots"] = "; ".join(pred_slots)
        compact_row["slot_results"] = "; ".join(slot_results)
        compact_row["wrong_slots"] = "; ".join(wrong_slots)
        compact_rows.append(compact_row)
    return compact_rows


def _compact_single_ce_rows(rows):
    compact_rows = []
    for row in rows:
        compact_row = {
            "sample_id": row.get("sample_id", ""),
            "result": "OK" if _is_truthy(row.get("correct")) else "ERR",
            "true_activity": row.get("true_activity", ""),
            "pred_activity": row.get("pred_activity", ""),
            "pred_probability": _format_float(row.get("pred_probability")),
            "true_probability": _format_float(row.get("true_probability")),
            "loss": _format_float(row.get("loss"), digits=6),
        }
        if "teacher_pred_activity" in row:
            compact_row["teacher_pred_activity"] = row.get("teacher_pred_activity", "")
            compact_row["teacher_result"] = "OK" if _is_truthy(row.get("teacher_correct")) else "ERR"
        compact_rows.append(compact_row)
    return compact_rows


def compact_prediction_rows(rows):
    if not rows:
        return []
    if rows[0].get("task_mode") == "multi_bce":
        return _compact_multi_bce_rows(rows)
    return _compact_single_ce_rows(rows)
