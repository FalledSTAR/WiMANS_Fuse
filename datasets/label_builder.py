import numpy as np
import pandas as pd

from .constants import ACTIVITY_COLS, ACTIVITY_TO_ID


def _is_present_activity(value) -> bool:
    return pd.notna(value) and str(value).lower() != "nan"


def build_single_user_label(row) -> int:
    activities = []
    for col in ACTIVITY_COLS:
        value = row[col]
        if _is_present_activity(value):
            activities.append(str(value))

    if len(activities) != 1:
        raise ValueError(f"Expected exactly one active user, got {activities}")

    activity = activities[0]
    if activity not in ACTIVITY_TO_ID:
        raise KeyError(f"Unknown activity label: {activity}")
    return ACTIVITY_TO_ID[activity]


def build_multi_user_activity_label(row) -> np.ndarray:
    label = np.zeros((6, 9), dtype=np.float32)
    for slot_idx, col in enumerate(ACTIVITY_COLS):
        value = row[col]
        if not _is_present_activity(value):
            continue

        activity = str(value)
        if activity not in ACTIVITY_TO_ID:
            raise KeyError(f"Unknown activity label: {activity}")
        label[slot_idx, ACTIVITY_TO_ID[activity]] = 1.0

    return label
