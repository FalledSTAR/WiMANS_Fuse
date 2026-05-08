import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.constants import ACTIVITY_COLS  # noqa: E402
from datasets import build_single_user_dataframe  # noqa: E402
from datasets.label_builder import build_single_user_label  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    annotation = resolve_path(PROJECT_ROOT, cfg["data"]["annotation"])
    data_root = Path(resolve_path(PROJECT_ROOT, cfg["data"]["root"]))

    dataframe = pd.read_csv(annotation, dtype=str)
    print(f"annotation_rows={len(dataframe)}")
    if len(dataframe) != 11286:
        raise AssertionError(f"Expected 11286 annotation rows, got {len(dataframe)}")

    selected = build_single_user_dataframe(
        annotation,
        wifi_band=cfg["data"]["wifi_band"],
        environment=cfg["data"]["environment"],
        num_users=cfg["data"]["num_users"],
        sample_limit=cfg["data"]["sample_limit"],
    )
    print(f"filters wifi_band={cfg['data']['wifi_band']} environment={cfg['data']['environment']} num_users={cfg['data']['num_users']}")
    print(f"selected_rows={len(selected)}")
    if len(selected) == 0:
        raise AssertionError("Selected data is empty")

    non_null_counts = selected[ACTIVITY_COLS].notna().sum(axis=1)
    if not (non_null_counts == 1).all():
        raise AssertionError("Every selected row must have exactly one non-null activity")

    labels = selected.apply(build_single_user_label, axis=1)
    class_counts = labels.value_counts().sort_index().to_dict()
    print(f"class_counts={class_counts}")
    if len(class_counts) != 9:
        raise AssertionError(f"Expected 9 activity classes, got {class_counts}")
    if cfg["data"]["sample_limit"] is None and len(set(class_counts.values())) != 1:
        raise AssertionError(f"Expected balanced classes for the selected split, got {class_counts}")

    missing_wifi = []
    missing_video = []
    for sample_id in selected["label"].tolist():
        if not (data_root / "wifi_csi" / "amp" / f"{sample_id}.npy").exists():
            missing_wifi.append(sample_id)
        if not (data_root / "video" / f"{sample_id}.mp4").exists():
            missing_video.append(sample_id)

    print(f"missing_wifi={len(missing_wifi)}")
    print(f"missing_video={len(missing_video)}")
    if missing_wifi or missing_video:
        raise AssertionError(f"Missing files: wifi={missing_wifi[:5]}, video={missing_video[:5]}")

    print("DATA_CHECK_OK")


if __name__ == "__main__":
    main()
