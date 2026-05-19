from pathlib import Path
from typing import Optional

import pandas as pd
from torch.utils.data import Dataset

from .constants import ACTIVITY_COLS
from .label_builder import build_multi_user_activity_label, build_multi_user_slot_label, build_single_user_label
from .video_loader import OnlineS3DVideoLoader
from .wifi_amp_loader import load_wifi_amplitude


def _as_list_or_none(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_annotation(annotation_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(annotation_path, dtype=str)
    required_cols = {"label", "environment", "wifi_band", "number_of_users", *ACTIVITY_COLS}
    missing = required_cols.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    return dataframe


def build_single_user_dataframe(
    annotation_path: str,
    wifi_band=None,
    environment=None,
    num_users=None,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    dataframe = load_annotation(annotation_path)

    wifi_band = _as_list_or_none(wifi_band)
    environment = _as_list_or_none(environment)
    num_users = _as_list_or_none(num_users)

    if wifi_band is not None:
        dataframe = dataframe[dataframe["wifi_band"].isin(wifi_band)]
    if environment is not None:
        dataframe = dataframe[dataframe["environment"].isin(environment)]
    if num_users is not None:
        dataframe = dataframe[dataframe["number_of_users"].isin(num_users)]

    dataframe = dataframe.copy().reset_index(drop=True)
    if sample_limit is not None:
        dataframe = dataframe.head(int(sample_limit)).copy().reset_index(drop=True)

    return dataframe


class WiMANSHARDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_root: str,
        label_mode: str = "single_ce",
        use_wifi: bool = True,
        use_video: bool = False,
        target_len: int = 3000,
        pad_mode: str = "left",
        truncate_mode: str = "tail",
        normalize: str = "log1p_zscore",
        video_num_frames: int = 90,
        video_transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.label_mode = label_mode
        self.use_wifi = use_wifi
        self.use_video = use_video
        self.target_len = target_len
        self.pad_mode = pad_mode
        self.truncate_mode = truncate_mode
        self.normalize = normalize
        self.video_loader = OnlineS3DVideoLoader(num_frames=video_num_frames, transform=video_transform) if use_video else None

    def __len__(self):
        return len(self.dataframe)

    def _paths_for_label(self, label: str):
        wifi_path = self.data_root / "wifi_csi" / "amp" / f"{label}.npy"
        video_path = self.data_root / "video" / f"{label}.mp4"
        return wifi_path, video_path

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        sample_label = row["label"]
        wifi_path, video_path = self._paths_for_label(sample_label)

        item = {
            "sample_id": sample_label,
        }

        if self.use_wifi:
            item["wifi"] = load_wifi_amplitude(
                str(wifi_path),
                target_len=self.target_len,
                pad_mode=self.pad_mode,
                truncate_mode=self.truncate_mode,
                normalize=self.normalize,
            )

        if self.label_mode == "single_ce":
            item["label"] = build_single_user_label(row)
        elif self.label_mode == "multi_bce":
            item["label"] = build_multi_user_activity_label(row)
        elif self.label_mode == "multi_slot_ce":
            item["label"] = build_multi_user_slot_label(row)
        else:
            raise ValueError(f"Unsupported label_mode: {self.label_mode}")

        if self.use_video:
            item["video"] = self.video_loader(str(video_path))

        return item
