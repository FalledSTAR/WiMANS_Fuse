from pathlib import Path

import torch
import torchvision
from torchvision.models.video import S3D_Weights


def _sample_temporal(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    if num_frames is None or video.shape[1] == num_frames:
        return video
    if video.shape[1] < 1:
        raise ValueError("Video has no frames after transform")

    indices = torch.linspace(0, video.shape[1] - 1, steps=num_frames).long()
    return video[:, indices]


class OnlineS3DVideoLoader:
    def __init__(self, num_frames: int = 90):
        self.num_frames = num_frames
        self.transform = S3D_Weights.DEFAULT.transforms()

    def __call__(self, path: str) -> torch.Tensor:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)

        video, _, _ = torchvision.io.read_video(str(path_obj), output_format="TCHW", pts_unit="sec")
        video = self.transform(video)
        video = _sample_temporal(video, self.num_frames)
        return video.float()
