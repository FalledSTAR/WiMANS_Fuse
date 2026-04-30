import torch
from torch import nn
from torchvision.models.video import S3D_Weights, s3d


def _resolve_s3d_weights(name: str):
    if name in (None, "none", "random"):
        return None
    if name in ("default", "kinetics400", "KINETICS400_V1"):
        return S3D_Weights.KINETICS400_V1
    raise ValueError(f"Unsupported S3D weights: {name}")


class S3DTeacher(nn.Module):
    def __init__(self, weights: str = "kinetics400", freeze: bool = True):
        super().__init__()
        self.model = s3d(weights=_resolve_s3d_weights(weights))
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.freeze = freeze
        if self.freeze:
            self.model.eval()

    @property
    def output_dim(self) -> int:
        return 1024

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [B,C,T,H,W], got {tuple(video.shape)}")

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            features = self.model.features(video)
            features = features.mean(dim=(2, 3, 4))
        return features

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self
