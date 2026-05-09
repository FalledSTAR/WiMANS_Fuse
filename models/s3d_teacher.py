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
    def __init__(self, weights: str = "kinetics400", freeze: bool = True, checkpoint_path: str = None):
        super().__init__()
        init_weights = None if checkpoint_path else _resolve_s3d_weights(weights)
        self.model = s3d(weights=init_weights)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_extra = None
        self.checkpoint_load_info = None
        if checkpoint_path:
            self.checkpoint_extra, self.checkpoint_load_info = self._load_video_teacher_checkpoint(checkpoint_path)
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.freeze = freeze
        if self.freeze:
            self.model.eval()

    def _load_video_teacher_checkpoint(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location="cpu")
        source_state = payload.get("model", payload)
        target_state = {}
        for key, value in source_state.items():
            if key.startswith("model."):
                key = key[len("model."):]
            if key.startswith("classifier."):
                continue
            if key.startswith("head_module.") or key.startswith("head_root."):
                continue
            target_state[key] = value

        load_result = self.model.load_state_dict(target_state, strict=False)
        load_info = {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "loaded_keys": len(target_state),
        }
        return payload.get("extra", {}), load_info

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
