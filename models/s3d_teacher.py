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
    def __init__(
        self,
        weights: str = "kinetics400",
        freeze: bool = True,
        checkpoint_path: str = None,
        num_classes: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        init_weights = None if checkpoint_path else _resolve_s3d_weights(weights)
        self.model = s3d(weights=init_weights)
        self.num_classes = num_classes
        if self.num_classes is not None:
            self._replace_classifier(int(self.num_classes), float(dropout))
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

    def _replace_classifier(self, num_classes: int, dropout: float):
        in_channels = self.model.classifier[1].in_channels
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv3d(in_channels, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

    def _load_video_teacher_checkpoint(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location="cpu")
        source_state = payload.get("model", payload)
        current_state = self.model.state_dict()
        target_state = {}
        skipped = []
        for key, value in source_state.items():
            if key.startswith("video_teacher.model."):
                key = key[len("video_teacher.model."):]
            if key.startswith("model."):
                key = key[len("model."):]
            if (
                key.startswith("video_teacher.head_module.")
                or key.startswith("video_teacher.head_root.")
                or key.startswith("head_module.")
                or key.startswith("head_root.")
                or key.startswith("video_projector.")
                or key.startswith("projector_classifier.")
            ):
                continue
            if key not in current_state:
                skipped.append({"key": key, "reason": "not_in_target"})
                continue
            if tuple(current_state[key].shape) != tuple(value.shape):
                skipped.append(
                    {
                        "key": key,
                        "reason": "shape_mismatch",
                        "source_shape": tuple(value.shape),
                        "target_shape": tuple(current_state[key].shape),
                    }
                )
                continue
            target_state[key] = value

        load_result = self.model.load_state_dict(target_state, strict=False)
        load_info = {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "loaded_keys": len(target_state),
            "skipped_keys": skipped,
        }
        return payload.get("extra", {}), load_info

    @property
    def output_dim(self) -> int:
        return 1024

    def forward(self, video: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [B,C,T,H,W], got {tuple(video.shape)}")

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            features = self.model.features(video)
            feature_vector = features.mean(dim=(2, 3, 4))
            if return_logits:
                logits = self.model.avgpool(features)
                logits = self.model.classifier(logits)
                logits = torch.mean(logits, dim=(2, 3, 4))
                return {"feature": feature_vector, "logits": logits}
        return feature_vector

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self
