import torch
from torch import nn

from .hybrid_projector import HybridProjector
from .s3d_teacher import S3DTeacher
from .xfi_wifi_resnet import XFiWiFiStudent


class RSDProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gamma: int = 2):
        super().__init__()
        hidden_dim = int(in_dim) * int(gamma)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"RSDProjector expects [B,D] features, got {tuple(x.shape)}")
        return self.net(x)


class VideoWiFiCAFDModel(nn.Module):
    def __init__(
        self,
        xfi_weight_path: str,
        num_classes: int = 9,
        s3d_weights: str = "kinetics400",
        teacher_checkpoint_path: str = None,
        freeze_s3d: bool = True,
        projector_hidden_dim: int = 256,
        projector_out_dim: int = 256,
        projector_num_heads: int = 2,
        projector_target: str = "video_feature",
        freeze_video_projector: bool = True,
        use_projector_logits: bool = True,
        projector_dropout: float = 0.2,
        rsd_gamma: int = 2,
    ):
        super().__init__()
        if projector_target not in {"video_feature", "projected"}:
            raise ValueError("projector_target must be 'video_feature' or 'projected'")
        self.projector_target = projector_target
        self.use_projector_logits_requested = bool(use_projector_logits)
        self.wifi_student = XFiWiFiStudent(weight_path=xfi_weight_path, num_classes=num_classes)
        self.video_teacher = S3DTeacher(
            weights=s3d_weights,
            freeze=freeze_s3d,
            checkpoint_path=teacher_checkpoint_path,
            num_classes=num_classes,
        )
        wifi_projector_out_dim = (
            self.video_teacher.output_dim if projector_target == "video_feature" else projector_out_dim
        )
        self.wifi_projector = HybridProjector(
            in_dim=self.wifi_student.feature_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=wifi_projector_out_dim,
            num_heads=projector_num_heads,
        )
        self.rsd_projector = RSDProjector(
            in_dim=self.wifi_student.feature_dim,
            out_dim=wifi_projector_out_dim,
            gamma=rsd_gamma,
        )
        self.video_projector = HybridProjector(
            in_dim=self.video_teacher.output_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
            num_heads=projector_num_heads,
        )
        self.projector_classifier = nn.Sequential(
            nn.Dropout(p=float(projector_dropout)),
            nn.Linear(projector_out_dim, num_classes),
        )
        self.video_projector_checkpoint_load_info = None
        self.projector_classifier_checkpoint_load_info = None
        self.projector_classifier_available = False
        if teacher_checkpoint_path is not None:
            self.video_projector_checkpoint_load_info = self._load_video_projector_checkpoint(teacher_checkpoint_path)
            self.projector_classifier_checkpoint_load_info = self._load_projector_classifier_checkpoint(
                teacher_checkpoint_path
            )
            self.projector_classifier_available = self.projector_classifier_checkpoint_load_info["loaded_keys"] > 0
        if self.projector_target == "projected":
            projector_loaded_keys = (
                0
                if self.video_projector_checkpoint_load_info is None
                else int(self.video_projector_checkpoint_load_info["loaded_keys"])
            )
            if projector_loaded_keys == 0:
                raise ValueError(
                    "projector_target='projected' requires a teacher checkpoint with video_projector.* weights"
                )
            if self.use_projector_logits_requested and not self.projector_classifier_available:
                raise ValueError(
                    "use_projector_logits=True requires a teacher checkpoint with projector_classifier.* weights"
                )
        self.freeze_video_projector = freeze_video_projector
        if self.freeze_video_projector:
            for parameter in self.video_projector.parameters():
                parameter.requires_grad = False
        for parameter in self.projector_classifier.parameters():
            parameter.requires_grad = False
        self.projector_classifier.eval()
        self.teacher_logits_source = (
            "projector_classifier"
            if self._should_use_projector_logits()
            else "s3d_classifier"
        )

    def _load_video_projector_checkpoint(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location="cpu")
        source_state = payload.get("model", payload)
        current_state = self.video_projector.state_dict()
        target_state = {}
        skipped = []
        for key, value in source_state.items():
            if key.startswith("video_projector."):
                key = key[len("video_projector."):]
            elif key.startswith("projector."):
                key = key[len("projector."):]
            else:
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
        load_result = self.video_projector.load_state_dict(target_state, strict=False)
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "loaded_keys": len(target_state),
            "skipped_keys": skipped,
        }

    def _load_projector_classifier_checkpoint(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location="cpu")
        source_state = payload.get("model", payload)
        current_state = self.projector_classifier.state_dict()
        target_state = {}
        skipped = []
        for key, value in source_state.items():
            if key.startswith("projector_classifier."):
                key = key[len("projector_classifier."):]
            else:
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
        load_result = self.projector_classifier.load_state_dict(target_state, strict=False)
        return {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "loaded_keys": len(target_state),
            "skipped_keys": skipped,
        }

    def _should_use_projector_logits(self) -> bool:
        return (
            self.projector_target == "projected"
            and self.use_projector_logits_requested
            and self.projector_classifier_available
        )

    def forward(self, wifi, video):
        wifi_out = self.wifi_student(wifi, return_features=True)
        video_out = self.video_teacher(video, return_logits=True)
        video_feature = video_out["feature"]
        wifi_projected = self.wifi_projector(wifi_out["tokens"])
        wifi_rsd_feature = self.rsd_projector(wifi_out["feature"])
        if self.freeze_video_projector:
            with torch.no_grad():
                video_projected = self.video_projector(video_feature)
        else:
            video_projected = self.video_projector(video_feature)
        if self.projector_target == "video_feature":
            teacher_distill_feature = video_feature
        else:
            teacher_distill_feature = video_projected
        projector_logits = None
        teacher_logits = video_out["logits"]
        if self._should_use_projector_logits():
            with torch.no_grad():
                projector_logits = self.projector_classifier(video_projected)
            teacher_logits = projector_logits
        return {
            "logits": wifi_out["logits"],
            "teacher_logits": teacher_logits,
            "s3d_logits": video_out["logits"],
            "projector_logits": projector_logits,
            "teacher_logits_source": self.teacher_logits_source,
            "wifi_feature": wifi_out["feature"],
            "video_feature": video_feature,
            "wifi_projected": wifi_projected,
            "wifi_rsd_feature": wifi_rsd_feature,
            "video_projected": video_projected,
            "wifi_distill_feature": wifi_projected,
            "teacher_distill_feature": teacher_distill_feature,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_video_projector:
            self.video_projector.eval()
        self.projector_classifier.eval()
        return self
