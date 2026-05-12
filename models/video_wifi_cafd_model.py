import torch
from torch import nn

from .hybrid_projector import HybridProjector
from .s3d_teacher import S3DTeacher
from .xfi_wifi_resnet import XFiWiFiStudent


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
    ):
        super().__init__()
        if projector_target not in {"video_feature", "projected"}:
            raise ValueError("projector_target must be 'video_feature' or 'projected'")
        self.projector_target = projector_target
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
        self.video_projector = HybridProjector(
            in_dim=self.video_teacher.output_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
            num_heads=projector_num_heads,
        )
        self.freeze_video_projector = freeze_video_projector
        if self.freeze_video_projector:
            for parameter in self.video_projector.parameters():
                parameter.requires_grad = False

    def forward(self, wifi, video):
        wifi_out = self.wifi_student(wifi, return_features=True)
        video_out = self.video_teacher(video, return_logits=True)
        video_feature = video_out["feature"]
        wifi_projected = self.wifi_projector(wifi_out["tokens"])
        if self.freeze_video_projector:
            with torch.no_grad():
                video_projected = self.video_projector(video_feature)
        else:
            video_projected = self.video_projector(video_feature)
        if self.projector_target == "video_feature":
            teacher_distill_feature = video_feature
        else:
            teacher_distill_feature = video_projected
        return {
            "logits": wifi_out["logits"],
            "teacher_logits": video_out["logits"],
            "wifi_feature": wifi_out["feature"],
            "video_feature": video_feature,
            "wifi_projected": wifi_projected,
            "video_projected": video_projected,
            "wifi_distill_feature": wifi_projected,
            "teacher_distill_feature": teacher_distill_feature,
        }
