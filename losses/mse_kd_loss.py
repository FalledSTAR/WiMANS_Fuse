import torch
import torch.nn.functional as F
from torch import nn


class FeatureMSEKDLoss(nn.Module):
    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = bool(normalize)

    def forward(self, student_feature: torch.Tensor, teacher_feature: torch.Tensor) -> torch.Tensor:
        if student_feature.shape != teacher_feature.shape:
            raise ValueError(
                "FeatureMSEKDLoss expects matching feature shapes, "
                f"got student={tuple(student_feature.shape)} teacher={tuple(teacher_feature.shape)}"
            )
        teacher_feature = teacher_feature.detach()
        if self.normalize:
            student_feature = F.normalize(student_feature, dim=-1)
            teacher_feature = F.normalize(teacher_feature, dim=-1)
        return F.mse_loss(student_feature, teacher_feature)
