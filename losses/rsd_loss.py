import torch
from torch import nn


def normalize_batch(feature: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (feature - feature.mean(dim=0, keepdim=True)) / (
        feature.std(dim=0, unbiased=False, keepdim=True) + eps
    )


class RSDLoss(nn.Module):
    """
    Redundancy Suppression Distillation.

    This follows the RSD public implementation: maximize teacher-student
    cross-correlation on the diagonal and suppress off-diagonal redundancy.
    """

    def __init__(self, kappa: float = 0.01, reduction: str = "sum", eps: float = 1e-5):
        super().__init__()
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.kappa = float(kappa)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, return_details: bool = False):
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                "RSDLoss requires matching feature shapes, "
                f"got student={tuple(student_feat.shape)} teacher={tuple(teacher_feat.shape)}"
            )
        if student_feat.ndim != 2:
            raise ValueError(f"RSDLoss expects [B,D] features, got {tuple(student_feat.shape)}")

        batch_size, feature_dim = student_feat.shape
        if batch_size < 2:
            zero = student_feat.sum() * 0.0
            details = {
                "invariance": zero.detach(),
                "decorrelation": zero.detach(),
                "diag_mean": zero.detach(),
                "offdiag_abs_mean": zero.detach(),
            }
            return (zero, details) if return_details else zero

        student = normalize_batch(student_feat, self.eps)
        teacher = normalize_batch(teacher_feat.detach(), self.eps)
        rcc = torch.matmul(teacher.t(), student) / float(batch_size)

        identity = torch.eye(feature_dim, device=rcc.device, dtype=rcc.dtype)
        diagonal_mask = identity.bool()
        off_diagonal_mask = ~diagonal_mask
        diff_square = (rcc - identity).pow(2)

        if self.reduction == "sum":
            invariance = diff_square[diagonal_mask].sum()
            decorrelation = diff_square[off_diagonal_mask].sum()
        else:
            invariance = diff_square[diagonal_mask].mean()
            decorrelation = diff_square[off_diagonal_mask].mean()

        loss = invariance + self.kappa * decorrelation
        details = {
            "invariance": invariance.detach(),
            "decorrelation": decorrelation.detach(),
            "diag_mean": torch.diagonal(rcc).mean().detach(),
            "offdiag_abs_mean": rcc[off_diagonal_mask].abs().mean().detach(),
        }
        if return_details:
            return loss, details
        return loss
