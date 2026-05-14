import torch
from torch import nn
import torch.nn.functional as F


def bi_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    kl_pq = F.kl_div(
        F.log_softmax(p, dim=-1),
        F.softmax(q, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    kl_qp = F.kl_div(
        F.log_softmax(q, dim=-1),
        F.softmax(p, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def compute_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return torch.matmul(x, y.transpose(0, 1))


def compute_weighted_mse_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
):
    mse_per_sample = F.mse_loss(student_feat, teacher_feat, reduction="none").mean(dim=1)
    plain_mse = mse_per_sample.mean()

    student_teacher_similarity = compute_similarity(student_feat, teacher_feat)
    teacher_teacher_similarity = compute_similarity(teacher_feat, teacher_feat)

    sim_diff = torch.abs(teacher_teacher_similarity - student_teacher_similarity)
    diagonal_gap = torch.diagonal(sim_diff).sum() / sim_diff.sum().clamp_min(eps)

    relation_kl_per_sample = bi_kl_divergence(
        student_teacher_similarity / temperature,
        teacher_teacher_similarity / temperature,
    )
    weighted_mse = (relation_kl_per_sample * mse_per_sample).mean()

    return weighted_mse, relation_kl_per_sample.mean(), plain_mse, diagonal_gap


class CAFDLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.eps = eps

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, return_details: bool = False):
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                "CAFDLoss requires matching feature shapes, "
                f"got student={tuple(student_feat.shape)} teacher={tuple(teacher_feat.shape)}"
            )

        student = student_feat
        teacher = teacher_feat.detach()
        batch_size = int(student.shape[0])

        plain_mse_per_sample = F.mse_loss(student, teacher, reduction="none").mean(dim=-1)
        plain_mse = plain_mse_per_sample.mean()

        zero = student.new_zeros(())
        weighted_mse = plain_mse
        correlation = zero
        diagonal_gap = zero
        relation_kl = zero

        if batch_size > 1:
            weighted_mse, relation_kl, plain_mse, diagonal_gap = compute_weighted_mse_loss(
                student,
                teacher,
                temperature=self.temperature,
                eps=self.eps,
            )

        loss = weighted_mse + diagonal_gap
        details = {
            "weighted_mse": weighted_mse.detach(),
            "correlation": correlation.detach(),
            "diagonal_gap": diagonal_gap.detach(),
            "relation_kl": relation_kl.detach(),
            "plain_mse": plain_mse.detach(),
        }
        if return_details:
            return loss, details
        return loss
