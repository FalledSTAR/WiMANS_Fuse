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


class CAFDLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        use_weighted_mse: bool = True,
        use_correlation: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_weighted_mse = use_weighted_mse
        self.use_correlation = use_correlation
        self.eps = eps

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, return_details: bool = False):
        if student_feat.shape != teacher_feat.shape:
            raise ValueError(
                "CAFDLoss requires matching feature shapes, "
                f"got student={tuple(student_feat.shape)} teacher={tuple(teacher_feat.shape)}"
            )

        student = F.normalize(student_feat, dim=-1)
        teacher = F.normalize(teacher_feat.detach(), dim=-1)
        batch_size = int(student.shape[0])

        plain_mse_per_sample = F.mse_loss(student, teacher, reduction="none").mean(dim=-1)
        plain_mse = plain_mse_per_sample.mean()

        zero = student.new_zeros(())
        weighted_mse = plain_mse if self.use_weighted_mse else zero
        correlation = zero
        diagonal_gap = zero
        relation_kl = zero

        if batch_size > 1:
            student_teacher_similarity = compute_similarity(student, teacher)
            teacher_teacher_similarity = compute_similarity(teacher, teacher)

            relation_kl_per_sample = bi_kl_divergence(
                student_teacher_similarity / self.temperature,
                teacher_teacher_similarity / self.temperature,
            )
            relation_kl = relation_kl_per_sample.mean()

            sim_diff = torch.abs(teacher_teacher_similarity - student_teacher_similarity)
            diagonal_gap = torch.diagonal(sim_diff).sum() / sim_diff.sum().clamp_min(self.eps)

            if self.use_weighted_mse:
                weighted_mse = (relation_kl_per_sample * plain_mse_per_sample).mean()

            if self.use_correlation:
                student_student_similarity = compute_similarity(student, student)
                correlation = bi_kl_divergence(
                    student_student_similarity / self.temperature,
                    teacher_teacher_similarity / self.temperature,
                ).mean()

        loss = self.alpha * weighted_mse + self.beta * correlation + self.gamma * diagonal_gap
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
