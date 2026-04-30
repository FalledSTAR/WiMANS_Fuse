import torch
from torch import nn
import torch.nn.functional as F


class CAFDLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        student = F.normalize(student_feat, dim=-1)
        teacher = F.normalize(teacher_feat.detach(), dim=-1)

        feature_loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1).mean()

        teacher_logits = torch.matmul(teacher, teacher.t()) / self.temperature
        student_teacher_logits = torch.matmul(student, teacher.t()) / self.temperature

        teacher_relation = F.softmax(teacher_logits, dim=-1)
        student_teacher_log_relation = F.log_softmax(student_teacher_logits, dim=-1)

        relation_loss = F.kl_div(student_teacher_log_relation, teacher_relation, reduction="batchmean")
        return self.alpha * feature_loss + self.beta * relation_loss
