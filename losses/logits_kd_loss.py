import torch
import torch.nn.functional as F
from torch import nn


class LogitsKDLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, multi_user_slots: int = 6, slot_classes: int = 10):
        super().__init__()
        self.temperature = float(temperature)
        self.multi_user_slots = int(multi_user_slots)
        self.slot_classes = int(slot_classes)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, label_mode: str) -> torch.Tensor:
        if teacher_logits is None:
            raise ValueError("LogitsKDLoss requires teacher_logits, got None")
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "LogitsKDLoss expects matching logits shapes, "
                f"got student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
            )

        label_mode = str(label_mode)
        if label_mode == "multi_slot_ce":
            student_logits = student_logits.reshape(-1, self.slot_classes)
            teacher_logits = teacher_logits.reshape(-1, self.slot_classes)
            return self._softmax_kl(student_logits, teacher_logits)

        if label_mode == "single_ce":
            return self._softmax_kl(student_logits, teacher_logits)

        if label_mode == "multi_bce":
            return self._sigmoid_soft_bce(student_logits, teacher_logits)

        raise ValueError(f"Unsupported label_mode for logits KD: {label_mode}")

    def _softmax_kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)

    def _sigmoid_soft_bce(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        soft_targets = torch.sigmoid(teacher_logits.detach() / temperature)
        return F.binary_cross_entropy_with_logits(student_logits / temperature, soft_targets) * (temperature ** 2)
