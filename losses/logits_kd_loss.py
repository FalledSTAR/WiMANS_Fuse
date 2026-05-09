import torch
import torch.nn.functional as F


def logits_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    confidence_threshold: float = 0.0,
) -> torch.Tensor:
    teacher_logits = teacher_logits.detach()
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if confidence_threshold > 0:
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        keep = teacher_probs.max(dim=-1).values >= float(confidence_threshold)
        if not bool(keep.any()):
            return student_logits.new_zeros(())
        student_logits = student_logits[keep]
        teacher_logits = teacher_logits[keep]

    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
