import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, label_mode: str) -> torch.Tensor:
    if label_mode == "single_ce":
        return F.cross_entropy(logits, labels.long())

    if label_mode == "multi_bce":
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    raise ValueError(f"Unsupported label_mode: {label_mode}")
