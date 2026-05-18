import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, label_mode: str, pos_weight=None) -> torch.Tensor:
    if label_mode == "single_ce":
        return F.cross_entropy(logits, labels.long())

    if label_mode == "multi_bce":
        target = labels.reshape(labels.shape[0], -1).float()
        weight = None
        if pos_weight is not None:
            if torch.is_tensor(pos_weight):
                weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
            else:
                weight = torch.full((target.shape[-1],), float(pos_weight), device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)

    raise ValueError(f"Unsupported label_mode: {label_mode}")
