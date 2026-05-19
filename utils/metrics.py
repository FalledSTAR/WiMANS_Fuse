import torch


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels.long()).float().mean().item()


def official_slot_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    """WiMANS activity accuracy: exact 9-way vector match per user slot."""
    labels_view = labels.reshape(labels.shape[0], 6, 9).float()
    predictions = (torch.sigmoid(logits).reshape(labels.shape[0], 6, 9) > float(threshold)).float()
    slot_correct = predictions.eq(labels_view).all(dim=-1)
    return slot_correct.float().mean().item()


def active_slot_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    labels_view = labels.reshape(labels.shape[0], 6, 9).float()
    predictions = (torch.sigmoid(logits).reshape(labels.shape[0], 6, 9) > float(threshold)).float()
    active = labels_view.sum(dim=-1) > 0
    if not bool(active.any()):
        return 0.0
    slot_correct = predictions.eq(labels_view).all(dim=-1)
    return slot_correct[active].float().mean().item()


def sample_exact_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    labels_flat = labels.reshape(labels.shape[0], -1).float()
    predictions = (torch.sigmoid(logits) > float(threshold)).float()
    return predictions.eq(labels_flat).all(dim=-1).float().mean().item()


def slot_argmax_prediction(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Decode each slot as at most one activity while keeping the 9-bit label space."""
    probs = torch.sigmoid(logits).reshape(logits.shape[0], 6, 9)
    max_probs, max_ids = probs.max(dim=-1)
    predictions = torch.zeros_like(probs)
    active = max_probs > float(threshold)
    predictions.scatter_(-1, max_ids.unsqueeze(-1), 1.0)
    predictions = predictions * active.unsqueeze(-1).float()
    return predictions


def slot_argmax_official_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    labels_view = labels.reshape(labels.shape[0], 6, 9).float()
    predictions = slot_argmax_prediction(logits, threshold=threshold)
    slot_correct = predictions.eq(labels_view).all(dim=-1)
    return slot_correct.float().mean().item()


def slot_argmax_active_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    labels_view = labels.reshape(labels.shape[0], 6, 9).float()
    predictions = slot_argmax_prediction(logits, threshold=threshold)
    active = labels_view.sum(dim=-1) > 0
    if not bool(active.any()):
        return 0.0
    slot_correct = predictions.eq(labels_view).all(dim=-1)
    return slot_correct[active].float().mean().item()


def slot_argmax_sample_exact_accuracy(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    labels_view = labels.reshape(labels.shape[0], 6, 9).float()
    predictions = slot_argmax_prediction(logits, threshold=threshold)
    return predictions.eq(labels_view).all(dim=-1).all(dim=-1).float().mean().item()


def multi_slot_ce_prediction(logits: torch.Tensor) -> torch.Tensor:
    return logits.reshape(logits.shape[0], 6, 10).argmax(dim=-1)


def multi_slot_ce_official_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = multi_slot_ce_prediction(logits)
    labels_view = labels.reshape(labels.shape[0], 6).long()
    return predictions.eq(labels_view).float().mean().item()


def multi_slot_ce_active_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = multi_slot_ce_prediction(logits)
    labels_view = labels.reshape(labels.shape[0], 6).long()
    active = labels_view > 0
    if not bool(active.any()):
        return 0.0
    return predictions.eq(labels_view)[active].float().mean().item()


def multi_slot_ce_sample_exact_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = multi_slot_ce_prediction(logits)
    labels_view = labels.reshape(labels.shape[0], 6).long()
    return predictions.eq(labels_view).all(dim=-1).float().mean().item()


def accuracy_for_mode(logits: torch.Tensor, labels: torch.Tensor, label_mode: str, threshold: float = 0.5) -> float:
    if label_mode == "single_ce":
        return accuracy_top1(logits, labels)
    if label_mode == "multi_bce":
        return official_slot_accuracy(logits, labels, threshold=threshold)
    if label_mode == "multi_slot_ce":
        return multi_slot_ce_official_accuracy(logits, labels)
    raise ValueError(f"Unsupported label_mode: {label_mode}")
