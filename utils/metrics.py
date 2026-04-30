import torch


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return (predictions == labels.long()).float().mean().item()
