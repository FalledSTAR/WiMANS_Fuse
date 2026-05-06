from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def _profile_ptflops(module: torch.nn.Module, input_res: Tuple[int, ...]):
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return {"available": False, "error": "ptflops is not installed"}

    module_cpu = copy.deepcopy(module).cpu().eval()
    profiled_params = count_parameters(module_cpu)["total"]
    for child in module_cpu.modules():
        for attr in ("__flops__", "__params__", "__batch_counter__"):
            if hasattr(child, attr):
                delattr(child, attr)
    try:
        macs, params = get_model_complexity_info(
            module_cpu,
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return {
            "available": True,
            "input_res": list(input_res),
            "macs": int(macs),
            "flops_approx": int(macs * 2),
            "params_profiled": int(profiled_params),
            "ptflops_params_raw": int(params),
            "note": "ptflops reports MACs; flops_approx is MACs * 2.",
        }
    except Exception as exc:  # pragma: no cover - profiler support differs by layer/version
        return {"available": False, "input_res": list(input_res), "error": repr(exc)}
    finally:
        del module_cpu


def _projector_macs(num_tokens: int, in_dim: int, hidden_dim: int, out_dim: int) -> int:
    # Rough single-sample MAC estimate for HybridProjector.
    fc_in = num_tokens * in_dim * hidden_dim
    qkv_and_out = 4 * num_tokens * hidden_dim * hidden_dim
    attention_scores = 2 * num_tokens * num_tokens * hidden_dim
    fc_out = hidden_dim * out_dim
    return int(fc_in + qkv_and_out + attention_scores + fc_out)


def build_model_summary(model: torch.nn.Module, stage: str, cfg: dict) -> dict:
    summary = {
        "stage": stage,
        "parameters": count_parameters(model),
        "flops": {},
    }

    target_len = int(cfg["data"]["target_len"])
    if stage == "v0":
        summary["flops"]["model"] = _profile_ptflops(model, (int(cfg["model"]["wifi_input_channels"]), target_len))
        return summary

    wifi_tokens = max((target_len + 255) // 256, 1)
    video_profile_frames = int(cfg.get("logging", {}).get("flops_video_frames", min(int(cfg["video"]["num_frames"]), 16)))
    summary["flops"]["wifi_student"] = _profile_ptflops(
        model.wifi_student,
        (int(cfg["model"]["wifi_input_channels"]), target_len),
    )
    summary["flops"]["video_teacher"] = _profile_ptflops(
        model.video_teacher,
        (3, video_profile_frames, 224, 224),
    )
    summary["flops"]["projectors_rough"] = {
        "available": True,
        "note": "Rough MAC estimate for HybridProjector modules only.",
        "wifi_tokens_assumed": wifi_tokens,
        "video_tokens_assumed": 1,
        "macs": _projector_macs(
            wifi_tokens,
            512,
            int(cfg["projector"]["hidden_dim"]),
            int(cfg["projector"]["out_dim"]),
        )
        + _projector_macs(
            1,
            1024,
            int(cfg["projector"]["hidden_dim"]),
            int(cfg["projector"]["out_dim"]),
        ),
    }
    summary["flops"]["projectors_rough"]["flops_approx"] = int(summary["flops"]["projectors_rough"]["macs"] * 2)
    return summary
