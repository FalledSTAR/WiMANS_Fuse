import torch
from torch import nn
from torchvision.models.video import (
    MC3_18_Weights,
    MViT_V1_B_Weights,
    MViT_V2_S_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    S3D_Weights,
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
    mc3_18,
    mvit_v1_b,
    mvit_v2_s,
    r2plus1d_18,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)


VIDEO_BACKBONES = {
    "s3d": (s3d, S3D_Weights),
    "r3d_18": (r3d_18, R3D_18_Weights),
    "mc3_18": (mc3_18, MC3_18_Weights),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights),
    "mvit_v1_b": (mvit_v1_b, MViT_V1_B_Weights),
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights),
    "swin3d_t": (swin3d_t, Swin3D_T_Weights),
    "swin3d_s": (swin3d_s, Swin3D_S_Weights),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights),
}

VIDEO_BACKBONE_ALIASES = {
    "s3d": "s3d",
    "resnet": "r3d_18",
    "r3d-18": "r3d_18",
    "mc3-18": "mc3_18",
    "r2plus1d-18": "r2plus1d_18",
    "mvit-v1": "mvit_v1_b",
    "mvit-v2": "mvit_v2_s",
    "swin-t": "swin3d_t",
    "swin-s": "swin3d_s",
    "swin-b": "swin3d_b",
}


def normalize_video_backbone_name(backbone: str) -> str:
    name = str(backbone).strip()
    return VIDEO_BACKBONE_ALIASES.get(name, VIDEO_BACKBONE_ALIASES.get(name.lower(), name.lower()))


def resolve_video_weights(backbone: str, weights: str):
    backbone = normalize_video_backbone_name(backbone)
    if backbone not in VIDEO_BACKBONES:
        raise ValueError(f"Unsupported video backbone: {backbone}. Choices: {sorted(VIDEO_BACKBONES)}")
    weights_name = None if weights is None else str(weights).strip()
    weights_key = None if weights_name is None else weights_name.lower()
    if weights_key in (None, "none", "random"):
        return None

    _, weights_enum = VIDEO_BACKBONES[backbone]
    if weights_key in ("default", "kinetics400", "kinetics400_v1"):
        return weights_enum.DEFAULT
    if hasattr(weights_enum, weights_name):
        return getattr(weights_enum, weights_name)
    if hasattr(weights_enum, weights_name.upper()):
        return getattr(weights_enum, weights_name.upper())
    raise ValueError(f"Unsupported weights '{weights}' for video backbone '{backbone}'")


def build_video_transform(backbone: str, weights: str):
    backbone = normalize_video_backbone_name(backbone)
    resolved_weights = resolve_video_weights(backbone, weights)
    if resolved_weights is not None:
        return resolved_weights.transforms()
    return VIDEO_BACKBONES[backbone][1].DEFAULT.transforms()


def _replace_head(model: nn.Module, backbone: str, num_classes: int, dropout: float):
    if backbone == "s3d":
        in_features = model.classifier[1].in_channels
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv3d(in_features, num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )
        return model.classifier[1], model.classifier, in_features

    if backbone in {"r3d_18", "mc3_18", "r2plus1d_18"}:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model.fc, model.fc, in_features

    if backbone in {"mvit_v1_b", "mvit_v2_s"}:
        in_features = model.head[1].in_features
        model.head = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model.head[1], model.head, in_features

    if backbone in {"swin3d_t", "swin3d_s", "swin3d_b"}:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model.head, model.head, in_features

    raise ValueError(f"Unsupported video backbone: {backbone}")


class VideoTeacherClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "s3d",
        num_classes: int = 9,
        weights: str = "kinetics400",
        freeze_backbone: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        backbone = normalize_video_backbone_name(backbone)
        if backbone not in VIDEO_BACKBONES:
            raise ValueError(f"Unsupported video backbone: {backbone}. Choices: {sorted(VIDEO_BACKBONES)}")

        builder, _ = VIDEO_BACKBONES[backbone]
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.model = builder(weights=resolve_video_weights(backbone, weights))
        self.head_module, self.head_root, self.feature_dim = _replace_head(self.model, backbone, num_classes, dropout)

        if self.freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            for parameter in self.head_module.parameters():
                parameter.requires_grad = True
            self._set_backbone_eval()

    def _set_backbone_eval(self):
        for module in self.model.children():
            module.eval()
        self.head_root.train(self.training)

    def head_parameters(self):
        return self.head_root.parameters()

    def backbone_parameters(self):
        head_ids = {id(parameter) for parameter in self.head_root.parameters()}
        return [parameter for parameter in self.model.parameters() if id(parameter) not in head_ids]

    def forward(self, video: torch.Tensor, return_features: bool = False):
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [B,C,T,H,W], got {tuple(video.shape)}")

        captured = {}

        def capture_head_input(_module, inputs):
            feature = inputs[0]
            captured["feature"] = feature.flatten(1) if feature.ndim > 2 else feature

        handle = self.head_module.register_forward_pre_hook(capture_head_input) if return_features else None
        logits = self.model(video)
        if handle is not None:
            handle.remove()

        if return_features:
            return {"logits": logits, "feature": captured["feature"]}
        return logits

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self._set_backbone_eval()
        return self


S3DVideoTeacherClassifier = VideoTeacherClassifier
