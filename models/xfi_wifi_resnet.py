import sys
import types
from pathlib import Path

import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=group)


def conv1x1(in_planes, out_planes, stride=1, group=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, group=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, group=group)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride, group=group)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, group=group)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, layers, inchannel=270, activity_num=55):
        super().__init__()
        self.inplanes = 128
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=7, stride=2, padding=3, bias=False, groups=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, group=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, group=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, group=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, group=1)
        self.conv4 = conv3x3(512, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, activity_num)

    def _make_layer(self, block, planes, blocks, stride=1, group=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, group, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group=group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def install_xfi_pickle_aliases():
    backbone_pkg = sys.modules.setdefault("backbone_models", types.ModuleType("backbone_models"))
    wifi_pkg = sys.modules.setdefault("backbone_models.WIFI", types.ModuleType("backbone_models.WIFI"))
    current_module = sys.modules[__name__]
    sys.modules["backbone_models.WIFI.ResNet"] = current_module
    setattr(backbone_pkg, "WIFI", wifi_pkg)
    setattr(wifi_pkg, "ResNet", current_module)


def load_xfi_wifi_resnet18(weight_path: str, map_location="cpu") -> nn.Module:
    path = Path(weight_path)
    if not path.exists():
        raise FileNotFoundError(f"X-Fi WiFi ResNet-18 weight not found: {path}")

    install_xfi_pickle_aliases()
    model = torch.load(str(path), map_location=map_location)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a pickled nn.Module, got {type(model)}")
    return model


class XFiWiFiStudent(nn.Module):
    def __init__(self, weight_path: str, num_classes: int = 9, freeze_backbone: bool = False):
        super().__init__()
        backbone = load_xfi_wifi_resnet18(weight_path, map_location="cpu")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = 512
        self.norm = nn.LayerNorm(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

    def extract_tokens(self, wifi: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(wifi)
        features = features.view(features.size(0), self.feature_dim, -1)
        return features.permute(0, 2, 1).contiguous()

    def forward(self, wifi: torch.Tensor, return_features: bool = False):
        tokens = self.extract_tokens(wifi)
        pooled = tokens.mean(dim=1)
        logits = self.classifier(self.norm(pooled))
        if return_features:
            return {"logits": logits, "feature": pooled, "tokens": tokens}
        return logits


class XFiWiFiOriginalFC(nn.Module):
    def __init__(self, weight_path: str, num_classes: int = 9, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = load_xfi_wifi_resnet18(weight_path, map_location="cpu")
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith("fc.")

    def extract_tokens(self, wifi: torch.Tensor) -> torch.Tensor:
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(wifi)))
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = x.view(x.size(0), self.feature_dim, -1)
        return x.permute(0, 2, 1).contiguous()

    def forward(self, wifi: torch.Tensor, return_features: bool = False):
        tokens = self.extract_tokens(wifi)
        pooled = self.backbone.avg_pool(tokens.permute(0, 2, 1).contiguous())
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.backbone.fc(pooled)
        if return_features:
            return {"logits": logits, "feature": pooled, "tokens": tokens}
        return logits
