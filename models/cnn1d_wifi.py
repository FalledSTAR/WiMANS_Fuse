import torch
from torch import nn


class CNN1DWiFi(nn.Module):
    """
    WiMANS-style CNN-1D WiFi CSI baseline.

    The official WiMANS implementation reshapes CSI to [B, T, 270] and then
    permutes to [B, 270, T]. This project loader already returns [B, 270, T],
    so the same Conv1d stack is applied directly.
    """

    def __init__(self, input_channels: int = 270, num_classes: int = 54, dropout: float = 0.2):
        super().__init__()
        self.feature_dim = 512
        self.weight_path = None
        self.loaded_backbone_type = "CNN1DWiFi"
        self.input_norm = nn.BatchNorm1d(input_channels)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=29, stride=13),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(128, 256, kernel_size=15, stride=7),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extract_tokens(self, wifi: torch.Tensor) -> torch.Tensor:
        if wifi.ndim != 3:
            raise ValueError(f"CNN1DWiFi expects [B,C,T] input, got {tuple(wifi.shape)}")
        features = self.feature_extractor(self.input_norm(wifi))
        return features.permute(0, 2, 1).contiguous()

    def forward(self, wifi: torch.Tensor, return_features: bool = False):
        tokens = self.extract_tokens(wifi)
        pooled = tokens.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        if return_features:
            return {"logits": logits, "feature": pooled, "tokens": tokens}
        return logits
