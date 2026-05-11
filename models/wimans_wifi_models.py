import torch
from torch import nn


class WiMANSWiFiCNN2D(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.feature_dim = 128
        self.layer_norm_0 = nn.BatchNorm2d(1)
        self.layer_norm_1 = nn.BatchNorm2d(32)
        self.layer_norm_2 = nn.BatchNorm2d(64)
        self.layer_norm_3 = nn.BatchNorm2d(128)
        self.layer_cnn_2d_0 = nn.Conv2d(1, 32, kernel_size=(27, 27), stride=(7, 7))
        self.layer_cnn_2d_1 = nn.Conv2d(32, 64, kernel_size=(15, 15), stride=(3, 3))
        self.layer_cnn_2d_2 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(1, 1))
        self.layer_linear = nn.Linear(self.feature_dim, num_classes)
        self.layer_leakyrelu = nn.LeakyReLU()
        self.layer_dropout = nn.Dropout(0.2)

        nn.init.xavier_uniform_(self.layer_cnn_2d_0.weight)
        nn.init.xavier_uniform_(self.layer_cnn_2d_1.weight)
        nn.init.xavier_uniform_(self.layer_cnn_2d_2.weight)
        nn.init.xavier_uniform_(self.layer_linear.weight)

    def extract_feature(self, wifi: torch.Tensor) -> torch.Tensor:
        if wifi.ndim != 3:
            raise ValueError(f"Expected WiFi shape [B,270,T], got {tuple(wifi.shape)}")
        x = wifi.transpose(1, 2).contiguous()
        x = x.unsqueeze(1)
        x = self.layer_norm_0(x)
        x = self.layer_dropout(self.layer_leakyrelu(self.layer_cnn_2d_0(x)))
        x = self.layer_norm_1(x)
        x = self.layer_dropout(self.layer_leakyrelu(self.layer_cnn_2d_1(x)))
        x = self.layer_norm_2(x)
        x = self.layer_dropout(self.layer_leakyrelu(self.layer_cnn_2d_2(x)))
        x = self.layer_norm_3(x)
        return torch.mean(x, dim=(-2, -1))

    def forward(self, wifi: torch.Tensor, return_features: bool = False):
        feature = self.extract_feature(wifi)
        logits = self.layer_linear(feature)
        if return_features:
            return {"logits": logits, "feature": feature, "tokens": feature.unsqueeze(1)}
        return logits


class GaussianPosition(nn.Module):
    def __init__(self, feature_dim: int, time_dim: int, num_gaussian: int = 10):
        super().__init__()
        embedding = torch.zeros([num_gaussian, feature_dim], dtype=torch.float)
        self.embedding = nn.Parameter(embedding, requires_grad=True)
        nn.init.xavier_uniform_(self.embedding)
        position = torch.arange(0.0, time_dim).unsqueeze(1).repeat(1, num_gaussian)
        self.position = nn.Parameter(position, requires_grad=False)
        mu = torch.arange(0.0, time_dim, time_dim / num_gaussian).unsqueeze(0)
        sigma = torch.tensor([50.0] * num_gaussian).unsqueeze(0)
        self.mu = nn.Parameter(mu, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pdf = self.position - self.mu
        pdf = -pdf * pdf
        pdf = pdf / self.sigma / self.sigma / 2
        pdf = pdf - torch.log(self.sigma)
        pdf = torch.softmax(pdf, dim=-1)
        position_encoding = torch.matmul(pdf, self.embedding)
        return x + position_encoding.unsqueeze(0)


class THATEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_head: int = 10, cnn_sizes=None):
        super().__init__()
        if cnn_sizes is None:
            cnn_sizes = [1, 3, 5]
        self.layer_norm_0 = nn.LayerNorm(feature_dim, eps=1e-6)
        self.layer_attention = nn.MultiheadAttention(feature_dim, num_head, batch_first=True)
        self.layer_dropout_0 = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(feature_dim, eps=1e-6)
        self.layer_cnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(feature_dim, feature_dim, kernel_size=size, padding="same"),
                    nn.BatchNorm1d(feature_dim),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(),
                )
                for size in cnn_sizes
            ]
        )
        self.layer_dropout_1 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.layer_norm_0(x)
        attn_out, _ = self.layer_attention(attn_in, attn_in, attn_in)
        x = self.layer_dropout_0(attn_out) + x
        conv_in = self.layer_norm_1(x).permute(0, 2, 1)
        conv_out = torch.stack([layer(conv_in) for layer in self.layer_cnn], dim=0).mean(dim=0)
        conv_out = self.layer_dropout_1(conv_out).permute(0, 2, 1)
        return conv_out + x


class WiMANSWiFiTHAT(nn.Module):
    def __init__(self, num_classes: int = 9, time_dim: int = 3000, feature_dim: int = 270):
        super().__init__()
        self.feature_dim = 288
        pooled_time_dim = time_dim // 20

        self.layer_left_pooling = nn.AvgPool1d(kernel_size=20, stride=20)
        self.layer_left_gaussian = GaussianPosition(feature_dim, pooled_time_dim)
        self.layer_left_encoder = nn.ModuleList(
            [THATEncoder(feature_dim=feature_dim, num_head=10, cnn_sizes=[1, 3, 5]) for _ in range(4)]
        )
        self.layer_left_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.layer_left_cnn_0 = nn.Conv1d(feature_dim, 128, kernel_size=8)
        self.layer_left_cnn_1 = nn.Conv1d(feature_dim, 128, kernel_size=16)
        self.layer_left_dropout = nn.Dropout(0.5)

        self.layer_right_pooling = nn.AvgPool1d(kernel_size=20, stride=20)
        self.layer_right_encoder = nn.ModuleList(
            [THATEncoder(feature_dim=pooled_time_dim, num_head=10, cnn_sizes=[1, 2, 3])]
        )
        self.layer_right_norm = nn.LayerNorm(pooled_time_dim, eps=1e-6)
        self.layer_right_cnn_0 = nn.Conv1d(pooled_time_dim, 16, kernel_size=2)
        self.layer_right_cnn_1 = nn.Conv1d(pooled_time_dim, 16, kernel_size=4)
        self.layer_right_dropout = nn.Dropout(0.5)

        self.layer_leakyrelu = nn.LeakyReLU()
        self.layer_output = nn.Linear(self.feature_dim, num_classes)

    def extract_feature(self, wifi: torch.Tensor) -> torch.Tensor:
        if wifi.ndim != 3:
            raise ValueError(f"Expected WiFi shape [B,270,T], got {tuple(wifi.shape)}")
        x = wifi.transpose(1, 2).contiguous()

        left = x.permute(0, 2, 1)
        left = self.layer_left_pooling(left).permute(0, 2, 1)
        left = self.layer_left_gaussian(left)
        for layer in self.layer_left_encoder:
            left = layer(left)
        left = self.layer_left_norm(left).permute(0, 2, 1)
        left_0 = self.layer_leakyrelu(self.layer_left_cnn_0(left)).sum(dim=-1)
        left_1 = self.layer_leakyrelu(self.layer_left_cnn_1(left)).sum(dim=-1)
        left = self.layer_left_dropout(torch.cat([left_0, left_1], dim=-1))

        right = x.permute(0, 2, 1)
        right = self.layer_right_pooling(right)
        for layer in self.layer_right_encoder:
            right = layer(right)
        right = self.layer_right_norm(right).permute(0, 2, 1)
        right_0 = self.layer_leakyrelu(self.layer_right_cnn_0(right)).sum(dim=-1)
        right_1 = self.layer_leakyrelu(self.layer_right_cnn_1(right)).sum(dim=-1)
        right = self.layer_right_dropout(torch.cat([right_0, right_1], dim=-1))

        return torch.cat([left, right], dim=-1)

    def forward(self, wifi: torch.Tensor, return_features: bool = False):
        feature = self.extract_feature(wifi)
        logits = self.layer_output(feature)
        if return_features:
            return {"logits": logits, "feature": feature, "tokens": feature.unsqueeze(1)}
        return logits
