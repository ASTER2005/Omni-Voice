"""
wake_word/model.py
────────────────────────────────────────────────────────────────
Depthwise Separable CNN for wake word detection.

Input:  [B, 1, 64, 101]  — 1-second log-Mel spectrogram
Output: [B, num_classes]  — logits (background + wake words)

Architecture: DS-CNN (< 500K params, < 10ms inference on CPU)
  3 × DepthwiseSeparableConv blocks
  Global Average Pooling
  Fully-connected classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block: DW-Conv + PW-Conv + BN + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        return self.relu(x)


class WakeWordCNN(nn.Module):
    """
    Lightweight Depthwise-Separable CNN for keyword spotting.

    Param count: ~180K (well under 500K target).
    """

    def __init__(self, num_classes: int = 2, n_mels: int = 64):
        super().__init__()

        # Initial standard conv (small, to preserve resolution)
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 3 × DS-Conv blocks with progressive pooling
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2),   # [B, 64, 32, 50]
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2, 2),   # [B, 128, 16, 25]
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.MaxPool2d(2, 2),   # [B, 128, 8, 12]
        )

        # Global Average Pooling → [B, 128]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, n_mels, time_frames]
        Returns:
            logits: [B, num_classes]
        """
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)               # [B, 128, 1, 1]
        x = x.flatten(1)              # [B, 128]
        x = self.dropout(x)
        return self.classifier(x)     # [B, num_classes]

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Standalone sanity check ─────────────────────────────────────────────────
if __name__ == "__main__":
    model = WakeWordCNN(num_classes=2, n_mels=64)
    dummy = torch.randn(4, 1, 64, 101)
    out = model(dummy)
    print(f"Output shape:    {out.shape}")
    print(f"Parameters:      {model.count_parameters():,}")
    proba = model.predict_proba(dummy)
    print(f"Probabilities:   {proba[0].detach().numpy()}")
