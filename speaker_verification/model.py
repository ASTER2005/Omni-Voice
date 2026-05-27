"""
speaker_verification/model.py
────────────────────────────────────────────────────────────────
Two speaker embedding models:

Track A — ECAPA-TDNN via SpeechBrain (primary, pretrained)
Track B — ResNet-18 trained from scratch with ArcFace loss

Plus ArcFaceHead: additive angular margin loss head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ═══════════════════════════════════════════════════════════════
#  ArcFace Loss Head
# ═══════════════════════════════════════════════════════════════
class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin (ArcFace) softmax head.
    margin=0.5, scale=64 (standard settings).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalised [B, D]
            labels:     ground truth [B] (required during training)
        Returns:
            logits: [B, num_classes]
        """
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight, dim=1),
        )
        if labels is None:
            return cosine * self.scale   # inference mode

        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1) + 1e-7)
        phi  = cosine * self.cos_m - sine * self.sin_m
        phi  = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


# ═══════════════════════════════════════════════════════════════
#  Track A — ECAPA-TDNN (SpeechBrain wrapper)
# ═══════════════════════════════════════════════════════════════
class ECAPASpeakerModel(nn.Module):
    """
    Wraps SpeechBrain's pretrained ECAPA-TDNN for speaker embedding.
    Output: 192-dim L2-normalised speaker embedding.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier
        self._encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.embedding_dim = 192

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] float32 @ 16kHz
        Returns:
            embeddings: [B, 192] L2-normalised
        """
        embeddings = self._encoder.encode_batch(waveform)   # [B, 1, 192]
        embeddings = embeddings.squeeze(1)                   # [B, 192]
        return F.normalize(embeddings, dim=1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encode(waveform)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ═══════════════════════════════════════════════════════════════
#  Track B — ResNet-18 Speaker Encoder (custom, from scratch)
# ═══════════════════════════════════════════════════════════════
class ResNet18SpeakerEncoder(nn.Module):
    """
    Modified lightweight ResNet-18 for speaker embedding from 80-band Mel spectrograms.

    Input:  [B, 1, 80, T]
    Output: [B, embedding_dim] L2-normalised embedding
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Custom lightweight channels to massively reduce overfitting: 32, 64, 128, 128
        self.in_channels = 32

        self.conv1 = nn.Conv2d(
            1, 32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 Residual layers with lightweight width
        self.layer1 = self._make_layer(BasicBlock, 32, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, 2, stride=2)

        # Global stats pooling (mean + std concatenated)
        self.stats_pool = _StatsPool()

        # Regularized embedding projection with ReLU and Dropout(p=0.4)
        self.embedding = nn.Sequential(
            nn.Linear(128 * 2 * BasicBlock.expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 80, T]
        Returns:
            embeddings: [B, embedding_dim] L2-normalised
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feat = x.permute(0, 2, 3, 1)          # [B, H', T', 128]
        feat = feat.reshape(feat.size(0), -1, 128)  # [B, H'*T', 128]
        feat = self.stats_pool(feat)              # [B, 256]
        emb  = self.embedding(feat)              # [B, embedding_dim]
        return F.normalize(emb, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class _StatsPool(nn.Module):
    """Mean + standard deviation temporal pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std  = x.std(dim=1)
        return torch.cat([mean, std], dim=1)


# ═══════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════
def build_speaker_model(backend: str = "resnet18", **kwargs):
    """
    Factory function.
    backend: "ecapa" | "resnet18"
    """
    if backend == "ecapa":
        return ECAPASpeakerModel(**kwargs)
    elif backend == "resnet18":
        return ResNet18SpeakerEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ── Standalone sanity check ─────────────────────────────────────────────────
if __name__ == "__main__":
    # ResNet-18
    enc = ResNet18SpeakerEncoder(embedding_dim=512)
    dummy = torch.randn(4, 1, 80, 300)
    emb = enc(dummy)
    print(f"ResNet-18 embedding: {emb.shape}")
    print(f"ResNet-18 params:    {enc.count_parameters():,}")

    # ArcFace head
    head = ArcFaceHead(embedding_dim=512, num_classes=921)
    labels = torch.randint(0, 921, (4,))
    logits = head(emb, labels)
    print(f"ArcFace logits:      {logits.shape}")
