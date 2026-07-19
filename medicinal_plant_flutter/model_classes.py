"""
VECTVMixer Model Classes - Extracted from training notebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        tv_loss = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() + \
                  torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return x, tv_loss


class ColorBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 11, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(11)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        tv_loss = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() + \
                  torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return x, tv_loss


class VeinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        tv_loss = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() + \
                  torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
        return x, tv_loss


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(num_tokens, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = x + self.mlp1(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp2(self.norm2(x))
        return x


class VECTVMixer(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        self.edge = EdgeBranch()
        self.color = ColorBranch()
        self.vein = VeinBranch()

        self.fusion = nn.Sequential(
            nn.Conv2d(35, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2)
        )

        self.constituent = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )

        self.num_tokens = 14 * 14

        self.mixer = nn.Sequential(
            MixerBlock(self.num_tokens, 128),
            MixerBlock(self.num_tokens, 128),
            MixerBlock(self.num_tokens, 128),
            MixerBlock(self.num_tokens, 128)
        )

        self.norm = nn.LayerNorm(128)
        self.fc1 = nn.Linear(128, 28)
        self.bn = nn.BatchNorm1d(28)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(28, num_classes)

    def forward(self, x):
        e, tv_e = self.edge(x)
        c, tv_c = self.color(x)
        v, tv_v = self.vein(x)

        x = torch.cat([e, c, v], dim=1)
        x = self.fusion(x)
        x = self.constituent(x)
        x = self.down1(x)
        x = self.down2(x)

        B, C, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.mixer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits
