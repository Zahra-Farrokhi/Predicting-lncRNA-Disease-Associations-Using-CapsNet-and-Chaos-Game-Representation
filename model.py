# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import squash

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key   = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        return torch.matmul(attn, V)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_ch, out_caps, cap_dim, k, s):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_caps * cap_dim, k, s)
        self.out_caps, self.cap_dim = out_caps, cap_dim

    def forward(self, x):
        u = self.conv(x)
        B, C, H, W = u.size()
        u = u.view(B, self.out_caps, self.cap_dim, H * W).permute(0, 1, 3, 2)
        u = u.contiguous().view(B, -1, self.cap_dim)
        return squash(u, dim=-1)

class DigitCapsules(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, iters=5):
        super().__init__()
        self.out_caps, self.out_dim, self.iters = out_caps, out_dim, iters
        self.W = None

    def forward(self, x):
        B, in_caps, in_dim = x.size()
        if self.W is None or self.W.size(1) != in_caps:
            self.W = nn.Parameter(torch.randn(
                1, in_caps, self.out_caps, self.out_dim, in_dim,
                device=x.device
            ))
        x = x.view(B, in_caps, in_dim, 1).unsqueeze(2)
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x)
        b = torch.zeros(B, in_caps, self.out_caps, 1, 1, device=x.device)
        for r in range(self.iters):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = squash(s, dim=-2)
            if r < self.iters - 1:
                b = b + (u_hat * v).sum(dim=-1, keepdim=True)
        v = v.squeeze(1).squeeze(-1)
        return v

class ImageCapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.primary = PrimaryCapsules(256, 8, 32, 9, 2)
        self.digit   = DigitCapsules(1152, 32, 16, 16)

    def forward(self, x):
        x = self.features(x)
        u = self.primary(x)
        v = self.digit(u)
        return v.norm(dim=-1)

class DiseaseNet(nn.Module):
    def __init__(self, ng):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ng, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU()
        )
    def forward(self, x):
        return self.mlp(x)

class JointModel(nn.Module):
    def __init__(self, ng):
        super().__init__()
        self.img_net = ImageCapsNet()
        self.dis_net = DiseaseNet(ng)
        self.attn    = Attention(80)
        self.fusion  = nn.Sequential(
            nn.Linear(80, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    def forward(self, i, df):
        img_feat = self.img_net(i)
        dis_feat = self.dis_net(df)
        combined = torch.cat([img_feat, dis_feat], 1).unsqueeze(1)
        attended = self.attn(combined).squeeze(1)
        return self.fusion(attended).squeeze(1)
