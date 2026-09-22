"""Frozen Stage8/9 GraphMS v3.5.1 Hybrid geometry."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

GAT_HIDDEN = 128
FPN_CHANNELS = 32
GUIDE_DROPOUT = 0.30

class SE3D(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        h = max(8, c // 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, h, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(h, c, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.net(x)

class BottleneckSelfAttention(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
    def forward(self, x):
        b, c, d, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        q = self.ln(t)
        a, _ = self.attn(q, q, q, need_weights=False)
        return (t + a).transpose(1, 2).reshape(b, c, d, h, w)

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.GELU(),
            nn.Dropout3d(GUIDE_DROPOUT),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class SpatialHybrid(nn.Module):
    """Exact promoted SE + self-attention + multi-scale fusion geometry."""
    def __init__(self, stage_dims):
        super().__init__()
        self.stage_dims = list(map(int, stage_dims))
        C = FPN_CHANNELS
        self.cproj = nn.ModuleList([nn.Conv3d(c, C, 1) for c in self.stage_dims])
        self.gproj = nn.Conv3d(GAT_HIDDEN * 2, C, 1)
        self.glob = nn.Linear(GAT_HIDDEN * 2, C)
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(C * 3, C, 1),
                nn.GELU(),
                SE3D(C),
                ConvBlock(C, C),
            )
            for _ in range(4)
        ])
        self.self_attn = BottleneckSelfAttention(C, 4)
        self.lat3 = ConvBlock(C, C)
        self.lat2 = ConvBlock(C, C)
        self.lat1 = ConvBlock(C, C)
        self.up32 = nn.ConvTranspose3d(C, C // 2, 2, stride=2)
        self.ctx32 = nn.Conv3d(1, C // 2, 1)
        self.dec32 = ConvBlock(C, C // 2)
        self.up64 = nn.ConvTranspose3d(C // 2, C // 4, 2, stride=2)
        self.ctx64 = nn.Conv3d(1, C // 4, 1)
        self.dec64 = ConvBlock(C // 2, C // 4)
        self.out = nn.Conv3d(C // 4 + 1, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.out.weight[0, -1, 0, 0, 0] = 1.0

    def forward(self, cnn_scales, h_grid, v_grid, global_desc, context64):
        g = self.gproj(torch.cat([h_grid, v_grid], 1))
        gl = self.glob(global_desc).view(global_desc.shape[0], -1, 1, 1, 1)
        fs = []
        for i, (cg, pool) in enumerate(zip(cnn_scales, [1, 2, 4, 8])):
            c = self.cproj[i](cg)
            gg = g
            if pool > 1:
                c = F.avg_pool3d(c, pool, pool)
                gg = F.avg_pool3d(gg, pool, pool)
            ge = gl.expand(-1, -1, *c.shape[-3:])
            fs.append(self.fuse[i](torch.cat([c, gg, ge], 1)))
        p4 = self.self_attn(fs[3])
        p3 = self.lat3(fs[2] + F.interpolate(
            p4, size=fs[2].shape[-3:], mode="trilinear", align_corners=False
        ))
        p2 = self.lat2(fs[1] + F.interpolate(
            p3, size=fs[1].shape[-3:], mode="trilinear", align_corners=False
        ))
        p1 = self.lat1(fs[0] + F.interpolate(
            p2, size=fs[0].shape[-3:], mode="trilinear", align_corners=False
        ))
        y = self.up32(p1)
        y = self.dec32(torch.cat(
            [y, self.ctx32(F.avg_pool3d(context64, 2, 2))], 1
        ))
        y = self.up64(y)
        y = self.dec64(torch.cat([y, self.ctx64(context64)], 1))
        return self.out(torch.cat([y, context64[:, 0:1]], 1))
