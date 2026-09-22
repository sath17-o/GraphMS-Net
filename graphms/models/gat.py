"""Frozen Stage7 GAT geometry from the audited GraphMS lineage."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

CELL_SIZE = 4
GAT_HIDDEN = 128
GAT_HEADS = 4
GUIDE_DROPOUT = 0.30

def segment_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    _, heads = scores.shape
    idx = dst[:, None].expand(-1, heads)
    s = scores.float()
    mx = torch.full((num_nodes, heads), -torch.inf, device=s.device, dtype=s.dtype)
    mx.scatter_reduce_(0, idx, s, reduce="amax", include_self=True)
    ex = torch.exp(s - mx[dst])
    den = torch.zeros((num_nodes, heads), device=s.device, dtype=s.dtype)
    den.index_add_(0, dst, ex)
    return (ex / (den[dst] + 1e-8)).to(scores.dtype)

class SparseEdgeGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.dk = out_dim // heads
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dk))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dk))
        self.edge_bias = nn.Sequential(nn.Linear(edge_dim, heads), nn.Tanh())
        self.out = nn.Linear(out_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.edge_bias[0].weight)
        nn.init.zeros_(self.edge_bias[0].bias)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, edge_index, edge_attr):
        n = x.shape[0]
        src, dst = edge_index
        h = self.lin(x).view(n, self.heads, self.dk)
        e = (
            (h[src] * self.a_src).sum(-1)
            + (h[dst] * self.a_dst).sum(-1)
            + self.edge_bias(edge_attr)
        )
        a = segment_softmax(F.leaky_relu(e, 0.2), dst, n)
        a = F.dropout(a, p=self.dropout, training=self.training)
        msg = a[:, :, None] * h[src]
        out = msg.new_zeros((n, self.heads, self.dk))
        out.index_add_(0, dst, msg)
        return self.out(out.reshape(n, -1))

class TrueGAT(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int = 7):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, GAT_HIDDEN), nn.GELU(), nn.Dropout(GUIDE_DROPOUT)
        )
        self.g1 = SparseEdgeGATLayer(
            GAT_HIDDEN, GAT_HIDDEN, GAT_HEADS, edge_dim, GUIDE_DROPOUT
        )
        self.n1 = nn.LayerNorm(GAT_HIDDEN)
        self.g2 = SparseEdgeGATLayer(
            GAT_HIDDEN, GAT_HIDDEN, GAT_HEADS, edge_dim, GUIDE_DROPOUT
        )
        self.n2 = nn.LayerNorm(GAT_HIDDEN)
        self.pool = nn.Linear(GAT_HIDDEN, 1)
        self.vproj = nn.Sequential(
            nn.LayerNorm((CELL_SIZE ** 3) * 4),
            nn.Linear((CELL_SIZE ** 3) * 4, GAT_HIDDEN),
            nn.GELU(),
            nn.Dropout(GUIDE_DROPOUT),
        )
        fused_dim = GAT_HIDDEN * 5
        self.node_head = nn.Linear(fused_dim + CELL_SIZE ** 3, CELL_SIZE ** 3)
        nn.init.zeros_(self.node_head.weight)
        nn.init.zeros_(self.node_head.bias)
        with torch.no_grad():
            self.node_head.weight[:, fused_dim:] = torch.eye(CELL_SIZE ** 3)

    def forward(self, x, edge_index, edge_attr, voxel_logits, voxel_mri):
        h0 = self.proj(self.in_norm(x))
        h = self.n1(
            h0 + F.dropout(
                F.gelu(self.g1(h0, edge_index, edge_attr)),
                p=GUIDE_DROPOUT,
                training=self.training,
            )
        )
        h = self.n2(
            h + F.dropout(
                F.gelu(self.g2(h, edge_index, edge_attr)),
                p=GUIDE_DROPOUT,
                training=self.training,
            )
        )
        w = torch.softmax(self.pool(h).squeeze(1), 0)
        ca = (w[:, None] * h).sum(0)
        cm = h.mean(0)
        vc = torch.cat([voxel_logits, voxel_mri.reshape(voxel_mri.shape[0], -1)], 1)
        vctx = self.vproj(vc)
        fused = torch.cat(
            [h0, h, ca[None].expand_as(h), cm[None].expand_as(h), vctx], 1
        )
        final_logits = self.node_head(torch.cat([fused, voxel_logits], 1))
        return final_logits, h, vctx, torch.cat([ca, cm], 0)
