"""Wireframe flow AE: wireframe encoder + conditional flow-matching denoiser.

Latent (z) and joint tokens are dropped per-sample during training so one
model learns both z-on reconstruction (diagnostic) and constraint-only
generation (the actual target), and classifier-free guidance comes for free.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import JOINT_FEAT_DIM, TYPE_NAMES

N_TYPES = len(TYPE_NAMES)
POINT_FEAT_DIM = 3 + N_TYPES + 3  # xyz + type onehot + tangent (encoder input)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t[:, None] * freqs[None, :] * 1000.0
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Block(nn.Module):
    """Pre-LN: self-attention over points, cross-attention to condition, MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ca = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, cond, cond_pad_mask):
        h = self.n1(x)
        x = x + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.ca(h, cond, cond, key_padding_mask=cond_pad_mask, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class RelationBlock(nn.Module):
    """Self-attention among constraint tokens with an additive logits bias
    (the constraint relation graph A_ij, e.g. -geodesic/tau)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.sa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, bias, key_pad_mask):
        h = self.n1(x)
        attn_bias = bias.repeat_interleave(self.heads, dim=0)  # (B*heads, K, K)
        x = x + self.sa(h, h, h, attn_mask=attn_bias,
                        key_padding_mask=key_pad_mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class WireframeEncoder(nn.Module):
    """Wireframe points -> fixed set of latent tokens via learned queries."""

    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 3, n_latent: int = 32):
        super().__init__()
        self.embed = nn.Linear(POINT_FEAT_DIM, dim)
        enc_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.queries = nn.Parameter(torch.randn(n_latent, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, points, type_ids, tangents):
        feat = torch.cat([points, F.one_hot(type_ids, N_TYPES).float(), tangents], dim=-1)
        h = self.encoder(self.embed(feat))
        q = self.queries[None].expand(h.shape[0], -1, -1)
        z, _ = self.pool(q, h, h, need_weights=False)
        return self.norm(z)


class FlowDenoiser(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 6):
        super().__init__()
        self.point_in = nn.Linear(3, dim)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.joint_in = nn.Linear(JOINT_FEAT_DIM, dim)
        self.relation = nn.ModuleList(RelationBlock(dim, heads) for _ in range(2))
        self.bbox_in = nn.Linear(4, dim)
        self.z_proj = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(dim)
        self.v_head = nn.Linear(dim, 3)
        self.type_head = nn.Linear(dim, N_TYPES)
        self.dim = dim

    def forward(self, x_t, t, z, joints, joints_mask, bbox, relation_bias=None):
        """z: (B,Lz,dim) or None. joints_mask: (B,K) True=valid.
        relation_bias: (B,K,K) additive attention logits among joints, or None."""
        B = x_t.shape[0]
        tok = self.point_in(x_t) + self.t_mlp(timestep_embedding(t, self.dim))[:, None, :]
        cond = [self.bbox_in(bbox)[:, None, :]]
        pad = [torch.zeros(B, 1, dtype=torch.bool, device=x_t.device)]  # bbox always valid
        jt = self.joint_in(joints)
        if relation_bias is not None:
            kpm = ~joints_mask
            all_masked = kpm.all(dim=1)
            if all_masked.any():  # avoid NaN when a sample has zero valid joints
                kpm = kpm.clone()
                kpm[all_masked, 0] = False
            for blk in self.relation:
                jt = blk(jt, relation_bias, kpm)
        cond.append(jt)
        pad.append(~joints_mask)
        if z is not None:
            cond.append(self.z_proj(z))
            pad.append(torch.zeros(B, z.shape[1], dtype=torch.bool, device=x_t.device))
        cond = torch.cat(cond, dim=1)
        pad = torch.cat(pad, dim=1)
        for blk in self.blocks:
            tok = blk(tok, cond, pad)
        h = self.out_norm(tok)
        return self.v_head(h), self.type_head(h)


class WireFlowModel(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, enc_layers: int = 3,
                 dec_layers: int = 6, n_latent: int = 32, type_loss_weight: float = 0.1):
        super().__init__()
        self.encoder = WireframeEncoder(dim, heads, enc_layers, n_latent)
        self.denoiser = FlowDenoiser(dim, heads, dec_layers)
        self.type_loss_weight = type_loss_weight

    @staticmethod
    @torch.no_grad()
    def _ot_match(x0: torch.Tensor, x1: torch.Tensor, eps: float = 0.01,
                  iters: int = 50) -> torch.Tensor:
        """Sinkhorn coupling: for each source point, the best-matching target
        index. Independent coupling makes crossing flows whose average is a
        blurry field; OT pairing straightens the paths."""
        logk = -torch.cdist(x0, x1) ** 2 / eps  # (B,M,M)
        u = torch.zeros(logk.shape[:2], device=logk.device)
        v = torch.zeros_like(u)
        for _ in range(iters):
            u = -torch.logsumexp(logk + v[:, None, :], dim=2)
            v = -torch.logsumexp(logk + u[:, :, None], dim=1)
        return (logk + u[:, :, None] + v[:, None, :]).argmax(dim=2)

    @staticmethod
    def relation_bias_from_batch(batch, mode: str) -> torch.Tensor:
        """'none' arm still runs the relation encoder (zero bias) so the three
        arms differ only in the bias content, not in architecture."""
        if mode == "oracle":
            return batch["relation_oracle"]
        if mode == "heuristic":
            return batch["relation_heuristic"]
        return torch.zeros_like(batch["relation_oracle"])

    def training_losses(self, batch, z_dropout: float, joint_dropout: float = 0.1,
                        ot_coupling: bool = False, relation_mode: str = "none"):
        pts, tids, tans = batch["points"], batch["type_ids"], batch["tangents"]
        B = pts.shape[0]
        device = pts.device
        z = self.encoder(pts, tids, tans)
        keep_z = (torch.rand(B, device=device) >= z_dropout).float()[:, None, None]
        z = z * keep_z  # dropped samples see zero latent tokens
        jmask = batch["joints_mask"]
        keep_j = torch.rand(B, device=device) >= joint_dropout
        jmask = jmask & keep_j[:, None]

        # linear flow-matching path from bbox-uniform source to the wireframe
        he = batch["bbox"][:, None, :3]
        x0 = (torch.rand_like(pts) * 2.0 - 1.0) * he
        if ot_coupling:
            idx = self._ot_match(x0, pts)
            pts = torch.gather(pts, 1, idx[:, :, None].expand(-1, -1, 3))
            tids = torch.gather(tids, 1, idx)
        t = torch.rand(B, device=device)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * pts
        v_target = pts - x0
        v_pred, type_logits = self.denoiser(
            x_t, t, z, batch["joints"], jmask, batch["bbox"],
            relation_bias=self.relation_bias_from_batch(batch, relation_mode))
        loss_v = F.mse_loss(v_pred, v_target)
        loss_type = F.cross_entropy(type_logits.reshape(-1, N_TYPES), tids.reshape(-1))
        return {"loss": loss_v + self.type_loss_weight * loss_type,
                "loss_v": loss_v.detach(), "loss_type": loss_type.detach()}

    @torch.no_grad()
    def sample(self, joints, joints_mask, bbox, n_points: int, steps: int = 30,
               z: torch.Tensor | None = None, guidance: float = 1.0,
               generator: torch.Generator | None = None,
               relation_bias: torch.Tensor | None = None):
        """Euler integration of the learned flow from bbox-uniform noise."""
        B = bbox.shape[0]
        device = bbox.device
        he = bbox[:, None, :3]
        x = (torch.rand(B, n_points, 3, device=device, generator=generator) * 2 - 1) * he
        dt = 1.0 / steps
        type_logits = None
        no_joints = torch.zeros_like(joints_mask)
        for i in range(steps):
            t = torch.full((B,), i * dt, device=device)
            v, type_logits = self.denoiser(x, t, z, joints, joints_mask, bbox,
                                           relation_bias=relation_bias)
            if guidance != 1.0:
                v_u, _ = self.denoiser(x, t, None, joints, no_joints, bbox)
                v = v_u + guidance * (v - v_u)
            x = x + v * dt
        return x, type_logits.argmax(dim=-1)
