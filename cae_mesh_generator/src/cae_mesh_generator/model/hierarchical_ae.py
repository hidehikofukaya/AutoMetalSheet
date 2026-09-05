"""Hierarchical midsurface point-cloud autoencoder prototype."""

from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
import torch.nn.functional as F


def farthest_point_sample(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Deterministic batched FPS on normalized points.

    points: [B, N, 3]
    returns indices [B, n_samples]
    """

    bsz, n_points, _ = points.shape
    n_samples = min(n_samples, n_points)
    device = points.device
    centroids = torch.zeros(bsz, n_samples, dtype=torch.long, device=device)
    distance = torch.full((bsz, n_points), 1.0e10, device=device)
    farthest = torch.zeros(bsz, dtype=torch.long, device=device)
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest].view(bsz, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather values [B,N,C] by indices [B,M] or [B,M,K]."""

    bsz = values.shape[0]
    if indices.ndim == 2:
        batch = torch.arange(bsz, device=values.device)[:, None]
        return values[batch, indices]
    if indices.ndim == 3:
        batch = torch.arange(bsz, device=values.device)[:, None, None]
        return values[batch, indices]
    raise ValueError(f"unsupported indices shape: {indices.shape}")


class MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                layers.append(nn.GELU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalPatchEncoder(nn.Module):
    """FPS+kNN patch tokenizer with shared local PointNet."""

    def __init__(
        self,
        in_dim: int,
        token_dim: int = 128,
        n_patches: int = 64,
        k_neighbors: int = 32,
    ) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.k_neighbors = k_neighbors
        self.local_mlp = MLP([in_dim + 3, token_dim, token_dim])
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        return_centers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, n_points, _ = points.shape
        center_idx = farthest_point_sample(points, min(self.n_patches, n_points))
        centers = batched_index_select(points, center_idx)
        dists = torch.cdist(centers, points)
        knn = torch.topk(dists, k=min(self.k_neighbors, n_points), largest=False).indices
        neigh_points = batched_index_select(points, knn)
        neigh_features = batched_index_select(features, knn)
        rel = neigh_points - centers[:, :, None, :]
        local = torch.cat([rel, neigh_features], dim=-1)
        token = self.local_mlp(local).max(dim=2).values
        token = self.out_norm(token)
        if return_centers:
            return token, centers
        return token


class CoarseGraphEncoder(nn.Module):
    """Coarse token encoder over FPS nodes with sparse self-attention."""

    def __init__(
        self,
        in_dim: int,
        token_dim: int = 128,
        n_coarse: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_coarse = n_coarse
        self.input = MLP([in_dim, token_dim, token_dim])
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        return_centers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        idx = farthest_point_sample(points, min(self.n_coarse, points.shape[1]))
        coarse = batched_index_select(features, idx)
        token = self.input(coarse)
        token = self.norm(self.encoder(token))
        if return_centers:
            return token, batched_index_select(points, idx)
        return token


class BoundaryTokenEncoder(nn.Module):
    """Single summary token pooled from boundary-marked input points."""

    def __init__(
        self,
        in_dim: int,
        token_dim: int = 128,
        boundary_feature_index: int = -1,
    ) -> None:
        super().__init__()
        self.boundary_feature_index = boundary_feature_index
        self.input = MLP([in_dim, token_dim, token_dim])
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_weights = features[..., self.boundary_feature_index].clamp(0.0, 1.0)
        has_boundary = raw_weights.sum(dim=1, keepdim=True) > 0
        encoded = self.input(features)
        pooled = (encoded * raw_weights[..., None]).sum(dim=1) / raw_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        pooled = self.out_norm(pooled) * has_boundary.to(encoded.dtype)
        return pooled[:, None, :]


class BoundaryPatchEncoder(nn.Module):
    """Learned boundary queries that read boundary-weighted input evidence."""

    def __init__(
        self,
        in_dim: int,
        token_dim: int = 128,
        boundary_feature_index: int = -1,
        n_tokens: int = 8,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if n_tokens <= 0:
            raise ValueError("n_tokens must be positive")
        self.boundary_feature_index = boundary_feature_index
        self.n_tokens = n_tokens
        self.queries = nn.Parameter(torch.randn(n_tokens, token_dim) * 0.02)
        self.input = MLP([in_dim, token_dim, token_dim])
        self.attn = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        bsz = features.shape[0]
        raw_weights = features[..., self.boundary_feature_index].clamp(0.0, 1.0)
        has_boundary = raw_weights.sum(dim=1, keepdim=True) > 0
        encoded = self.input(features) * raw_weights[..., None]
        q = self.queries[None, :, :].expand(bsz, -1, -1)
        q = q + self.attn(self.norm_q(q), self.norm_kv(encoded), self.norm_kv(encoded), need_weights=False)[0]
        q = q + self.ff(q)
        gate = has_boundary.to(q.dtype)[:, :, None]
        return self.out_norm(q) * gate


def make_boundary_encoder(
    in_dim: int,
    token_dim: int,
    boundary_feature_index: int | None,
    boundary_token_count: int = 1,
) -> nn.Module | None:
    if boundary_feature_index is None:
        return None
    if boundary_token_count <= 1:
        return BoundaryTokenEncoder(in_dim, token_dim, boundary_feature_index)
    return BoundaryPatchEncoder(in_dim, token_dim, boundary_feature_index, n_tokens=boundary_token_count)


class LatentFusion(nn.Module):
    """Learnable latent queries that attend to coarse and fine tokens."""

    def __init__(
        self,
        token_dim: int = 128,
        n_latents: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, token_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross": nn.MultiheadAttention(token_dim, n_heads, batch_first=True),
                        "self": nn.MultiheadAttention(token_dim, n_heads, batch_first=True),
                        "norm_q": nn.LayerNorm(token_dim),
                        "norm_kv": nn.LayerNorm(token_dim),
                        "norm_s": nn.LayerNorm(token_dim),
                        "ff": nn.Sequential(
                            nn.LayerNorm(token_dim),
                            nn.Linear(token_dim, token_dim * 4),
                            nn.GELU(),
                            nn.Linear(token_dim * 4, token_dim),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        z = self.latents[None, :, :].expand(bsz, -1, -1)
        for layer in self.layers:
            q = layer["norm_q"](z)
            kv = layer["norm_kv"](tokens)
            z = z + layer["cross"](q, kv, kv, need_weights=False)[0]
            s = layer["norm_s"](z)
            z = z + layer["self"](s, s, s, need_weights=False)[0]
            z = z + layer["ff"](z)
        return z


class TypedCrossAttentionLatticeLayer(nn.Module):
    """One semantic exchange step between global, local, and boundary streams."""

    STREAMS = ("global", "local", "boundary")

    def __init__(self, token_dim: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {name: nn.MultiheadAttention(token_dim, n_heads, batch_first=True) for name in self.STREAMS}
        )
        self.cross_attn = nn.ModuleDict(
            {name: nn.MultiheadAttention(token_dim, n_heads, batch_first=True) for name in self.STREAMS}
        )
        self.norm_self = nn.ModuleDict({name: nn.LayerNorm(token_dim) for name in self.STREAMS})
        self.norm_cross_q = nn.ModuleDict({name: nn.LayerNorm(token_dim) for name in self.STREAMS})
        self.norm_cross_kv = nn.ModuleDict({name: nn.LayerNorm(token_dim) for name in self.STREAMS})
        self.ff = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(token_dim),
                    nn.Linear(token_dim, token_dim * 4),
                    nn.GELU(),
                    nn.Linear(token_dim * 4, token_dim),
                )
                for name in self.STREAMS
            }
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
        boundary_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        streams: dict[str, torch.Tensor | None] = {
            "global": global_tokens,
            "local": local_tokens,
            "boundary": boundary_tokens,
        }
        self_updated = {
            name: self._self_update(name, tokens)
            for name, tokens in streams.items()
            if tokens is not None
        }
        updated: dict[str, torch.Tensor | None] = {}
        for name in self.STREAMS:
            tokens = self_updated.get(name)
            if tokens is None:
                updated[name] = None
                continue
            context = [value for other, value in self_updated.items() if other != name and value is not None]
            updated[name] = self._cross_update(name, tokens, context)
        return updated["global"], updated["local"], updated["boundary"]

    def _self_update(self, name: str, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_self[name](tokens)
        tokens = tokens + self.self_attn[name](normalized, normalized, normalized, need_weights=False)[0]
        return tokens

    def _cross_update(self, name: str, tokens: torch.Tensor, context_parts: list[torch.Tensor]) -> torch.Tensor:
        if context_parts:
            context = torch.cat(context_parts, dim=1)
            tokens = tokens + self.cross_attn[name](
                self.norm_cross_q[name](tokens),
                self.norm_cross_kv[name](context),
                self.norm_cross_kv[name](context),
                need_weights=False,
            )[0]
        return tokens + self.ff[name](tokens)


class TypedCrossAttentionLattice(nn.Module):
    """Semantic interaction network for global, local, and boundary token streams."""

    def __init__(self, token_dim: int = 128, n_layers: int = 1, n_heads: int = 4) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [TypedCrossAttentionLatticeLayer(token_dim=token_dim, n_heads=n_heads) for _ in range(n_layers)]
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        local_tokens: torch.Tensor,
        boundary_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        for layer in self.layers:
            global_tokens, local_tokens, boundary_tokens = layer(global_tokens, local_tokens, boundary_tokens)
        return {
            "global": global_tokens,
            "local": local_tokens,
            "boundary": boundary_tokens,
        }


class PointCloudDecoder(nn.Module):
    """Query decoder from latent set to an unordered reconstructed point cloud."""

    def __init__(
        self,
        token_dim: int = 128,
        n_output: int = 1024,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_output, token_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(token_dim, n_heads, batch_first=True),
                        "norm_q": nn.LayerNorm(token_dim),
                        "norm_kv": nn.LayerNorm(token_dim),
                        "ff": nn.Sequential(
                            nn.LayerNorm(token_dim),
                            nn.Linear(token_dim, token_dim * 4),
                            nn.GELU(),
                            nn.Linear(token_dim * 4, token_dim),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.head = MLP([token_dim, token_dim, 6])

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = latent.shape[0]
        q = self.queries[None, :, :].expand(bsz, -1, -1)
        for layer in self.layers:
            attn = layer["attn"](
                layer["norm_q"](q),
                layer["norm_kv"](latent),
                layer["norm_kv"](latent),
                need_weights=False,
            )[0]
            q = q + attn
            q = q + layer["ff"](q)
        raw = self.head(q)
        xyz = torch.tanh(raw[..., :3]) * 0.75
        normals = F.normalize(raw[..., 3:], dim=-1)
        return xyz, normals


class HierarchicalMidsurfaceAutoencoder(nn.Module):
    """Prototype AE with coarse global and fine local midsurface encoders."""

    def __init__(
        self,
        feature_dim: int = 7,
        token_dim: int = 128,
        n_points_out: int = 1024,
        n_coarse: int = 128,
        n_patches: int = 64,
        k_neighbors: int = 32,
        n_latents: int = 64,
        boundary_feature_index: int | None = None,
        boundary_token_count: int = 1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.boundary_feature_index = boundary_feature_index
        self.boundary_token_count = boundary_token_count
        self.coarse = CoarseGraphEncoder(feature_dim, token_dim, n_coarse=n_coarse)
        self.local = LocalPatchEncoder(feature_dim, token_dim, n_patches=n_patches, k_neighbors=k_neighbors)
        self.boundary = make_boundary_encoder(feature_dim, token_dim, boundary_feature_index, boundary_token_count)
        self.type_embed = nn.Parameter(torch.randn(3 if self.boundary is not None else 2, token_dim) * 0.02)
        self.fusion = LatentFusion(token_dim, n_latents=n_latents)
        self.decoder = PointCloudDecoder(token_dim, n_output=n_points_out)

    def encode(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        coarse = self.coarse(points, features) + self.type_embed[0]
        local = self.local(points, features) + self.type_embed[1]
        tokens = [coarse, local]
        if self.boundary is not None:
            tokens.append(self.boundary(features) + self.type_embed[2])
        tokens = torch.cat(tokens, dim=1)
        return self.fusion(tokens)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(points, features)
        recon, normals = self.decoder(z)
        return {"points": recon, "normals": normals, "latent": z}


class ScaffoldDecoder(nn.Module):
    """Decode coarse scaffold nodes by attending to latent and coarse memory."""

    def __init__(
        self,
        token_dim: int = 128,
        n_scaffold: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_scaffold, token_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(token_dim, n_heads, batch_first=True),
                        "norm_q": nn.LayerNorm(token_dim),
                        "norm_kv": nn.LayerNorm(token_dim),
                        "ff": nn.Sequential(
                            nn.LayerNorm(token_dim),
                            nn.Linear(token_dim, token_dim * 4),
                            nn.GELU(),
                            nn.Linear(token_dim * 4, token_dim),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.xyz_head = MLP([token_dim, token_dim, 3])
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = memory.shape[0]
        q = self.queries[None, :, :].expand(bsz, -1, -1)
        for layer in self.layers:
            attn = layer["attn"](
                layer["norm_q"](q),
                layer["norm_kv"](memory),
                layer["norm_kv"](memory),
                need_weights=False,
            )[0]
            q = q + attn
            q = q + layer["ff"](q)
        scaffold = torch.tanh(self.xyz_head(q)) * 0.75
        return scaffold, self.norm(q)


class AnchoredScaffoldDecoder(nn.Module):
    """Decode scaffold nodes as residuals from encoder-derived anchor centers."""

    def __init__(
        self,
        token_dim: int = 128,
        n_scaffold: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        self.n_scaffold = int(n_scaffold)
        self.residual_scale = float(residual_scale)
        self.anchor_pos = MLP([3, token_dim, token_dim])
        self.slot_embed = nn.Parameter(torch.randn(n_scaffold, token_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(token_dim, n_heads, batch_first=True),
                        "norm_q": nn.LayerNorm(token_dim),
                        "norm_kv": nn.LayerNorm(token_dim),
                        "ff": nn.Sequential(
                            nn.LayerNorm(token_dim),
                            nn.Linear(token_dim, token_dim * 4),
                            nn.GELU(),
                            nn.Linear(token_dim * 4, token_dim),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.residual_head = MLP([token_dim, token_dim, 3])
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        memory: torch.Tensor,
        anchor_points: torch.Tensor,
        anchor_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = anchor_tokens + self.anchor_pos(anchor_points) + self.slot_embed[None, :, :]
        for layer in self.layers:
            attn = layer["attn"](
                layer["norm_q"](q),
                layer["norm_kv"](memory),
                layer["norm_kv"](memory),
                need_weights=False,
            )[0]
            q = q + attn
            q = q + layer["ff"](q)
        residual = torch.tanh(self.residual_head(q)) * self.residual_scale
        scaffold = torch.clamp(anchor_points + residual, min=-0.75, max=0.75)
        return scaffold, self.norm(q), residual


def tangent_basis_from_normals(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a stable orthonormal tangent basis for each normal vector."""

    z_axis = torch.zeros_like(normals)
    z_axis[..., 2] = 1.0
    y_axis = torch.zeros_like(normals)
    y_axis[..., 1] = 1.0
    helper = torch.where(normals[..., 2:3].abs() < 0.9, z_axis, y_axis)
    tangent_u = F.normalize(torch.cross(helper, normals, dim=-1), dim=-1)
    tangent_v = F.normalize(torch.cross(normals, tangent_u, dim=-1), dim=-1)
    return tangent_u, tangent_v


class LocalRefinementDecoder(nn.Module):
    """Refine each scaffold node with nearby fine tokens and local offset queries."""

    def __init__(
        self,
        token_dim: int = 128,
        points_per_scaffold: int = 4,
        n_local_tokens: int = 8,
        n_heads: int = 4,
        refinement_mode: str = "free",
        tangent_offset_scale: float = 0.20,
        normal_offset_scale: float = 0.02,
        patch_type_count: int = 4,
    ) -> None:
        super().__init__()
        if refinement_mode not in {"free", "tangent"}:
            raise ValueError("refinement_mode must be 'free' or 'tangent'")
        if patch_type_count <= 0:
            raise ValueError("patch_type_count must be positive")
        self.points_per_scaffold = points_per_scaffold
        self.n_local_tokens = n_local_tokens
        self.refinement_mode = refinement_mode
        self.tangent_offset_scale = float(tangent_offset_scale)
        self.normal_offset_scale = float(normal_offset_scale)
        self.patch_type_count = int(patch_type_count)
        self.offset_queries = nn.Parameter(torch.randn(points_per_scaffold, token_dim) * 0.02)
        self.scaffold_to_query = nn.Linear(token_dim, token_dim)
        self.local_attn = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim * 4),
            nn.GELU(),
            nn.Linear(token_dim * 4, token_dim),
        )
        if refinement_mode == "tangent":
            self.frame_head = MLP([token_dim, token_dim, 4])
            self.scale_head = MLP([token_dim, token_dim, 3])
            self.patch_type_head = MLP([token_dim, token_dim, self.patch_type_count])
            self.head = MLP([token_dim, token_dim, 4])
        else:
            self.frame_head = None
            self.scale_head = None
            self.patch_type_head = None
            self.head = MLP([token_dim, token_dim, 7])

    def forward(
        self,
        scaffold_points: torch.Tensor,
        scaffold_tokens: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_centers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, n_scaffold, _ = scaffold_points.shape
        k = min(self.n_local_tokens, fine_tokens.shape[1])
        dists = torch.cdist(scaffold_points, fine_centers)
        local_idx = torch.topk(dists, k=k, largest=False).indices
        local_memory = batched_index_select(fine_tokens, local_idx)
        base = self.scaffold_to_query(scaffold_tokens)[:, :, None, :]
        q = base + self.offset_queries[None, None, :, :]
        q_flat = q.reshape(bsz * n_scaffold, self.points_per_scaffold, -1)
        mem_flat = local_memory.reshape(bsz * n_scaffold, k, -1)
        refined = q_flat + self.local_attn(q_flat, mem_flat, mem_flat, need_weights=False)[0]
        refined = refined + self.ff(refined)
        if self.refinement_mode == "tangent":
            raw = self.head(refined).reshape(bsz, n_scaffold, self.points_per_scaffold, 4)
            frame_raw = self.frame_head(scaffold_tokens)
            frame_normals = F.normalize(frame_raw[..., :3], dim=-1)
            tangent_u, tangent_v = tangent_basis_from_normals(frame_normals)
            axis_angle = torch.tanh(frame_raw[..., 3:4]) * math.pi
            cos_angle = torch.cos(axis_angle)
            sin_angle = torch.sin(axis_angle)
            tangent_u, tangent_v = (
                cos_angle * tangent_u + sin_angle * tangent_v,
                -sin_angle * tangent_u + cos_angle * tangent_v,
            )
            patch_scales = torch.sigmoid(self.scale_head(scaffold_tokens))
            max_scales = patch_scales.new_tensor(
                [self.tangent_offset_scale, self.tangent_offset_scale, self.normal_offset_scale]
            )
            patch_scales = patch_scales * max_scales
            tangent_xy = torch.tanh(raw[..., :2]) * patch_scales[:, :, None, :2]
            normal_offset = torch.tanh(raw[..., 2:3]) * patch_scales[:, :, None, 2:3]
            tangent_u = tangent_u[:, :, None, :]
            tangent_v = tangent_v[:, :, None, :]
            frame_normals_expanded = frame_normals[:, :, None, :]
            offsets = (
                tangent_xy[..., 0:1] * tangent_u
                + tangent_xy[..., 1:2] * tangent_v
                + normal_offset * frame_normals_expanded
            )
            normals = frame_normals_expanded.expand(-1, -1, self.points_per_scaffold, -1)
            refinement_logits = raw[..., 3]
        else:
            raw = self.head(refined).reshape(bsz, n_scaffold, self.points_per_scaffold, 7)
            offsets = torch.tanh(raw[..., :3]) * 0.20
            normals = F.normalize(raw[..., 3:6], dim=-1)
            refinement_logits = raw[..., 6]
        points = scaffold_points[:, :, None, :] + offsets
        out = {
            "points": points.reshape(bsz, n_scaffold * self.points_per_scaffold, 3),
            "normals": normals.reshape(bsz, n_scaffold * self.points_per_scaffold, 3),
            "refinement_logits": refinement_logits,
            "local_indices": local_idx,
        }
        if self.refinement_mode == "tangent":
            out["refinement_frame_normals"] = frame_normals
            out["refinement_frame_tangent_u"] = tangent_u[:, :, 0, :]
            out["refinement_frame_tangent_v"] = tangent_v[:, :, 0, :]
            out["patch_scales"] = patch_scales
            out["patch_type_logits"] = self.patch_type_head(scaffold_tokens)
        return out


class StructuredScaffoldAutoencoder(nn.Module):
    """AE with coarse scaffold decoding followed by local fine-token refinement."""

    def __init__(
        self,
        feature_dim: int = 7,
        token_dim: int = 128,
        n_points_out: int = 1024,
        n_coarse: int = 128,
        n_patches: int = 64,
        k_neighbors: int = 32,
        n_latents: int = 64,
        n_scaffold: int = 64,
        points_per_scaffold: int | None = None,
        n_local_tokens: int = 8,
        boundary_feature_index: int | None = None,
        boundary_token_count: int = 1,
        lattice_layers: int = 0,
        lattice_heads: int = 4,
        refinement_mode: str = "free",
        tangent_offset_scale: float = 0.20,
        normal_offset_scale: float = 0.02,
        patch_type_count: int = 4,
        scaffold_mode: str = "learned",
        scaffold_anchor_source: str = "coarse_fine",
        scaffold_anchor_residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if n_scaffold <= 0:
            raise ValueError("n_scaffold must be positive")
        if lattice_layers < 0:
            raise ValueError("lattice_layers must be non-negative")
        if scaffold_mode not in {"learned", "anchored"}:
            raise ValueError("scaffold_mode must be 'learned' or 'anchored'")
        if scaffold_anchor_source not in {"coarse", "fine", "coarse_fine"}:
            raise ValueError("scaffold_anchor_source must be 'coarse', 'fine', or 'coarse_fine'")
        self.feature_dim = feature_dim
        self.boundary_feature_index = boundary_feature_index
        self.boundary_token_count = boundary_token_count
        self.coarse = CoarseGraphEncoder(feature_dim, token_dim, n_coarse=n_coarse)
        self.local = LocalPatchEncoder(feature_dim, token_dim, n_patches=n_patches, k_neighbors=k_neighbors)
        self.boundary = make_boundary_encoder(feature_dim, token_dim, boundary_feature_index, boundary_token_count)
        self.type_embed = nn.Parameter(torch.randn(4 if self.boundary is not None else 3, token_dim) * 0.02)
        self.n_lattice_layers = lattice_layers
        self.lattice_heads = lattice_heads
        self.refinement_mode = refinement_mode
        self.tangent_offset_scale = float(tangent_offset_scale)
        self.normal_offset_scale = float(normal_offset_scale)
        self.patch_type_count = int(patch_type_count)
        self.scaffold_mode = scaffold_mode
        self.scaffold_anchor_source = scaffold_anchor_source
        self.scaffold_anchor_residual_scale = float(scaffold_anchor_residual_scale)
        self.lattice = (
            TypedCrossAttentionLattice(token_dim=token_dim, n_layers=lattice_layers, n_heads=lattice_heads)
            if lattice_layers > 0
            else None
        )
        self.fusion = LatentFusion(token_dim, n_latents=n_latents)
        self.n_scaffold = n_scaffold
        self.points_per_scaffold = points_per_scaffold or max(1, math.ceil(n_points_out / n_scaffold))
        if self.points_per_scaffold <= 0:
            raise ValueError("points_per_scaffold must be positive")
        self.n_generated_points = n_scaffold * self.points_per_scaffold
        if self.n_generated_points < n_points_out:
            raise ValueError("n_scaffold * points_per_scaffold must cover n_points_out")
        self.n_points_out = n_points_out
        if scaffold_mode == "anchored":
            self.scaffold_decoder = AnchoredScaffoldDecoder(
                token_dim=token_dim,
                n_scaffold=n_scaffold,
                residual_scale=scaffold_anchor_residual_scale,
            )
        else:
            self.scaffold_decoder = ScaffoldDecoder(token_dim=token_dim, n_scaffold=n_scaffold)
        self.refine_decoder = LocalRefinementDecoder(
            token_dim=token_dim,
            points_per_scaffold=self.points_per_scaffold,
            n_local_tokens=n_local_tokens,
            refinement_mode=refinement_mode,
            tangent_offset_scale=tangent_offset_scale,
            normal_offset_scale=normal_offset_scale,
            patch_type_count=patch_type_count,
        )

    def scaffold_activity(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.arange(self.n_scaffold, device=device) * self.points_per_scaffold
        counts = (self.n_points_out - starts).clamp(min=0, max=self.points_per_scaffold).long()
        return counts > 0, counts

    def encode_memory(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        coarse, coarse_centers = self.coarse(points, features, return_centers=True)
        fine, fine_centers = self.local(points, features, return_centers=True)
        coarse = coarse + self.type_embed[0]
        fine = fine + self.type_embed[1]
        fusion_tokens = [coarse, fine]
        latent_type_index = 2
        boundary = None
        if self.boundary is not None:
            boundary = self.boundary(features) + self.type_embed[2]
            fusion_tokens.append(boundary)
            latent_type_index = 3
        if self.lattice is not None:
            lattice = self.lattice(coarse, fine, boundary)
            coarse = lattice["global"]
            fine = lattice["local"]
            boundary = lattice["boundary"]
            fusion_tokens = [coarse, fine]
            if boundary is not None:
                fusion_tokens.append(boundary)
        latent = self.fusion(torch.cat(fusion_tokens, dim=1)) + self.type_embed[latent_type_index]
        return {
            "coarse": coarse,
            "coarse_centers": coarse_centers,
            "fine": fine,
            "fine_centers": fine_centers,
            "latent": latent,
            "boundary": boundary,
        }

    def select_scaffold_anchors(self, memory: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        centers: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        if self.scaffold_anchor_source in {"coarse", "coarse_fine"}:
            centers.append(memory["coarse_centers"])
            tokens.append(memory["coarse"])
        if self.scaffold_anchor_source in {"fine", "coarse_fine"}:
            centers.append(memory["fine_centers"])
            tokens.append(memory["fine"])
        candidate_centers = torch.cat(centers, dim=1)
        candidate_tokens = torch.cat(tokens, dim=1)
        n_candidates = candidate_centers.shape[1]
        if n_candidates >= self.n_scaffold:
            anchor_idx = farthest_point_sample(candidate_centers, self.n_scaffold)
            return (
                batched_index_select(candidate_centers, anchor_idx),
                batched_index_select(candidate_tokens, anchor_idx),
            )
        repeats = math.ceil(self.n_scaffold / n_candidates)
        anchor_points = candidate_centers.repeat(1, repeats, 1)[:, : self.n_scaffold, :]
        anchor_tokens = candidate_tokens.repeat(1, repeats, 1)[:, : self.n_scaffold, :]
        return anchor_points, anchor_tokens

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encode_memory(points, features)
        scaffold_memory_parts = [memory["latent"], memory["coarse"]]
        if memory["boundary"] is not None:
            scaffold_memory_parts.append(memory["boundary"])
        scaffold_memory = torch.cat(scaffold_memory_parts, dim=1)
        scaffold_anchor_points = None
        scaffold_residuals = None
        if self.scaffold_mode == "anchored":
            scaffold_anchor_points, scaffold_anchor_tokens = self.select_scaffold_anchors(memory)
            scaffold_points, scaffold_tokens, scaffold_residuals = self.scaffold_decoder(
                scaffold_memory,
                scaffold_anchor_points,
                scaffold_anchor_tokens,
            )
        else:
            scaffold_points, scaffold_tokens = self.scaffold_decoder(scaffold_memory)
        refined = self.refine_decoder(
            scaffold_points=scaffold_points,
            scaffold_tokens=scaffold_tokens,
            fine_tokens=memory["fine"],
            fine_centers=memory["fine_centers"],
        )
        active_scaffold_mask, scaffold_point_counts = self.scaffold_activity(points.device)
        bsz = points.shape[0]
        out = {
            "points": refined["points"][:, : self.n_points_out, :],
            "normals": refined["normals"][:, : self.n_points_out, :],
            "latent": memory["latent"],
            "scaffold_points": scaffold_points,
            "scaffold_tokens": scaffold_tokens,
            "refinement_logits": refined["refinement_logits"].reshape(bsz, self.n_generated_points)[
                :, : self.n_points_out
            ],
            "refinement_logits_grid": refined["refinement_logits"],
            "active_scaffold_mask": active_scaffold_mask[None, :].expand(bsz, -1),
            "scaffold_point_counts": scaffold_point_counts[None, :].expand(bsz, -1),
            "local_indices": refined["local_indices"],
        }
        if scaffold_anchor_points is not None and scaffold_residuals is not None:
            out["scaffold_anchor_points"] = scaffold_anchor_points
            out["scaffold_residuals"] = scaffold_residuals
        if "refinement_frame_normals" in refined:
            out["refinement_frame_normals"] = refined["refinement_frame_normals"]
            out["refinement_frame_tangent_u"] = refined["refinement_frame_tangent_u"]
            out["refinement_frame_tangent_v"] = refined["refinement_frame_tangent_v"]
            out["patch_scales"] = refined["patch_scales"]
            out["patch_type_logits"] = refined["patch_type_logits"]
        return out


def chamfer_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_to_target, target_to_pred = chamfer_components(pred, target)
    return pred_to_target + target_to_pred


def chamfer_components(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(pred, target)
    return dist.min(dim=2).values.mean(), dist.min(dim=1).values.mean()


def target_coverage_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_mm: torch.Tensor,
    threshold_mm: float,
) -> torch.Tensor:
    """Penalize target points that remain farther than a world-mm threshold."""

    dist = torch.cdist(target, pred)
    target_to_pred = dist.min(dim=2).values
    threshold = threshold_mm / scale_mm.clamp_min(1.0e-6)
    while threshold.ndim < target_to_pred.ndim:
        threshold = threshold.unsqueeze(-1)
    return torch.relu(target_to_pred - threshold).mean()


def target_coverage_fraction(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_mm: torch.Tensor,
    threshold_mm: float,
) -> torch.Tensor:
    dist = torch.cdist(target, pred)
    target_to_pred = dist.min(dim=2).values
    threshold = threshold_mm / scale_mm.clamp_min(1.0e-6)
    while threshold.ndim < target_to_pred.ndim:
        threshold = threshold.unsqueeze(-1)
    return (target_to_pred <= threshold).float().mean()


def prediction_surface_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_mm: torch.Tensor,
    threshold_mm: float,
) -> torch.Tensor:
    """Penalize generated points that drift away from the target surface."""

    dist = torch.cdist(pred, target)
    pred_to_target = dist.min(dim=2).values
    threshold = threshold_mm / scale_mm.clamp_min(1.0e-6)
    while threshold.ndim < pred_to_target.ndim:
        threshold = threshold.unsqueeze(-1)
    return torch.relu(pred_to_target - threshold).mean()


def prediction_surface_fraction(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_mm: torch.Tensor,
    threshold_mm: float,
) -> torch.Tensor:
    dist = torch.cdist(pred, target)
    pred_to_target = dist.min(dim=2).values
    threshold = threshold_mm / scale_mm.clamp_min(1.0e-6)
    while threshold.ndim < pred_to_target.ndim:
        threshold = threshold.unsqueeze(-1)
    return (pred_to_target <= threshold).float().mean()


def scaffold_chamfer_loss(
    scaffold: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise scaffold points against FPS pseudo-scaffold targets."""

    if active_mask is not None:
        active = active_mask[0].bool()
        scaffold = scaffold[:, active, :]
    if scaffold.shape[1] == 0:
        return scaffold.sum() * 0.0
    target_idx = farthest_point_sample(target, scaffold.shape[1])
    target_scaffold = batched_index_select(target, target_idx)
    return chamfer_loss(scaffold, target_scaffold)


def normal_chamfer_loss(
    pred_points: torch.Tensor,
    pred_normals: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
) -> torch.Tensor:
    """Nearest-surface normal agreement with orientation sign ignored."""

    dist = torch.cdist(pred_points, target_points)
    pred_to_target = dist.min(dim=2).indices
    matched_target_normals = batched_index_select(target_normals, pred_to_target)
    cos = torch.sum(pred_normals * matched_target_normals, dim=-1).abs().clamp(max=1.0)
    return (1.0 - cos).mean()


def spread_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Weak bbox/statistics regularizer to stabilize early AE training."""

    pred_mean = pred.mean(dim=1)
    target_mean = target.mean(dim=1)
    pred_std = pred.std(dim=1)
    target_std = target.std(dim=1)
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
