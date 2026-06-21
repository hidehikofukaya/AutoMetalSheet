"""Independent VecSet model for softmin-derived surface guidance.

The encoder is intentionally assembled from the existing ``vecset_ae``
tokenizer/encoder components.  Only the query decoder and its output contract
are specific to this model.

All coordinate-valued predictions are in the model's normalized coordinate
system.  Dataset targets remain in world millimetres; callers convert predicted
``signed_potential`` and ``step_distance`` to millimetres by multiplying by the
per-part ``scale`` returned by :mod:`softmin_guidance_dataset`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .vecset_ae import (
        FourierFeatures,
        LatentCompressor,
        PointTokenizer,
        PositionalMLP,
        TransformerEncoder,
    )
except ImportError:  # Support direct imports when src/model is on sys.path.
    from vecset_ae import (  # type: ignore[no-redef]
        FourierFeatures,
        LatentCompressor,
        PointTokenizer,
        PositionalMLP,
        TransformerEncoder,
    )


CHECKPOINT_SCHEMA_VERSION = "model.softmin_guidance.v1"
STAGE_A_SCHEMA_VERSION = "stage_a.softmin_guidance.v1"


class SoftminGuidancePrediction(NamedTuple):
    """Per-query output returned by :meth:`SoftminGuidanceModel.decode`."""

    soft_potential: torch.Tensor
    step_distance: torch.Tensor
    direction_toward_surface: torch.Tensor
    branch_ambiguity: torch.Tensor

    @property
    def signed_potential(self) -> torch.Tensor:
        """Semantic alias emphasizing that ``soft_potential`` is signed."""

        return self.soft_potential

    @property
    def toward_surface(self) -> torch.Tensor:
        return self.direction_toward_surface

    @property
    def ambiguity(self) -> torch.Tensor:
        return self.branch_ambiguity


class SoftminGuidanceForwardOutput(NamedTuple):
    """Full output returned by :meth:`SoftminGuidanceModel.forward`."""

    soft_potential: torch.Tensor
    step_distance: torch.Tensor
    direction_toward_surface: torch.Tensor
    branch_ambiguity: torch.Tensor
    latent: torch.Tensor

    @property
    def signed_potential(self) -> torch.Tensor:
        return self.soft_potential

    @property
    def toward_surface(self) -> torch.Tensor:
        return self.direction_toward_surface

    @property
    def ambiguity(self) -> torch.Tensor:
        return self.branch_ambiguity


def _unit_vector_with_fallback(
    vector: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Normalize vectors and use +X for the exactly-zero, undefined case."""

    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    normalized = vector / norm.clamp_min(eps)
    fallback = torch.zeros_like(vector)
    fallback[..., 0] = 1.0
    return torch.where(norm > eps, normalized, fallback)


class SoftminGuidanceDecoder(nn.Module):
    """Query cross-attention decoder with four semantically separate heads."""

    def __init__(
        self,
        dim: int = 384,
        n_heads: int = 6,
        n_layers: int = 2,
        pos_hidden: int = 128,
        dropout: float = 0.0,
        n_freqs: int = 8,
    ):
        super().__init__()
        self.fourier = FourierFeatures(n_freqs=n_freqs, include_input=True)
        self.query_embed = PositionalMLP(
            dim, hidden=pos_hidden, in_ch=self.fourier.out_dim
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_q": nn.LayerNorm(dim),
                        "norm_kv": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(
                            dim, n_heads, dropout=dropout, batch_first=True
                        ),
                        "ff": nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 6),
        )

        # Start the positive step head close to zero, as in VecSetAE's UDF
        # head, while keeping potential signed and ambiguity neutral.
        with torch.no_grad():
            self.head[-1].bias[1] = -5.0

    def forward(
        self, query_xyz: torch.Tensor, latent: torch.Tensor
    ) -> SoftminGuidancePrediction:
        query = self.query_embed(self.fourier(query_xyz))
        for layer in self.layers:
            query_norm = layer["norm_q"](query)
            latent_norm = layer["norm_kv"](latent)
            attention, _ = layer["attn"](
                query_norm, latent_norm, latent_norm, need_weights=False
            )
            query = query + attention
            query = query + layer["ff"](query)

        raw = self.head(query)
        soft_potential = raw[..., 0]
        step_distance = F.softplus(raw[..., 1]) + torch.finfo(raw.dtype).eps
        direction_toward_surface = _unit_vector_with_fallback(raw[..., 2:5])
        branch_ambiguity = torch.sigmoid(raw[..., 5])
        return SoftminGuidancePrediction(
            soft_potential=soft_potential,
            step_distance=step_distance,
            direction_toward_surface=direction_toward_surface,
            branch_ambiguity=branch_ambiguity,
        )


class SoftminGuidanceModel(nn.Module):
    """VecSet encoder plus a dedicated softmin-guidance query decoder."""

    def __init__(
        self,
        token_dim: int = 384,
        n_tokens: int = 256,
        k_neighbors: int = 32,
        n_latents: int = 128,
        enc_layers: int = 6,
        dec_layers: int = 2,
        n_heads: int = 6,
        use_normals_in_encoder: bool = True,
        dropout: float = 0.0,
        n_freqs: int = 8,
    ):
        super().__init__()
        if token_dim % n_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.use_normals = use_normals_in_encoder
        self._architecture = {
            "token_dim": token_dim,
            "n_tokens": n_tokens,
            "k_neighbors": k_neighbors,
            "n_latents": n_latents,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "n_heads": n_heads,
            "use_normals_in_encoder": use_normals_in_encoder,
            "dropout": dropout,
            "n_freqs": n_freqs,
        }

        extra_channels = 3 if use_normals_in_encoder else 0
        self.tokenizer = PointTokenizer(
            n_centers=n_tokens,
            k=k_neighbors,
            in_extra_ch=extra_channels,
            token_dim=token_dim,
        )
        self.token_pos_enc = PositionalMLP(token_dim)
        self.encoder = TransformerEncoder(
            dim=token_dim,
            n_layers=enc_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.compressor = LatentCompressor(
            dim=token_dim,
            n_latents=n_latents,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.decoder = SoftminGuidanceDecoder(
            dim=token_dim,
            n_heads=n_heads,
            n_layers=dec_layers,
            dropout=dropout,
            n_freqs=n_freqs,
        )

    def _validate_encoder_inputs(
        self, points: torch.Tensor, normals: torch.Tensor | None
    ) -> None:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [B,N,3], got {points.shape}")
        if self.tokenizer.n_centers > points.shape[1]:
            raise ValueError("n_tokens cannot exceed the number of input points")
        if self.tokenizer.k > points.shape[1]:
            raise ValueError("k_neighbors cannot exceed the number of input points")
        if self.use_normals:
            if normals is None:
                raise ValueError("normals are required when use_normals_in_encoder=True")
            if normals.shape != points.shape:
                raise ValueError(
                    f"normals must match points shape {points.shape}, got {normals.shape}"
                )

    def encode(
        self, points: torch.Tensor, normals: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._validate_encoder_inputs(points, normals)
        features = normals if self.use_normals else None
        tokens, center_xyz = self.tokenizer(points, features)
        tokens = tokens + self.token_pos_enc(center_xyz)
        tokens = self.encoder(tokens)
        return self.compressor(tokens)

    def decode(
        self, query_xyz: torch.Tensor, latent: torch.Tensor
    ) -> SoftminGuidancePrediction:
        if query_xyz.ndim != 3 or query_xyz.shape[-1] != 3:
            raise ValueError(
                f"query_xyz must have shape [B,Nq,3], got {query_xyz.shape}"
            )
        if latent.ndim != 3 or latent.shape[0] != query_xyz.shape[0]:
            raise ValueError(
                "latent must have shape [B,L,D] with the same batch size as query_xyz"
            )
        return self.decoder(query_xyz, latent)

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor | None,
        query_xyz: torch.Tensor,
    ) -> SoftminGuidanceForwardOutput:
        latent = self.encode(points, normals)
        prediction = self.decode(query_xyz, latent)
        return SoftminGuidanceForwardOutput(
            soft_potential=prediction.soft_potential,
            step_distance=prediction.step_distance,
            direction_toward_surface=prediction.direction_toward_surface,
            branch_ambiguity=prediction.branch_ambiguity,
            latent=latent,
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata to store beside ``state_dict``."""

        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage_a_schema_version": STAGE_A_SCHEMA_VERSION,
            "model_type": type(self).__name__,
            "architecture": dict(self._architecture),
            "input_contract": {
                "points": "normalized_part_coordinates",
                "normals": "unit_vectors" if self.use_normals else "unused",
                "query_xyz": "normalized_part_coordinates",
                "world_transform": "world_xyz = normalized_xyz * scale + center",
            },
            "output_contract": {
                "soft_potential": {
                    "activation": "linear",
                    "signed": True,
                    "units": "normalized_coordinate_units",
                },
                "step_distance": {
                    "activation": "softplus",
                    "range": "(0, inf)",
                    "units": "normalized_coordinate_units",
                },
                "direction_toward_surface": {
                    "activation": "unit_normalize",
                    "gradient_convention": "toward_surface",
                },
                "branch_ambiguity": {
                    "activation": "sigmoid",
                    "range": "[0, 1]",
                },
            },
        }


def build_checkpoint_metadata(model: SoftminGuidanceModel) -> dict[str, Any]:
    """Convenience helper for checkpoint writers."""

    return model.checkpoint_metadata()
