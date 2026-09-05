"""
vecset_ae.py -- Stage B: VecSet Autoencoder for mid-surface shape reconstruction
(CLAUDE.md section 13, Round 6 PoC architecture).

Architecture
------------
    points[N,3] (+normals[N,3])
        --FPS(M=256 centers)+kNN(k=32)--> mini-PointNet --> tokens[256,token_dim]
        --Self-Attn Encoder x6L-->
        --Cross-Attn (128 learnable queries)--> Z[128,token_dim]   (latent shape code)

    query_xyz[Nq,3]
        --Cross-Attn Decoder (query attends to Z) x2L--> MLP head
        --> udf(query) >= 0,  grad(query) (unit vector)

No thickness decoder head (R6-06): thickness is a single scalar-per-part
estimate recorded in dataset metadata, not a learned per-point field.

R7-05: QueryDecoder applies a NeRF-style Fourier positional encoding to
query_xyz before its PositionalMLP, to counter coordinate-MLP spectral bias
that otherwise caps near-surface resolution at a coarse/global level
(confirmed empirically -- see FourierFeatures docstring below).

"grad(query)" supervises the UDF gradient direction (unit vector from the
query point toward its nearest mid-surface point), NOT a surface normal in
the classical sense -- an unsigned field has no well-defined normal sign
away from the surface. At distance ~0 this converges to the true surface
normal. This is the field Stage C's gradient-projection mesh extraction
needs. See R7-01 in CLAUDE.md section 13.

FPS / kNN are implemented in pure PyTorch. Neither pytorch3d nor
torch-cluster is installed in this conda environment (verified at Stage B
kickoff: only `torch` itself is available, 2.6.0+cu124, CUDA available on a
GTX 1660 SUPER). At this problem's scale (N=8192 points, M=256 centers) a
naive O(N*M) FPS loop and an O(M*N) cdist+topk kNN are both fast enough on
GPU, so no extra dependency was added.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# FPS / kNN (pure PyTorch -- no pytorch3d / torch-cluster, see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def furthest_point_sampling(
    xyz: torch.Tensor, n_samples: int, deterministic_start: bool = False
) -> torch.Tensor:
    """
    xyz: [B, N, 3]
    returns idx: [B, n_samples] long -- indices of the sampled points.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    dist = torch.full((B, N), float("inf"), device=device)
    if deterministic_start:
        center = xyz.mean(dim=1, keepdim=True)
        farthest = torch.sum((xyz - center) ** 2, dim=-1).argmax(dim=-1)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)
    for i in range(n_samples):
        idx[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)          # [B,1,3]
        d = torch.sum((xyz - centroid) ** 2, dim=-1)                  # [B,N]
        dist = torch.minimum(dist, d)
        farthest = torch.argmax(dist, dim=-1)
    return idx


def knn_gather_idx(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz:     [B, N, 3]  full point set to search in
    centers: [B, M, 3]  query centers
    returns idx: [B, M, k] long -- indices into xyz of the k nearest neighbors.
    """
    d = torch.cdist(centers, xyz)                          # [B, M, N]
    return d.topk(k, dim=-1, largest=False).indices         # [B, M, k]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: [B, N, C]
    idx:    [B, ...]  long indices into the N dimension
    returns [B, ..., C]
    """
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


# ─────────────────────────────────────────────────────────────────────────────
# mini-PointNet local tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class MiniPointNet(nn.Module):
    """Per-group local feature extractor: [B,M,k,Cin] -> token [B,M,Cout]."""

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 128):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden * 2, out_ch), nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.mlp1(x)                                          # [B,M,k,hidden]
        global_feat = feat.max(dim=2, keepdim=True).values            # [B,M,1,hidden]
        feat = torch.cat([feat, global_feat.expand_as(feat)], dim=-1)  # [B,M,k,2*hidden]
        feat = self.mlp2(feat)                                         # [B,M,k,out_ch]
        return feat.max(dim=2).values                                  # [B,M,out_ch]


class PointTokenizer(nn.Module):
    """FPS(M centers) + kNN(k) grouping + mini-PointNet -> per-group tokens."""

    def __init__(self, n_centers: int = 256, k: int = 32,
                 in_extra_ch: int = 3, token_dim: int = 384, hidden: int = 128,
                 deterministic_fps: bool = False):
        super().__init__()
        self.n_centers = n_centers
        self.k = k
        self.deterministic_fps = deterministic_fps
        self.pointnet = MiniPointNet(in_ch=3 + in_extra_ch, out_ch=token_dim, hidden=hidden)

    def forward(self, points: torch.Tensor, feats: torch.Tensor | None = None):
        """
        points: [B,N,3]
        feats:  [B,N,Cf] optional extra per-point features (e.g. normals)
        returns tokens [B,n_centers,token_dim], center_xyz [B,n_centers,3]
        """
        center_idx = furthest_point_sampling(
            points, self.n_centers, deterministic_start=self.deterministic_fps
        )                                                               # [B,M]
        center_xyz = index_points(points, center_idx)                   # [B,M,3]
        knn_idx = knn_gather_idx(points, center_xyz, self.k)             # [B,M,k]
        grouped_xyz = index_points(points, knn_idx)                      # [B,M,k,3]
        rel_xyz = grouped_xyz - center_xyz.unsqueeze(2)

        if feats is not None:
            grouped_feats = index_points(feats, knn_idx)                 # [B,M,k,Cf]
            group_in = torch.cat([rel_xyz, grouped_feats], dim=-1)
        else:
            group_in = rel_xyz

        tokens = self.pointnet(group_in)
        return tokens, center_xyz


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding, transformer encoder, latent compressor, query decoder
# ─────────────────────────────────────────────────────────────────────────────

class FourierFeatures(nn.Module):
    """
    R7-05: NeRF-style positional encoding -- [x, sin(2^k*pi*x), cos(2^k*pi*x)]
    for k=0..n_freqs-1 -- prepended to raw query xyz before QueryDecoder's
    PositionalMLP.

    A plain Linear(3,hidden) on raw normalized xyz suffers the well-known
    coordinate-MLP "spectral bias" (Rahaman et al. 2019; Tancik et al. 2020,
    "Fourier Features Let Networks Learn High Frequency Functions"): it
    cannot resolve spatial detail finer than some coarse intrinsic
    frequency, regardless of training duration or loss weighting. Confirmed
    empirically here: across three independent training configurations
    (1500-epoch unclamped, broken-init clamped, fixed-init clamped -- see
    R7-03/R7-04), near-surface UDF MAE plateaued at an essentially identical
    ~0.42-0.44mm. That value is suspiciously close to the theoretical MAE of
    a model that ignores local position entirely and predicts the
    near-band's median (~0.5mm for a uniform [0,2mm) target) -- i.e. the
    decoder was not resolving per-point local geometry at all, only a
    coarse/global field.

    n_freqs=8 default: highest frequency 2^7*pi ~ 400 in normalized
    [-1,1]^3 space, matched to needing to resolve ~1mm features when
    1 normalized unit ~ 300-450mm (R7-02 per-part scale) -> 1mm ~ 0.002-0.003
    normalized units -> wavelength of the highest frequency band should be
    on that order.
    """

    def __init__(self, n_freqs: int = 8, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        freq_bands = (2.0 ** torch.arange(n_freqs)) * torch.pi    # [n_freqs]
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.out_dim = 3 * ((1 if include_input else 0) + 2 * n_freqs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        feats = [xyz] if self.include_input else []
        for f in self.freq_bands:
            feats.append(torch.sin(xyz * f))
            feats.append(torch.cos(xyz * f))
        return torch.cat(feats, dim=-1)


class PositionalMLP(nn.Module):
    """Learned positional embedding MLP. `in_ch` lets callers feed raw xyz
    (3) or Fourier-expanded xyz (see FourierFeatures, used by QueryDecoder
    for R7-05)."""

    def __init__(self, dim: int, hidden: int = 128, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(xyz)


class TransformerEncoder(nn.Module):
    """Self-attention encoder stack, pre-norm, batch_first."""

    def __init__(self, dim: int = 384, n_layers: int = 6, n_heads: int = 6,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LatentCompressor(nn.Module):
    """Cross-attention: n_latents learnable queries attend to the input
    tokens, compressing 256 tokens -> Z[n_latents, dim]."""

    def __init__(self, dim: int = 384, n_latents: int = 128, n_heads: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        q0 = self.latents.unsqueeze(0).expand(B, -1, -1)        # [B,M,dim]
        q = self.norm_q(q0)
        kv = self.norm_kv(tokens)
        attn_out, _ = self.attn(q, kv, kv)
        z = q0 + attn_out
        z = z + self.ff(z)
        return z


class QueryDecoder(nn.Module):
    """Cross-attention decoder: query points attend to the latent code Z,
    followed by an MLP head producing (udf, grad_dir) per query."""

    def __init__(self, dim: int = 384, n_heads: int = 6, n_layers: int = 2,
                 pos_hidden: int = 128, dropout: float = 0.0, n_freqs: int = 8):
        super().__init__()
        self.fourier = FourierFeatures(n_freqs=n_freqs, include_input=True)
        self.query_embed = PositionalMLP(dim, hidden=pos_hidden, in_ch=self.fourier.out_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim),
                "norm_kv": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True),
                "ff": nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                ),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 5),
        )
        # R7-10: channel 4 is a confidence/uncertainty logit w in [0,1]
        # (sigmoid), trained independently of the UDF regression (own BCE
        # loss in train.py, target = clamp(1 - gt_udf/horizon_mm, 0, 1)).
        # Default zero-init bias -> sigmoid(0)=0.5 is a neutral starting
        # point, unlike channel 0 below which needs a deliberate negative
        # bias to avoid a training-stall bug (R7-04).
        #
        # Purpose (CLAUDE.md TD-16): the raw UDF regression itself produces
        # "ghost" near-zero predictions in the extrapolated region past an
        # open mid-surface boundary (the cut locus is a non-differentiable
        # kink no smooth coordinate-MLP can represent exactly -- see
        # arxiv.org/html/2306.02099v4's uncertainty-augmented SDF/UDF
        # design, which this head follows). A SEPARATE head with its own
        # classification-style loss is hypothesized to be more numerically
        # well-behaved near that singularity than thresholding the
        # regressed distance value directly, since it only needs the right
        # side of a soft threshold, not the exact extrapolated magnitude.
        # Stage C (reconstruct.py) gates candidate selection on this output
        # instead of/in addition to raw UDF rank (R7-06).
        # R7-04: bias the UDF logit so softplus(out[...,0]) starts near 0 at
        # init. Default nn.Linear init leaves out[...,0]~0, and
        # softplus(0)=ln(2)~0.693 -- after the *scale rescale to mm (R7-02,
        # scale ~300-450mm for this dataset) every query point predicts
        # ~200-300mm regardless of true distance, near or far alike.
        # Confirmed empirically: epoch-1 near_mae was ~381mm in both the
        # unclamped and (R7-03) clamped overfit runs. Worse, with the R7-03
        # truncated-distance clamp this is fatal, not just slow: clamp()'s
        # gradient is exactly zero outside the clip band, so predictions
        # starting ~300mm above a ~5mm clip threshold get zero gradient and
        # training stalls completely (confirmed: a 600-epoch run with this
        # bug showed udf_l1 flat at ~4.1 the entire run, near_mae *rising*
        # to ~390mm). Initializing the bias negative makes softplus(out)~0
        # at init -- inside the clip band for every point -- so the network
        # has to learn to push values UP for far points (clear, well-posed
        # gradient direction) instead of down from an arbitrary high offset.
        with torch.no_grad():
            self.head[-1].bias[0] = -5.0

    def forward(self, query_xyz: torch.Tensor, z: torch.Tensor):
        q = self.query_embed(self.fourier(query_xyz))    # [B,Nq,dim]
        for l in self.layers:
            qn = l["norm_q"](q)
            kvn = l["norm_kv"](z)
            attn_out, _ = l["attn"](qn, kvn, kvn)
            q = q + attn_out
            q = q + l["ff"](q)
        out = self.head(q)                               # [B,Nq,5]
        udf = F.softplus(out[..., 0])                     # UDF is non-negative by definition
        grad = F.normalize(out[..., 1:4], dim=-1, eps=1e-8)
        conf = torch.sigmoid(out[..., 4])                 # R7-10: confidence w in [0,1]
        return udf, grad, conf


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class VecSetAE(nn.Module):
    def __init__(self, token_dim: int = 384, n_tokens: int = 256, k_neighbors: int = 32,
                 n_latents: int = 128, enc_layers: int = 6, dec_layers: int = 2,
                 n_heads: int = 6, use_normals_in_encoder: bool = True,
                 dropout: float = 0.0, n_freqs: int = 8):
        super().__init__()
        self.use_normals = use_normals_in_encoder
        extra_ch = 3 if use_normals_in_encoder else 0
        self.tokenizer = PointTokenizer(n_centers=n_tokens, k=k_neighbors,
                                         in_extra_ch=extra_ch, token_dim=token_dim)
        self.token_pos_enc = PositionalMLP(token_dim)
        self.encoder = TransformerEncoder(dim=token_dim, n_layers=enc_layers,
                                           n_heads=n_heads, dropout=dropout)
        self.compressor = LatentCompressor(dim=token_dim, n_latents=n_latents,
                                            n_heads=n_heads, dropout=dropout)
        self.decoder = QueryDecoder(dim=token_dim, n_heads=n_heads,
                                     n_layers=dec_layers, dropout=dropout, n_freqs=n_freqs)

    def encode(self, points: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        feats = normals if self.use_normals else None
        tokens, center_xyz = self.tokenizer(points, feats)
        tokens = tokens + self.token_pos_enc(center_xyz)
        tokens = self.encoder(tokens)
        return self.compressor(tokens)

    def decode(self, query_xyz: torch.Tensor, z: torch.Tensor):
        return self.decoder(query_xyz, z)

    def forward(self, points: torch.Tensor, normals: torch.Tensor, query_xyz: torch.Tensor):
        z = self.encode(points, normals)
        udf, grad, conf = self.decode(query_xyz, z)
        return udf, grad, conf, z
