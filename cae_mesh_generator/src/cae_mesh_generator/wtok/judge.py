"""A learned judge of local wire structure, used to pick among draws.

The generator already produces good local structure sometimes and cannot tell
when. Measured over 30 val parts with 9 draws each:

    indicator     mean draw   medoid   best of 9
    is_arc            62%       63%       78%
    arc direction     69%       69%       87%
    turn error       13.5 deg  11.7 deg   7.9 deg

The medoid ranks by distance to the other draws, so it is blind to all of this --
it moves is_arc by one point. What is missing is a criterion, and hand-writing
one is what this project has repeatedly got wrong: a sagitta channel destroyed
40% of the arcs, a turn-angle channel decided for the model which relation
matters. So the criterion is learned.

The judge scores a LOCAL PATCH of the wire -- a slot and the edges either side --
as real or not. Its negatives are mostly STRUCTURAL PERTURBATIONS of the truth
(an arc flipped, a corner nudged, is_arc toggled, an edge split), not generator
output: a judge trained on one checkpoint's habits stops working on the next,
which is exactly how the consistency lambda failed (KB 05.17).

Nothing here names a corner, counts anything, or classifies a feature.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .frame import EDGE_SLOTS, FRAME_CH

K_SIDE = 3                      # edges either side of the slot being judged
PATCH_CH = 8                    # per neighbour: rel xyz 3, bulge 3, is_arc, live


def patch(x, k_side: int = K_SIDE):
    """(n_live, 2*k_side+1, PATCH_CH) -- every live slot with its neighbourhood.

    Normalised the way the spatial kernel normalises: centred on the slot, scaled
    by the patch's own mean edge length, and rotated into a frame built from the
    slot's own edge. That makes the judge blind to where the part is and how big
    it is, so it can only be judging structure.
    """
    live = np.flatnonzero(x[:, 7] < 0.5)
    n = len(live)
    if n < 2 * k_side + 1:
        return np.zeros((0, 2 * k_side + 1, PATCH_CH), np.float32)
    P = x[live, 0:3]
    B = x[live, 3:6]
    A = x[live, 6]
    d = np.roll(P, -1, 0) - P
    L = np.linalg.norm(d, axis=1)
    scale = max(float(L.mean()), 1e-9)

    out = np.zeros((n, 2 * k_side + 1, PATCH_CH), np.float32)
    for i in range(n):
        e1 = d[i] / max(L[i], 1e-9)
        # second axis from the patch's own spread, so the frame is determined by
        # the neighbourhood rather than by any global direction
        idx = [(i + o) % n for o in range(-k_side, k_side + 1)]
        w = P[idx] - P[i]
        r = w - np.outer(w @ e1, e1)
        nr = np.linalg.norm(r, axis=1)
        e2 = r[int(np.argmax(nr))] if nr.max() > 1e-9 else np.array([0.0, 0.0, 1.0])
        e2 = e2 - (e2 @ e1) * e1
        ne = np.linalg.norm(e2)
        e2 = e2 / ne if ne > 1e-9 else np.cross(e1, [1.0, 0.0, 0.0])
        Rm = np.stack([e1, e2, np.cross(e1, e2)])
        out[i, :, 0:3] = (w @ Rm.T) / scale
        out[i, :, 3:6] = (B[idx] @ Rm.T) / scale
        out[i, :, 6] = A[idx]
        out[i, :, 7] = 1.0
    return out


def perturb(x, rng, strength: float = 1.0):
    """One structural break, applied to a true frame. Returns (x, ok).

    These are the negatives. Each is a thing that makes a wire wrong locally
    while leaving it globally plausible -- which is precisely what the generator
    does and what Chamfer cannot see.
    """
    x = x.copy()
    live = np.flatnonzero(x[:, 7] < 0.5)
    if len(live) < 6:
        return x, False
    i = int(rng.choice(live))
    P = x[live, 0:3]
    L = float(np.median(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)))
    kind = rng.integers(4)
    if kind == 0:                                   # flip an arc
        if x[i, 6] < 0.5:
            return x, False
        x[i, 3:6] = -x[i, 3:6]
    elif kind == 1:                                 # toggle is_arc
        x[i, 6] = 0.0 if x[i, 6] > 0.5 else 1.0
        if x[i, 6] < 0.5:
            x[i, 3:6] = 0.0
        else:
            v = rng.normal(size=3)
            x[i, 3:6] = v / max(np.linalg.norm(v), 1e-9) * L * 0.08 * strength
    elif kind == 2:                                 # nudge a corner
        v = rng.normal(size=3)
        x[i, 0:3] += v / max(np.linalg.norm(v), 1e-9) * L * rng.uniform(
            0.10, 0.35) * strength
    else:                                           # scale a bulge
        if x[i, 6] < 0.5:
            return x, False
        x[i, 3:6] *= rng.uniform(2.0, 4.0) if rng.random() < 0.5 else rng.uniform(
            0.0, 0.3)
    return x, True


class Judge(nn.Module):
    """Scores one patch. Aggregated over a frame with softmin, so the weakest
    slot decides -- a single broken corner is what makes a wire unusable, and an
    average over 20 good slots would hide it."""

    def __init__(self, dim: int = 128, k_side: int = K_SIDE):
        super().__init__()
        self.k_side = k_side
        n = 2 * k_side + 1
        self.net = nn.Sequential(
            nn.Linear(n * PATCH_CH, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 1))

    def forward(self, p):                      # (B, n, PATCH_CH) -> (B,)
        return self.net(p.flatten(1)).squeeze(-1)

    def frame_score(self, x, tau: float = 4.0):
        """Softmin over the slots of one frame. Higher is more real."""
        p = patch(x, self.k_side)
        if not len(p):
            return -1e9
        with torch.no_grad():
            s = self(torch.from_numpy(p).float().to(
                next(self.parameters()).device))
        return float(-torch.logsumexp(-tau * s, 0) / tau)


def demo():
    """The patch must be blind to placement and scale, and perturbation must
    change the patch it targets."""
    rng = np.random.default_rng(0)
    x = np.zeros((EDGE_SLOTS, FRAME_CH))
    x[:, 7] = 1.0
    t = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    x[:10, 0:3] = np.stack([40 * np.cos(t), 25 * np.sin(t), np.zeros(10)], 1)
    x[:10, 7] = 0.0
    x[[2, 5, 8], 6] = 1.0
    x[[2, 5, 8], 3:6] = [[0, 2.0, 0], [1.5, 0, 0], [0, -2.0, 0]]
    a = patch(x)

    y = x.copy()
    y[:10, 0:3] = y[:10, 0:3] * 3.7 + np.array([120.0, -40.0, 15.0])
    y[:10, 3:6] *= 3.7
    b = patch(y)
    print(f"patch shape {a.shape}")
    print(f"  after scaling 3.7x and translating: max |diff| "
          f"{np.abs(a - b).max():.2e}  (must be ~0)")
    assert np.abs(a - b).max() < 1e-4, "the patch is not scale/placement blind"

    seen = set()
    for _ in range(200):
        z, ok = perturb(x, rng)
        if ok:
            c = patch(z)
            seen.add(int(np.abs(c - a).sum(axis=(1, 2)).argmax()))
    print(f"  perturbations touched {len(seen)} distinct slots")
    assert len(seen) >= 3
    j = Judge()
    print(f"  judge on a frame: {j.frame_score(x):+.3f}")
    print("ok")


if __name__ == "__main__":
    demo()


def train(out_dir="runs/judge1", n_parts=1000, epochs=40, dim=128,
          device="cuda", seed=0):
    """Train on true patches against perturbed ones.

    No generator output is used. A judge that learns one checkpoint's habits
    stops working on the next -- the same failure as a post-process tuned to a
    checkpoint (KB 05.17) -- so it is trained on what breaks a wire, and then
    TESTED on whether it can rank real generations.
    """
    import json
    import pathlib

    from .dataset_curve import load_curve_parts
    from .frame import frame_target
    from .meshgen import fastener_frame

    R = pathlib.Path(__file__).resolve().parents[4]
    out = R / out_dir
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in (R / "runs" / "mesh_synth" / "parts").glob("*.npz")}
    vn = set(json.loads((R / "runs" / "wtok_curve_synth_v1"
                         / "val_names_100.json").read_text()))
    allp = [p for p in load_curve_parts(R / "runs" / "wtok_synth")
            if p.name in have]
    tr = [p for p in allp if p.name not in vn][:n_parts]
    va = [p for p in allp if p.name in vn][:60]

    def frames(parts):
        out_ = []
        for p in parts:
            x = frame_target(p, fastener_frame(p))
            if x is not None:
                out_.append(x)
        return out_

    TR, VA = frames(tr), frames(va)
    print(f"judge: {len(TR)} train frames, {len(VA)} val frames", flush=True)

    m = Judge(dim).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    bce = nn.BCEWithLogitsLoss()

    def batch(pool, n=32):
        pos, neg = [], []
        for _ in range(n):
            x = pool[rng.integers(len(pool))]
            a = patch(x)
            if not len(a):
                continue
            pos.append(a[rng.integers(len(a))])
            for _ in range(6):
                y, ok = perturb(x, rng)
                if not ok:
                    continue
                b = patch(y)
                if len(b) != len(a):
                    continue
                # the slot the perturbation actually changed
                j = int(np.abs(b - a).sum(axis=(1, 2)).argmax())
                neg.append(b[j])
                break
        if not pos or not neg:
            return None
        p_ = torch.from_numpy(np.stack(pos + neg)).float().to(device)
        y_ = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]).to(device)
        return p_, y_

    def auc(pool, n=400):
        m.eval()
        s, y = [], []
        with torch.no_grad():
            for _ in range(n):
                b = batch(pool, 4)
                if b is None:
                    continue
                s.append(m(b[0]).cpu().numpy())
                y.append(b[1].cpu().numpy())
        m.train()
        s, y = np.concatenate(s), np.concatenate(y).astype(bool)
        o = np.argsort(s)
        r = np.empty(len(s))
        r[o] = np.arange(len(s))
        n1, n0 = y.sum(), (~y).sum()
        return (r[y].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)

    best = 0.0
    for ep in range(1, epochs + 1):
        tot = k = 0
        for _ in range(200):
            b = batch(TR)
            if b is None:
                continue
            loss = bce(m(b[0]), b[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            k += 1
        if ep % 5 == 0 or ep == epochs:
            a = auc(VA)
            print(f"epoch {ep}: loss {tot/max(k,1):.4f}  val AUC {a:.4f}"
                  + ("  *best*" if a > best else ""), flush=True)
            if a > best:
                best = a
                torch.save({"model": m.state_dict(), "dim": dim,
                            "k_side": K_SIDE, "auc": a, "epoch": ep},
                           out / "best.pt")
    print(f"best val AUC {best:.4f} -> {out/'best.pt'}")
    return best


def load(path="runs/judge1/best.pt", device="cuda"):
    import pathlib
    R = pathlib.Path(__file__).resolve().parents[4]
    ck = torch.load(R / path, map_location=device, weights_only=False)
    m = Judge(ck["dim"], ck["k_side"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck
