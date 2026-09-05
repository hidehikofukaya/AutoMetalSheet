# AutoMetalSheet CAE Midsurface AE Diagnostic Report for fable5

Date: 2026-07-02  
Authoring context: fable5 can access the local repository and run artifacts. This report is still readable by itself, but the local paths below are the canonical evidence trail and should be inspected directly when judging the model.

## 0. Local Inspection Index

Repository root:

- `C:\Users\hide2\IdeaBox\AutoMetalSheet`

Data handoff and source data context:

- `C:\Users\hide2\IdeaBox\fill_volume\HANDOFF.md`
- `C:\Users\hide2\IdeaBox\fill_volume\fill_mid_surf\A0072600002_AllCATPart\fill\`
- `C:\Users\hide2\IdeaBox\fill_volume\fill_mid_surf\A0072601285_AllCATPart\fill\`

Architecture and progress documents:

- `C:\Users\hide2\IdeaBox\AutoMetalSheet\docs\cae_adaptive_shell_mesh_model.md`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\docs\cae_structured_scaffold_autoencoder_architecture.md`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\docs\cae_typed_cross_attention_lattice_design.md`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\docs\cae_tangent_frame_decoder_design.md`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\docs\cae_structured_q20_60_cov_scaf_seed13_report.md`

Core implementation files:

- `C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src\cae_mesh_generator\data\fill_volume_dataset.py`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src\cae_mesh_generator\model\hierarchical_ae.py`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src\cae_mesh_generator\train_autoencoder.py`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src\cae_mesh_generator\evaluate_autoencoder.py`
- `C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\tests\`

Most important run artifacts to inspect first:

| purpose | local path | useful files |
|---|---|---|
| current strongest baseline, mirror-y augmentation | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13` | `history.json`, `best.pt`, `best_by_val_loss.pt`, `train_stdout.log` |
| current strongest visual/metric evaluation | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_best` | `index.html`, `aggregate_metrics.json`, `metrics.json`, `metrics.csv` |
| mirror-y alternate checkpoints | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_val_loss` and sibling eval dirs | compare checkpoint-selection sensitivity |
| negative tangent-frame decoder result | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_tangent_decoder_seed13` | `history.json`, checkpoints |
| tangent-frame evaluation | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_tangent_decoder_seed13_eval_all24_best` | `index.html`, `aggregate_metrics.json`, `metrics.json` |
| negative typed-lattice result | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13` | `history.json`, checkpoints |
| typed-lattice evaluation | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13_eval_all24_best` | `index.html`, `aggregate_metrics.json`, `metrics.json` |
| topology-boundary run used as an important comparison | `C:\Users\hide2\IdeaBox\AutoMetalSheet\runs\cae_mesh_structured_q20_60_n24_boundary_seed13_topology` | `history.json`, checkpoints, preview PLYs |

Suggested local checks for fable5:

```powershell
cd C:\Users\hide2\IdeaBox\AutoMetalSheet
Get-Content .\runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_best\aggregate_metrics.json
Get-Content .\runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13\train_stdout.log
python -m pytest .\cae_mesh_generator\tests
```

For visual diagnosis, open the relevant `index.html` files in the evaluation directories and compare target/reconstruction/error views, especially validation part `A0072600002_AllCATPart:081`.

## 1. What We Are Building

AutoMetalSheet is shifting from CAD/B-Rep or CATIA feature generation toward CAE-ready sheet-metal shell mesh generation.

Current MVP direction:

- Input: filled midsurface STEP files of sheet-metal parts.
- Current learning probe: midsurface point-cloud autoencoder.
- Future output target: analysis-ready midsurface shell mesh, preferably quad-dominant, with node/element sets for fasteners, mounts, loads, contacts, boundaries, and refinement regions.
- Current output in experiments: unordered reconstructed midsurface point cloud, not yet a CAE mesh.
- Important constraint: B-Rep direct generation is not the MVP because precision/topology robustness is risky.

The current AE is not intended to be the final generator. It is a diagnostic probe to see whether the network can understand and reconstruct sheet-metal midsurface structure.

## 2. Sheet-Metal-Specific Assumptions

Why this domain may benefit from specialized representation:

- Parts are thin midsurfaces, not volumetric solids.
- Shape is mostly one sheet-like surface with bends, flanges, ribs, emboss-like features, and open boundaries.
- Boundary placement matters for CAE.
- Fastener/mount regions and other constraints matter, but the current AE mostly uses shape-only input plus optional boundary indicators.
- In a bounding box, actual material surface occupies a very small fraction of space, so dense volumetric generation is likely inefficient.

## 3. Data Currently Used

Immediately usable assemblies:

| assembly | filled midsurface STEP count | notes |
|---|---:|---|
| `A0072600002_AllCATPart` | 43 | has filled midsurfaces and joints |
| `A0072601285_AllCATPart` | 32 | has filled midsurfaces and joints |

Additional assemblies exist but are not yet in the current training set because mid/fill extraction or annotations are incomplete:

- `A0072600081_AllCATPart`
- `A0072600367_AllCATPart`
- `A0072600529_AllCATPart`

For the recent formal comparisons:

- source assemblies: the 2 usable assemblies above
- max STEP file size: 5 MB
- candidate parts after this filter: 65
- size band: bbox diagonal quantile q20-60
- selected parts: 24
- split: random seed 13
- train base parts: 18
- validation parts: 6
- evaluation: full selected 24 parts are evaluated, but key comparison is the 6 validation parts

Validation part IDs in this split:

- `A0072600002_AllCATPart:026`
- `A0072600002_AllCATPart:077`
- `A0072600002_AllCATPart:024`
- `A0072600002_AllCATPart:081`
- `A0072601285_AllCATPart:10`
- `A0072600002_AllCATPart:074`

Recurring difficult validation part:

- `A0072600002_AllCATPart:081`
- It is a thin strip-like midsurface.
- Many model variants made false-positive sheet-like point clouds around it or missed its narrow structure.

## 4. Current Model Architecture

The best current baseline is a structured scaffold autoencoder with a free local refinement decoder and train-only mirror-y augmentation.

### 4.1 Input Features

Each sampled point has:

- normalized xyz
- normal vector
- joint-distance feature, usually constant when joint distance is disabled
- optional boundary indicator

Current formal runs use:

- boundary feature: enabled
- boundary sampling fraction: 0.25
- joint distance: disabled

### 4.2 Encoder

There are three possible streams:

1. Coarse/global stream
   - Uses farthest point sampling over the input.
   - Encodes whole-part context using transformer-style processing.
   - Intended meaning: part-scale structure and large footprint.

2. Local/fine stream
   - Uses FPS centers and k-nearest neighbors.
   - Encodes local patches using shared local MLP/PointNet-like pooling.
   - Intended meaning: local midsurface evidence, curvature, boundary-adjacent detail.

3. Boundary stream
   - Enabled when boundary feature is present.
   - Boundary feature is extracted from topology/open-edge sampling.
   - Intended meaning: where the generated sheet should stop or preserve open-edge details.

### 4.3 Latent Fusion

The global, local, and boundary tokens are fused into latent memory.

A Typed Cross-Attention Lattice was tested to let these streams repeatedly cross-attend before decoding, but it did not improve validation metrics in the current point-cloud decoder setting.

### 4.4 Decoder

Current stronger decoder path:

1. ScaffoldDecoder
   - Uses learned scaffold queries.
   - Queries attend to latent/global/boundary memory.
   - Outputs coarse scaffold points.

2. LocalRefinementDecoder
   - For each scaffold point, finds nearby fine tokens.
   - Uses per-scaffold point queries and attention to local tokens.
   - Outputs local refined point offsets, normals, and refinement logits.
   - In the strongest current result, offsets are still free 3D offsets.

This means the output is still an unordered point cloud; it does not yet represent mesh connectivity, boundary chains, or CAE element topology.

## 5. Losses and Metrics

Training losses used across recent formal runs:

- Chamfer reconstruction loss
- normal consistency loss
- spread loss
- target coverage loss: penalize target points farther than a threshold
- scaffold Chamfer loss
- boundary coverage loss
- typed scaffold target losses:
  - boundary scaffold
  - crease scaffold
  - corner scaffold
- optional prediction-surface loss: penalize generated points too far from target
- optional refinement occupancy BCE loss for tangent experiments

Evaluation metrics are world-mm nearest-distance metrics:

- Chamfer mean mm = mean recon-to-target + mean target-to-recon
- recon p95 mm = p95 of generated-point distance to target
- target p95 mm = p95 of target-point miss distance to reconstruction
- target within 5 mm
- boundary p95 mm
- boundary within 5 mm
- worst Chamfer part

The most important diagnostic separation:

- recon-to-target errors reveal false-positive generated sheet area.
- target-to-recon errors reveal missed coverage.

## 6. Experiment Timeline and Results

All main rows below use the same q20-60 n24 seed13 split unless stated otherwise. Values are validation split aggregates.

### 6.1 Structured baseline and coverage/scaffold loss

Early q20-60 structured run:

- 24 selected parts, train/val split
- Best validation around epoch 110
- Val Chamfer mean about 27.216 mm
- Val target p95 about 35.581 mm
- Epoch 500 worsened validation despite better train metrics

Coverage/scaffold loss improved:

- Val Chamfer mean improved to about 19.565 mm
- Val target p95 improved to about 24.237 mm
- Target within 5 mm improved to about 27.4%
- But last epoch overfit, so early stopping is required

### 6.2 Boundary-aware sampling and metrics

Topology-boundary run:

- Used topology-derived open boundary sampling and boundary coverage loss.
- Improved boundary metrics.
- Best practical checkpoint around epoch 200.

Validation metrics:

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| topology-boundary boundary-best | 23.448 | 30.706 | 19.334 | 0.201 | 20.711 | 0.270 | `081` | 35.969 |

Interpretation:

- Boundary sampling helped boundary-aware reconstruction.
- It did not fully solve thin-strip worst-case behavior.

### 6.3 Boundary token and composite CAE score

Boundary feature/token run:

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| boundary-token CAE best | 24.255 | 33.658 | 18.506 | 0.197 | 23.248 | 0.215 | `081` | 32.532 |

Interpretation:

- Target p95 improved.
- Boundary p95 worsened versus topology-boundary best.
- Single pooled boundary token is inconclusive.

### 6.4 Feature scaffold supervision

Merged feature scaffold supervision:

- Used fixed scaffold targets mixed from boundary, crease, corner, and area samples.
- Improved Chamfer in one checkpoint but hurt boundary p95.

Split feature scaffold supervision:

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| split-scaffold CAE best | 25.134 | 36.340 | 18.520 | 0.192 | 20.486 | 0.143 | `081` | 34.785 |

Interpretation:

- Target p95 and boundary p95 were competitive.
- Recon p95 and local density were worse.
- Visual inspection showed false-positive generated points around a thin target strip.

### 6.5 Prediction-surface loss

Prediction-surface loss was added to suppress false-positive generated points.

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| prediction-surface CAE best | 28.036 | 38.550 | 21.175 | 0.157 | 18.449 | 0.181 | `081` | 37.739 |

Interpretation:

- Improved boundary p95.
- Worsened Chamfer and target coverage.
- Scalar point-cloud losses appear to be near a ceiling.

### 6.6 Typed Cross-Attention Lattice

This added semantic cross-attention among global/local/boundary streams.

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| lattice val-loss best | 28.027 | 38.401 | 22.924 | 0.158 | 28.707 | 0.151 | `081` | 39.745 |

Interpretation:

- More semantic interaction in the encoder/fusion did not improve the current point-cloud AE.
- Deeper attention alone is lower priority than output representation and data diversity.

### 6.7 Tangent-frame decoder

This changed local decoding from free 3D offsets to scaffold-local tangent-frame patches, with patch scales, normals, and occupancy logits.

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| tangent CAE best | 44.288 | 61.379 | 31.371 | 0.075 | 36.983 | 0.061 | `081` | 79.294 |

Interpretation:

- Strong negative result.
- Train memorization was good by epoch 300, but validation collapsed.
- The failure suggests scaffold placement is poorly anchored; if scaffold points are wrong, local tangent patch constraints make the wrong patch more coherent rather than more correct.

### 6.8 Train-only mirror-y augmentation

This is the most recent and most encouraging result.

Design:

- Apply mirror augmentation only after part-level split.
- Train gets original + mirrored samples.
- Validation remains original only.
- Therefore no mirrored validation leakage into training.
- Mirrored consistently:
  - points
  - normals
  - scaffold targets
  - boundary targets
  - crease targets
  - corner targets

Formal run:

- base train parts: 18
- train samples after mirror-y: 36
- validation parts: 6, unaugmented
- architecture: structured free decoder, lattice off, tangent off
- losses: topology-boundary and typed scaffold losses enabled

Validation results:

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| mirror-y CAE best | 23.010 | 29.189 | 19.851 | 0.194 | 21.778 | 0.160 | `081` | 28.594 |
| mirror-y val-loss best | 23.876 | 30.256 | 21.062 | 0.189 | 22.605 | 0.169 | `081` | 32.257 |
| mirror-y last | 25.487 | 30.093 | 25.344 | 0.152 | 30.418 | 0.113 | `A0072601285:10` | 30.653 |

Comparison to prior stronger baselines:

| checkpoint | Chamfer mean mm | recon p95 mm | target p95 mm | target within 5mm | boundary p95 mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| mirror-y CAE best | 23.010 | 29.189 | 19.851 | 0.194 | 21.778 | 0.160 | `081` | 28.594 |
| topology-boundary boundary-best | 23.448 | 30.706 | 19.334 | 0.201 | 20.711 | 0.270 | `081` | 35.969 |
| boundary-token CAE best | 24.255 | 33.658 | 18.506 | 0.197 | 23.248 | 0.215 | `081` | 32.532 |
| split-scaffold CAE best | 25.134 | 36.340 | 18.520 | 0.192 | 20.486 | 0.143 | `081` | 34.785 |
| prediction-surface CAE best | 28.036 | 38.550 | 21.175 | 0.157 | 18.449 | 0.181 | `081` | 37.739 |
| tangent CAE best | 44.288 | 61.379 | 31.371 | 0.075 | 36.983 | 0.061 | `081` | 79.294 |

Interpretation:

- Data diversity was a real bottleneck.
- Mirror-y improved validation Chamfer slightly over topology-boundary best.
- Mirror-y improved recon p95.
- Mirror-y dramatically improved the recurring worst part `081`.
- Boundary p95 and boundary-within did not beat topology-boundary best.
- Last epoch still overfits; early stopping remains important.

## 7. Current Best Diagnosis

### 7.1 What seems true

1. The dataset is too small for generalization.
   - 18 base training parts is not enough.
   - Mirroring doubled train samples and immediately improved worst-case validation behavior.

2. Pure model complexity has not helped.
   - Cross-attention lattice did not help.
   - Tangent-frame local decoding was worse.
   - Scalar losses move tradeoffs but do not solve topology/placement.

3. Scaffold placement is a key bottleneck.
   - Learned scaffold queries can memorize training geometry.
   - On held-out thin strip parts, the scaffold can be placed incorrectly, causing local decoders to emit coherent but wrong sheet-like point clouds.

4. Boundary information matters, but current boundary treatment is incomplete.
   - Topology-boundary sampling improved boundary metrics.
   - Mirror-y improved Chamfer and worst cases but did not beat boundary-within metrics.

5. The current unordered point-cloud representation is near its diagnostic ceiling.
   - It cannot represent connectivity, boundary chains, or solver-ready shell elements.
   - It cannot fail closed in the way a CAE mesh generator must.

### 7.2 What remains uncertain

1. Whether mirror-y is physically valid for all parts.
   - Axis must match the true vehicle/CATIA coordinate convention.
   - If y is the left-right vehicle axis, mirror-y is likely meaningful.
   - If not, x or z may be better or harmful.

2. Whether mirror augmentation helps because of symmetry or because it regularizes coordinate bias.

3. Whether current validation split is representative.
   - It uses only 6 validation parts.
   - It is one random seed.
   - It is two assemblies only.

4. Whether training on all 65 size-filtered candidates or using additional assemblies would reduce the architecture sensitivity.

## 8. Recommended Next Experiments

### 8.1 Highest priority: augmentation axis comparison

Keep validation unaugmented and compare:

- no mirror
- mirror-x
- mirror-y
- mirror-z
- mirror-x+y
- possibly mirror-x+y+z only if physically meaningful

Use same q20-60 n24 seed13 split first, then repeat over at least 3 split seeds.

Primary question:

- Is mirror-y genuinely best, or was it an axis-specific accident?

### 8.2 Larger data selection

Run larger part counts if compute allows:

- q20-60, max_parts 32
- q20-60, max_parts 48
- all q20-60 available under 5 MB

Important:

- preserve part-level validation split before augmentation
- ensure no mirrored sibling leakage into validation

### 8.3 Stratified splits

Do not split only by random seed. Stratify by:

- bbox diagonal
- aspect ratio / slenderness
- boundary length or boundary sample count
- crease count / curvature proxy
- assembly ID

Goal:

- avoid placing all thin strip-like parts in validation or train by accident
- measure whether `081` is an isolated family or a systematic failure mode

### 8.4 Scaffold anchoring experiment

Instead of fully learned free scaffold queries, propose scaffold points from encoded geometry:

- use fine token centers or coarse FPS centers as scaffold anchors
- predict residual offsets from those anchors
- optionally add a small set of learned global scaffold proposals

Hypothesis:

- anchoring scaffold candidates to observed local evidence will reduce false-positive sheet patches and improve held-out thin strips.

### 8.5 Move toward CAE Mesh IR

Point-cloud AE is useful for diagnostics, but final direction should represent:

- mesh nodes
- element connectivity
- boundary chains
- feature/constraint sets
- local patch topology
- protected regions around fasteners/mounts/loads

Short-term bridge:

- use scaffold nodes as candidate graph nodes
- connect local neighborhoods
- use boundary targets as chain supervision
- let occupancy/refinement logits suppress inactive local neighborhoods during export

## 9. Specific Questions for fable5

Please diagnose the current direction with the following questions in mind:

1. Does the mirror-y result indicate true data diversity limitation, or could it be masking another issue such as coordinate-frame bias?

2. Is train-only mirroring a valid augmentation for sheet-metal midsurfaces in this setting?
   - What checks should be added to avoid unphysical mirrored examples?

3. Given the negative tangent-frame decoder result, should the next architecture focus on scaffold anchoring, topology/graph decoding, or more data augmentation first?

4. Are the current metrics sufficient?
   - Chamfer
   - recon p95
   - target p95
   - target within 5 mm
   - boundary p95
   - boundary within 5 mm

5. Should boundary metrics be weighted higher in checkpoint selection, given the CAE mesh goal?

6. What would be a robust split strategy for this small dataset?
   - random split
   - assembly holdout
   - family/shape-cluster holdout
   - stratified by slenderness/boundary/crease

7. Does the current structured scaffold AE still make sense, or should the project switch sooner to explicit CAE Mesh IR generation?

8. If staying with AE reconstruction for one more phase, what minimal next experiment has the best chance of revealing whether scaffold anchoring is worthwhile?

## 10. Proposed Immediate Plan

Current recommended order:

1. Run mirror-axis ablation on the same q20-60 n24 seed13 split.
2. Repeat the best mirror axis across at least 3 split seeds.
3. Increase part count beyond 24 if possible.
4. Implement scaffold anchoring from fine/coarse encoder centers.
5. Evaluate active occupancy gating only after scaffold anchoring.
6. Begin CAE Mesh IR topology representation once scaffold positions stabilize.

## 11. One-Sentence Summary

The strongest evidence so far is that limited data diversity, not transformer capacity, is the immediate bottleneck: train-only mirror-y doubled effective train samples and improved the recurring worst validation part substantially, while added cross-attention and tangent-frame decoding did not generalize.
