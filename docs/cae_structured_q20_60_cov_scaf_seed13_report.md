# Structured Scaffold AE q20-60 Coverage/Scaffold Loss Report

この文書は `docs/cae_structured_q20_60_e1000_diagnostic_report.md` の続編である。前回の q20-60 n24 seed13 run では、structured scaffold decoder は学習するものの validation coverage が低く、epoch を伸ばしても改善しないことが分かった。

今回の目的は、同一データ、同一 split、同一モデル容量で `target_coverage_loss` と `scaffold_chamfer_loss` を追加し、未知 validation 部品の coverage が改善するかを確認することである。

## Run

対象run:

- `runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13`
- baseline: `runs/cae_mesh_structured_q20_60_n24_e1000_seed13`
- bbox diagonal q20-60
- `max_file_mb=5.0`
- `max_parts=24`
- train / val = 18 / 6
- `split_strategy=random`
- `split_seed=13`
- `n_points=512`
- `token_dim=128`
- `n_coarse=128`
- `n_patches=64`
- `n_latents=64`
- `n_scaffold=128`
- `points_per_scaffold=4`
- `n_local_tokens=8`
- `device=cuda`

追加した学習条件:

```powershell
--lambda-target-coverage 0.5 `
--coverage-threshold-mm 5.0 `
--lambda-scaffold 0.2 `
--batch-size 2 `
--preload-data `
--pin-memory
```

`batch-size` は baseline の1から2に変えている。データ選択、split、モデル容量は同一だが、厳密に optimizer dynamics だけを揃える場合は batch size 1 の追試も候補である。

## Training

300 epoch 完走した。validation best は epoch 60 だった。

主な履歴:

| epoch | train loss | val loss | train chamfer | val chamfer | val coverage loss | val scaffold loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.731376 | 0.551857 | 0.473734 | 0.384707 | 0.128359 | 0.345684 |
| 20 | 0.240634 | 0.202131 | 0.153203 | 0.116931 | 0.026540 | 0.263826 |
| 60 | 0.143616 | 0.157421 | 0.088275 | 0.088571 | 0.022182 | 0.230882 |
| 120 | 0.113639 | 0.169067 | 0.074364 | 0.099392 | 0.020440 | 0.241877 |
| 220 | 0.100198 | 0.188795 | 0.066584 | 0.116324 | 0.022057 | 0.256803 |
| 300 | 0.110971 | 0.207805 | 0.075981 | 0.129533 | 0.031681 | 0.253422 |

読み取り:

- best は epoch 60 で、前回 baseline の best epoch 110 より早い。
- epoch 60 以降は train 側が改善しても validation は安定して改善しない。
- 追加 loss を入れても過学習は残る。early stopping は必須である。

## Evaluation

評価出力:

- best: `runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13/visual_eval_best_epoch60`
- last: `runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13/visual_eval_last_epoch300`
- 比較診断: `runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13/diagnostics`

集計比較:

| model | val Chamfer mean | val recon p95 mean | val target p95 mean | val target within 5mm | val recon within 5mm |
|---|---:|---:|---:|---:|---:|
| baseline best epoch110 | 27.216mm | 31.178mm | 35.581mm | 16.0% | 18.4% |
| baseline last epoch500 | 30.262mm | 35.616mm | 33.331mm | 12.9% | 13.2% |
| coverage/scaffold best epoch60 | 19.565mm | 20.614mm | 24.237mm | 27.4% | 25.6% |
| coverage/scaffold last epoch300 | 28.151mm | 35.909mm | 28.492mm | 13.3% | 11.6% |

best 同士の改善量:

- val Chamfer mean: 27.216mm -> 19.565mm, -7.652mm
- val target p95 mean: 35.581mm -> 24.237mm, -11.343mm
- val target within 5mm: 16.0% -> 27.4%, +11.4 percentage points
- worst Chamfer part `A0072600002_AllCATPart:081`: 37.402mm -> 24.358mm
- worst target p95: 41.553mm -> 35.060mm

![aggregate comparison](../runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13/diagnostics/aggregate_comparison.png)

![validation part comparison](../runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13/diagnostics/val_per_part_comparison.png)

## Worst Part

worst Chamfer part は引き続き `A0072600002_AllCATPart:081` である。

| model | Chamfer | recon p95 | target p95 | target within 5mm | recon within 5mm |
|---|---:|---:|---:|---:|---:|
| baseline best epoch110 | 37.402mm | 47.153mm | 41.553mm | 11.5% | 9.4% |
| coverage/scaffold best epoch60 | 24.358mm | 30.982mm | 24.415mm | 20.3% | 12.5% |
| coverage/scaffold last epoch300 | 38.954mm | 58.204mm | 29.368mm | 13.3% | 5.5% |

![worst 081 visual comparison](../runs/cae_mesh_structured_q20_60_n24_cov_scaf_seed13/diagnostics/worst_081_visual_comparison.png)

可視化上、coverage/scaffold best は baseline より target に近い領域へ復元点を広げている。一方で、まだ外れ点、端部miss、境界の曖昧さが残る。つまり、今回の改善は「面全体の存在範囲を覆う能力」を上げたが、CAE mesh として必要な境界線、稜線、接続、要素品質はまだ保証していない。

## Interpretation

coverage loss と scaffold supervision は有効である。同一 split では、未知 validation 部品に対する Chamfer、target p95、5mm coverage を同時に改善した。

ただし、これは最終的な CAE mesh generator として十分という意味ではない。現状の出力はまだ点群であり、以下が未解決である。

- 境界線と端部の明示的な保持
- scaffold node 間の edge/face topology
- local refinement が作る点の接続品質
- normal consistency と warpage/aspect ratio などの mesh quality
- solver import smoke test
- refinement logits の教師と評価

この結果から、次の判断は以下になる。

1. validation checkpoint selection を composite loss だけでなく、`target_within_5mm` または `target_p95` でも選べるようにする。
2. boundary-aware sampling と boundary-aware metric を追加する。
3. FPS pseudo scaffold に加えて、boundary-aware scaffold または skeleton scaffold を比較する。
4. `refinement_logits` に局所誤差、境界近傍、joint近傍、曲率proxyの教師を入れる。
5. CAE Mesh IR の node/edge/face 候補へ進み、点群復元から接続を持つ shell mesh 生成へ移る。

## Adoption

coverage/scaffold loss は採用する。今後の structured AE run では、少なくとも以下を標準候補にする。

```powershell
--lambda-target-coverage 0.5 `
--coverage-threshold-mm 5.0 `
--lambda-scaffold 0.2
```

ただし、`best.pt` の選択基準は今後見直す。今回の `best.pt` は composite `val_loss` で選ばれており、CAE用途で最も重要な `target_within_5mm` や `target_p95` の最良点と一致するとは限らない。

## Implementation Update: CAE-Oriented Best Checkpoint

このレポートの判断を受けて、学習中のbest checkpoint選択をCAE寄りの評価指標でも行えるようにした。

追加した学習メトリクス:

- `recon_mean_mm`
- `recon_p95_mm`
- `target_mean_mm`
- `target_p95_mm`
- `target_within_threshold`

`target_within_threshold` は既存の `--coverage-threshold-mm` を使う。たとえば `--coverage-threshold-mm 5.0` なら、学習ログ上は `val_tgt5` として表示される。

追加したbest metric候補:

```powershell
--best-metric val_target_within_threshold
--best-metric val_target_p95_mm
```

`val_target_within_threshold` は大きいほど良いので最大化する。`val_target_p95_mm` は小さいほど良いので最小化する。checkpointの `best` metadata には、`metric`, `mode`, `value`, `epoch` を保存する。

短いsmokeでは以下を確認した。

```text
unit tests: 15 passed
best metric smoke: val_target_within_threshold, success
best metric smoke: val_target_p95_mm, success
```

次の本runでは、前回と同じ q20-60 n24 seed13 条件に対して、以下のどちらかを使って比較する。

```powershell
--best-metric val_target_p95_mm
```

または、5mm coverageを最優先する場合:

```powershell
--best-metric val_target_within_threshold
```

## Follow-up Run: Target P95 Checkpoint Selection

`--best-metric val_target_p95_mm` を使って、同じ q20-60 n24 seed13 条件を再学習した。

対象run:

- `runs/cae_mesh_structured_q20_60_n24_cov_scaf_tgtp95_seed13`
- best checkpoint: epoch 110
- best metadata: `metric=val_target_p95_mm`, `mode=min`, `value=22.4825`

評価出力:

- best: `runs/cae_mesh_structured_q20_60_n24_cov_scaf_tgtp95_seed13/visual_eval_best_epoch110`
- last: `runs/cae_mesh_structured_q20_60_n24_cov_scaf_tgtp95_seed13/visual_eval_last_epoch300`
- 比較図: `runs/cae_mesh_structured_q20_60_n24_cov_scaf_tgtp95_seed13/diagnostics/aggregate_target_p95_selection_comparison.png`

比較:

| model | val Chamfer mean | val target p95 mean | val target within 5mm |
|---|---:|---:|---:|
| baseline best epoch110 | 27.216mm | 35.581mm | 16.0% |
| coverage/scaffold val-loss best epoch60 | 19.565mm | 24.237mm | 27.4% |
| coverage/scaffold target-p95 best epoch110 | 25.219mm | 21.424mm | 16.9% |
| coverage/scaffold target-p95 last epoch300 | 29.097mm | 30.876mm | 16.8% |

![target p95 selection comparison](../runs/cae_mesh_structured_q20_60_n24_cov_scaf_tgtp95_seed13/diagnostics/aggregate_target_p95_selection_comparison.png)

読み取り:

- `val_target_p95_mm` 単独選択は、確かに target p95 mean を 24.237mm から 21.424mm へ改善した。
- しかし Chamfer mean は 19.565mm から 25.219mm へ悪化した。
- target within 5mm は 27.4% から 16.9% へ悪化した。
- worst Chamfer part `081` は 24.358mm から 39.071mm へ悪化した。

したがって、`val_target_p95_mm` 単独を標準best指標にするべきではない。p95は端部missの長い裾を抑えるには有効だが、点群全体のcoverageと外れ点を同時に保てない。

更新後の判断:

1. `val_loss` best は現時点では最もバランスがよい。
2. `val_target_p95_mm` best は補助snapshotとして保存・比較する価値がある。
3. 次に必要なのは単独指標の置き換えではなく、multi-best checkpoint保存、または `Chamfer + target p95 + coverage penalty` の複合選択である。
4. 本質的には、点群メトリクスだけで境界・topologyを保証するのは限界があるため、boundary-aware metric と CAE Mesh IR へ進む優先度は変わらない。

## Implementation Update: Multi-Best Checkpoints

target p95単独選択の追試結果を受けて、1回の学習で複数のbest snapshotを保存できるようにした。

追加CLI:

```powershell
--extra-best-metrics val_target_within_threshold val_target_p95_mm
```

この指定により、主bestである `best.pt` に加えて以下が保存される。

```text
best_by_val_target_within_threshold.pt
best_by_val_target_p95_mm.pt
```

各checkpointの `best` metadataには、そのsnapshotを選んだmetric、mode、value、epochが入る。`best.pt` には `extra_best` metadataも入るため、どの補助snapshotがどのepochから来たかを後で確認できる。

smoke結果:

```text
unit tests: 16 passed
multi-best smoke: success
```

次の正式runでは、主指標を `val_loss` のままにして、以下を追加するのが推奨である。

```powershell
--extra-best-metrics val_target_within_threshold val_target_p95_mm
```

これにより、バランスのよい `val_loss` best、coverage最優先best、target p95最優先bestを同一学習軌跡から比較できる。

## Implementation Update: Boundary-Aware Sampling and Metrics

CAE shell mesh用途では、面全体のChamferやtarget p95だけでなく、境界、端部、細長い段付き構造のmissが重要である。このため、追加手作業なしでSTEPテセレーションからboundaryを自動抽出し、学習・評価へ流す実装を追加した。

### Boundary extraction

OpenCASCADEのface triangulationはCAD faceごとに頂点を複製することがある。そのため、三角形indexだけで1回出現edgeを数えると、CAD face間の継ぎ目までboundaryとして誤検出する。

今回の実装では以下の手順にした。

1. tessellated verticesを座標許容差でweldする。
2. welded vertex id上で三角形edgeをundirected edgeとして数える。
3. 出現回数が1回のedgeだけをopen boundary edgeとする。
4. boundary edge長に比例してboundary pointsをsampleする。

追加実装:

- `boundary_edges_from_faces`
- `weld_vertices`
- `sample_boundary_points`
- `TessellatedMesh.boundary_edges`
- `TessellatedMesh.boundary_edge_normals`

### Dataset

既定では既存挙動を壊さない。

```powershell
--boundary-sample-fraction 0.0
```

有効化すると、target点群の一部をboundary edge上の点に置き換える。

```powershell
--boundary-sample-fraction 0.25
```

DataLoaderで可変長tensorを扱わないため、datasetは固定長 `boundary_mask` を返す。boundary点は `points[boundary_mask]` として取得する。

### Training metrics and loss

追加metric:

- `boundary_coverage`
- `boundary_within_threshold`
- `boundary_p95_mm`
- `boundary_point_count`

追加loss:

```powershell
--lambda-boundary-coverage
--boundary-threshold-mm
```

boundary coverage lossは、boundary target pointから復元点への最近傍距離が指定mm閾値を超えた分を罰則にする。既定重みは0.0なので、既存runとは互換である。

### Evaluation

評価CLIにもboundary overrideを追加した。

```powershell
--boundary-sample-fraction 0.25
```

これにより、古いcheckpointでもboundary-aware評価を後付けできる。

出力に以下が追加される。

- `boundary_points.ply`
- `boundary_miss_error.ply`
- `boundary_p95_mm`
- `boundary_within_5mm`
- aggregateの `boundary_p95_mean_mm`
- aggregateの `boundary_within_5mm_mean`
- aggregateの `worst_boundary_p95_part`

### Smoke

```text
unit tests: 18 passed
boundary-aware train smoke: success
boundary-aware eval smoke: success
old checkpoint boundary eval smoke: success
```

### Review Follow-up

A verification subagent found one important risk before the formal run:

- Triangle single-use boundary extraction can misclassify internal CAD seams as open boundaries when two adjacent faces have non-conformal tessellation, such as a T-junction along the same geometric edge.
- Boundary cache keys did not include the boundary extraction algorithm version or tolerances.
- Boundary metrics were closer to per-part averages than sampled-boundary-point averages.

The implementation was revised before the next experiment:

- `tessellate_step` now prefers OCC B-Rep topology open edges from Edge-to-Face adjacency. Coordinate-welded triangle single-use edges remain only as a fallback.
- Topological boundary curves are split into bounded linear segments for boundary point sampling.
- Boundary segment normals are assigned from nearest tessellated face normals.
- The tessellation cache key now includes `BOUNDARY_EXTRACTION_VERSION`, weld tolerance, and boundary segmentation parameters.
- Boundary loss and boundary metrics are point-weighted over sampled boundary targets.
- `--best-metric` / `--extra-best-metrics` boundary choices emit a warning when `--boundary-sample-fraction 0.0` makes them constant.

Updated verification:

```text
unit tests: 19 passed
topology-boundary train smoke: success
topology-boundary eval smoke: success
smoke output: runs/cae_mesh_boundary_topology_smoke
```

smoke commandの代表:

```powershell
--boundary-sample-fraction 0.25 `
--lambda-boundary-coverage 0.2 `
--boundary-threshold-mm 5.0 `
--extra-best-metrics val_boundary_p95_mm val_boundary_within_threshold
```

### Next experiment

次の正式runでは、coverage/scaffold lossに加えてboundary-aware sampling/lossを有効化する。

推奨初期設定:

```powershell
--boundary-sample-fraction 0.25 `
--lambda-boundary-coverage 0.2 `
--boundary-threshold-mm 5.0 `
--extra-best-metrics val_target_within_threshold val_target_p95_mm val_boundary_p95_mm val_boundary_within_threshold
```

採否は、従来のval Chamfer / target p95 / target within 5mmに加えて、`boundary_p95_mean_mm` と `boundary_within_5mm_mean` で判断する。

## Formal Run: Topology Boundary Seed13

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_boundary_seed13_topology
```

Configuration:

- model: structured scaffold AE
- data: q20-60 bbox diagonal, `max_parts=24`, random split seed 13
- split: 18 train / 6 val
- epochs: 300
- device: CUDA, NVIDIA GeForce GTX 1660 SUPER
- losses: coverage/scaffold plus topology-boundary coverage
- boundary target fraction: 0.25
- primary checkpoint: `--best-metric val_loss`
- extra checkpoints: `val_target_within_threshold`, `val_target_p95_mm`, `val_boundary_p95_mm`, `val_boundary_within_threshold`

Training completed normally. The final epoch overfit relative to the best validation checkpoints.

### Checkpoint Comparison

All rows use the same val split and the same topology-boundary evaluation override. Lower is better for Chamfer/p95; higher is better for within-5mm metrics.

| checkpoint | epoch | Chamfer mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| old coverage/scaffold best | 60 | 21.109 | 26.888 | 0.228 | 31.480 | 0.165 |
| topology boundary val_loss | 190 | 22.149 | 19.615 | 0.196 | 22.125 | 0.227 |
| topology boundary target_within | 50 | 23.811 | 20.776 | 0.208 | 23.722 | 0.233 |
| topology boundary target_p95 | 200 | 23.448 | 19.334 | 0.201 | 20.711 | 0.270 |
| topology boundary boundary_p95 | 90 | 28.517 | 20.649 | 0.120 | 23.674 | 0.125 |
| topology boundary boundary_within | 200 | 23.448 | 19.334 | 0.201 | 20.711 | 0.270 |
| topology boundary last | 300 | 28.868 | 25.806 | 0.130 | 33.257 | 0.150 |

Artifacts:

- comparison CSV: `runs/cae_mesh_structured_q20_60_n24_boundary_seed13_topology/diagnostics/checkpoint_comparison.csv`
- comparison JSON: `runs/cae_mesh_structured_q20_60_n24_boundary_seed13_topology/diagnostics/checkpoint_comparison.json`
- comparison PNG: `runs/cae_mesh_structured_q20_60_n24_boundary_seed13_topology/diagnostics/checkpoint_comparison.png`
- per-part visual evaluations: `visual_eval_best_val_loss`, `visual_eval_best_target_p95`, `visual_eval_best_boundary_within`, etc.

### Interpretation

Topology-boundary training moved the model in the intended direction for missed structures and open edges:

- target p95 improved from 26.888 mm to 19.334-19.615 mm.
- boundary p95 improved from 31.480 mm to 20.711-22.125 mm.
- boundary within 5mm improved from 0.165 to 0.227-0.270.

The cost is worse broad Chamfer and target-within-5mm:

- Chamfer increased from 21.109 mm to 22.149-23.448 mm for the best practical topology-boundary snapshots.
- target within 5mm decreased from 0.228 to 0.196-0.201 for the best p95/boundary snapshots.

This suggests the boundary loss is pulling capacity toward high-miss boundary regions, but the decoder still lacks enough topology-aware structure to improve global coverage and local precision at the same time. The best practical checkpoint for CAE-oriented boundary preservation is currently epoch 200 (`best_by_val_target_p95_mm.pt` / `best_by_val_boundary_within_threshold.pt`). The best balanced validation-loss checkpoint is epoch 190 (`best.pt`). The final epoch should not be used.

### Next Actions

The next model change should not be simply longer training. The evidence points to architecture/loss balance:

- Add explicit boundary tokens in the encoder rather than only oversampling boundary targets.
- Add supervised scaffold targets from boundary-aware FPS or curvature/boundary stratified FPS.
- Replace fixed global loss weights with scheduled or normalized multi-objective weighting, because boundary improvement currently trades off target-within-5mm.
- Add a composite checkpoint metric that balances target p95, boundary p95, boundary within 5mm, and Chamfer instead of selecting any single metric alone.
- Move toward CAE Mesh IR topology prediction so boundaries are represented as constraints, not only as point samples.

## Implementation Update: Feature Scaffold Supervision

Implemented automatic scaffold target extraction without manual labels:

- boundary targets from OCC topology open edges.
- crease targets from coordinate-welded triangle edges whose adjacent face normals differ by at least 30 degrees.
- corner targets from branch or sharp-turn endpoints on boundary/crease edge chains.
- area targets from regular surface sampling.

Dataset output now includes fixed-length `scaffold_targets` when `--scaffold-target-points > 0`. Evaluation writes `scaffold_targets.ply` next to `scaffold.ply` so scaffold placement can be inspected visually.

Training additions:

```powershell
--lambda-feature-scaffold
--scaffold-target-points
--scaffold-target-boundary-fraction
--scaffold-target-crease-fraction
--scaffold-target-corner-fraction
```

The loss is one-way target-to-scaffold distance. This is intentional: feature targets must be covered by some scaffold nodes, but all scaffold nodes are not forced to collapse onto boundary/crease/corner features.

Smoke:

```text
unit tests: 25 passed
feature scaffold train smoke: success
feature scaffold eval smoke: success
scaffold_targets.ply output: success
```

## Formal Run: Feature Scaffold Seed13

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_feature_scaffold_seed13
```

Configuration:

- same q20-60 n24 seed13 split as previous formal runs
- 300 epochs on CUDA
- topology-boundary sampling/loss enabled
- boundary token disabled, to isolate scaffold supervision
- `--lambda-feature-scaffold 0.2`
- `--scaffold-target-points 128`
- target mix: boundary 0.35, crease 0.35, corner 0.10, area remainder 0.20
- primary checkpoint: `val_cae_score`

### Checkpoint Comparison

| checkpoint | epoch | Chamfer mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| old coverage/scaffold best | 60 | 21.109 | 26.888 | 0.228 | 31.480 | 0.165 |
| topology boundary loss-best | 190 | 22.149 | 19.615 | 0.196 | 22.125 | 0.227 |
| topology boundary p95/boundary-best | 200 | 23.448 | 19.334 | 0.201 | 20.711 | 0.270 |
| boundary token CAE-score best | 80 | 24.255 | 18.506 | 0.197 | 23.248 | 0.215 |
| feature scaffold CAE-score best | 90 | 25.089 | 19.817 | 0.169 | 24.140 | 0.186 |
| feature scaffold val-loss best | 50 | 20.098 | 21.220 | 0.258 | 27.906 | 0.184 |
| feature scaffold boundary-within best | 100 | 24.721 | 19.355 | 0.196 | 23.051 | 0.251 |
| feature scaffold last | 300 | 27.010 | 22.849 | 0.174 | 24.598 | 0.189 |

Artifacts:

- comparison CSV: `runs/cae_mesh_structured_q20_60_n24_feature_scaffold_seed13/diagnostics/checkpoint_comparison.csv`
- comparison JSON: `runs/cae_mesh_structured_q20_60_n24_feature_scaffold_seed13/diagnostics/checkpoint_comparison.json`
- comparison PNG: `runs/cae_mesh_structured_q20_60_n24_feature_scaffold_seed13/diagnostics/checkpoint_comparison.png`

### Interpretation

Feature scaffold supervision is directionally useful, but the first weighting is not yet the best boundary configuration.

- The val-loss checkpoint improves broad reconstruction: Chamfer 20.098 mm and target within 5mm 0.258, better than the old coverage/scaffold checkpoint on both metrics.
- The boundary-within checkpoint reaches boundary within 5mm 0.251, close to the topology-boundary best 0.270.
- Boundary p95 remains worse than the topology-boundary best: 23.051-24.140 mm versus 20.711 mm.

This suggests the new scaffold targets help scaffold placement and global coverage, but the target mixture/loss weight is still too blunt for worst-boundary misses. The likely next experiment is a lower or scheduled feature-scaffold weight, plus separating boundary/crease/corner losses instead of merging them into one target cloud.

## Follow-up Run: Boundary Token and Composite CAE Score

Implemented:

- `--use-boundary-feature`: appends one boundary indicator column to input features.
- Boundary summary token: when boundary feature is enabled, the encoder pools boundary-marked input points into an extra memory token used by latent fusion and scaffold decoding.
- `val_cae_score`: a composite checkpoint metric. Lower is better. It combines target p95, boundary p95, approximate Chamfer in mm, and rewards target/boundary within-5mm coverage.
- Old checkpoint compatibility remains intact; old 7D checkpoints still evaluate with the 7D model, while new checkpoints save `feature_dim=8` and `boundary_feature_index=7`.

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_boundary_token_cae_seed13
```

Configuration was otherwise kept aligned with the topology-boundary run:

- q20-60 bbox diagonal, `max_parts=24`, random split seed 13
- 18 train / 6 val
- 300 epochs on CUDA
- boundary target fraction 0.25
- coverage/scaffold/boundary losses enabled
- primary checkpoint: `--best-metric val_cae_score`

### Checkpoint Comparison

| checkpoint | epoch | Chamfer mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| old coverage/scaffold best | 60 | 21.109 | 26.888 | 0.228 | 31.480 | 0.165 |
| topology boundary loss-best | 190 | 22.149 | 19.615 | 0.196 | 22.125 | 0.227 |
| topology boundary p95/boundary-best | 200 | 23.448 | 19.334 | 0.201 | 20.711 | 0.270 |
| boundary token CAE-score best | 80 | 24.255 | 18.506 | 0.197 | 23.248 | 0.215 |
| boundary token val-loss best | 180 | 22.809 | 22.402 | 0.183 | 26.585 | 0.150 |
| boundary token boundary-within best | 100 | 25.200 | 22.869 | 0.185 | 27.469 | 0.241 |
| boundary token last | 300 | 26.470 | 29.199 | 0.159 | 35.444 | 0.156 |

Artifacts:

- comparison CSV: `runs/cae_mesh_structured_q20_60_n24_boundary_token_cae_seed13/diagnostics/checkpoint_comparison.csv`
- comparison JSON: `runs/cae_mesh_structured_q20_60_n24_boundary_token_cae_seed13/diagnostics/checkpoint_comparison.json`
- comparison PNG: `runs/cae_mesh_structured_q20_60_n24_boundary_token_cae_seed13/diagnostics/checkpoint_comparison.png`

### Interpretation

Boundary feature/token is not a clear win yet.

- Good: target p95 improved further, from 19.334 mm in the best topology-boundary snapshot to 18.506 mm.
- Bad: boundary p95 worsened from 20.711 mm to 23.248 mm, and boundary within 5mm worsened from 0.270 to 0.215.
- Chamfer also worsened to 24.255 mm.

The likely issue is that a single pooled boundary token gives the model global awareness that boundaries exist, but not enough localized boundary topology. It may help reduce broad target misses, while failing to place boundary detail consistently.

### Updated Recommendation

Do not spend more time simply increasing epochs for the boundary-token variant. The next useful change is supervised scaffold placement:

- Generate scaffold supervision targets by stratified FPS: boundary samples, high-curvature/normal-change samples, and area samples.
- Add a scaffold assignment loss that forces a portion of scaffold nodes to cover boundary/open-edge regions.
- Keep `val_cae_score` as an auxiliary checkpoint selector, but compare it against `val_target_p95_mm` and `val_boundary_within_threshold` snapshots.
- Consider multiple boundary tokens or boundary patch tokens only after scaffold supervision is in place.

## Follow-up Run: Split Feature Scaffold Losses

Implemented separated scaffold supervision targets and losses for boundary, crease, and corner features.

The previous feature-scaffold experiment used one merged feature target cloud. That improved broad reconstruction but gave a blunt objective: boundary, crease, corner, and area samples could all compete through one loss term. The split version keeps the fixed-size typed target tensors for batching, but masks invalid feature slots and reports each feature type separately.

Implementation details:

- `MidsurfacePointCloudDataset` now emits `scaffold_boundary_targets`, `scaffold_crease_targets`, `scaffold_corner_targets`, and matching masks.
- The merged `scaffold_targets` cloud is backfilled with surface samples when a part lacks a requested feature type, so legacy direct helper use no longer treats zero-filled missing slots as valid geometry.
- `feature_scaffold_target_metrics` accepts an optional target mask.
- Training adds `--lambda-boundary-scaffold`, `--lambda-crease-scaffold`, and `--lambda-corner-scaffold`.
- Typed scaffold metrics are aggregated by valid target count, not by batch count.
- Evaluation writes `scaffold_boundary_targets.ply`, `scaffold_crease_targets.ply`, and `scaffold_corner_targets.ply` when present.

Validation:

```text
unit tests: 29 passed
split scaffold train smoke: success
split scaffold eval smoke: success
```

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_split_scaffold_seed13_v2
```

Configuration:

- same q20-60 n24 seed13 split as previous formal runs
- 300 epochs on CUDA
- topology-boundary sampling/loss enabled
- boundary token disabled
- merged feature scaffold loss disabled
- `--lambda-boundary-scaffold 0.25`
- `--lambda-crease-scaffold 0.10`
- `--lambda-corner-scaffold 0.10`
- `--scaffold-target-points 128`
- target mix: boundary 0.35, crease 0.35, corner 0.10, area/backfill remainder
- primary checkpoint: `val_cae_score`

### Checkpoint Comparison

| checkpoint | epoch | Chamfer mean mm | recon p95 mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| topology boundary p95/boundary-best | 200 | 23.448 | 30.706 | 19.334 | 0.201 | 20.711 | 0.270 |
| boundary token CAE-score best | 80 | 24.255 | n/a | 18.506 | 0.197 | 23.248 | 0.215 |
| merged feature val-loss best | 50 | 20.098 | 24.081 | 21.220 | 0.258 | 27.906 | 0.184 |
| merged feature boundary-within best | 100 | 24.721 | n/a | 19.355 | 0.196 | 23.051 | 0.251 |
| split scaffold CAE-score best | 50 | 25.134 | 36.340 | 18.520 | 0.192 | 20.486 | 0.143 |
| split scaffold boundary-within best | 120 | 26.840 | 38.081 | 20.008 | 0.167 | 21.311 | 0.161 |
| split scaffold last | 300 | 29.894 | 31.116 | 34.366 | 0.132 | 41.552 | 0.087 |

Artifacts:

- best visual eval: `runs/cae_mesh_structured_q20_60_n24_split_scaffold_seed13_v2_eval_all_best`
- boundary-within visual eval: `runs/cae_mesh_structured_q20_60_n24_split_scaffold_seed13_v2_eval_all_boundary_within`
- last visual eval: `runs/cae_mesh_structured_q20_60_n24_split_scaffold_seed13_v2_eval_all_last`

### Interpretation

Split scaffold supervision recovered worst-boundary p95 but did not recover local density.

- Boundary p95 improved to 20.486 mm, slightly better than the topology-boundary best at 20.711 mm.
- Target p95 improved to 18.520 mm, close to the boundary-token best at 18.506 mm.
- Chamfer worsened to 25.134 mm, and boundary within 5mm fell to 0.143.
- The final epoch is severe overfit: target p95 34.366 mm and boundary p95 41.552 mm.

Visual inspection of `A0072600002_AllCATPart:081` shows a recurring false-positive mode. The target is a thin, narrow strip, but the decoder emits points in a thicker volume around it. Target-to-reconstruction misses are moderate, while reconstruction-to-target errors are large. This means the model is not only missing target regions; it is also generating non-existing sheet area.

## Follow-up Run: Prediction Surface Loss

Implemented a symmetric counterpart to target coverage:

```text
prediction_surface_loss(pred, target) = mean(max(0, nearest(pred -> target) - threshold))
```

New training flags:

```powershell
--lambda-pred-surface
--pred-surface-threshold-mm
--cae-score-recon-p95-weight
--cae-score-pred-within-weight
```

The goal was to suppress generated points that drift away from the target midsurface. This was a direct response to the false-positive point clouds seen in the split scaffold visual evaluation.

Validation:

```text
unit tests: 30 passed
prediction surface train smoke: success
```

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_split_pred_surface_seed13
```

Configuration:

- same q20-60 n24 seed13 split
- same split scaffold typed losses as above
- `--lambda-pred-surface 0.3`
- `--pred-surface-threshold-mm 5.0`
- primary checkpoint: `val_cae_score`, now also including recon p95 and predicted-within-threshold reward

### Checkpoint Comparison

| checkpoint | epoch | Chamfer mean mm | recon p95 mean mm | target p95 mean mm | target within 5mm | recon within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| split scaffold CAE-score best | 50 | 25.134 | 36.340 | 18.520 | 0.192 | 0.148 | 20.486 | 0.143 |
| prediction-surface CAE-score best | 120 | 28.036 | 38.550 | 21.175 | 0.157 | 0.120 | 18.449 | 0.181 |
| prediction-surface val-loss best | 90 | 24.479 | 34.043 | 21.866 | 0.185 | 0.151 | 26.021 | 0.206 |
| prediction-surface last | 300 | 26.995 | 35.443 | 22.358 | 0.147 | 0.114 | 26.928 | 0.143 |

Artifacts:

- best visual eval: `runs/cae_mesh_structured_q20_60_n24_split_pred_surface_seed13_eval_all_best`
- val-loss visual eval: `runs/cae_mesh_structured_q20_60_n24_split_pred_surface_seed13_eval_all_best_loss`
- last visual eval: `runs/cae_mesh_structured_q20_60_n24_split_pred_surface_seed13_eval_all_last`

### Interpretation

Prediction surface loss did not solve the false-positive mode.

- It can improve one axis of the tradeoff: the CAE-score best checkpoint reached boundary p95 18.449 mm.
- It worsened global shape quality: Chamfer 28.036 mm and target-within-5mm 0.157.
- The val-loss checkpoint reduced recon p95 to 34.043 mm and improved recon-within-5mm to 0.151, but boundary p95 worsened to 26.021 mm.
- The hard case `A0072600002_AllCATPart:081` remained the worst Chamfer part, still around 35.9-37.7 mm depending on checkpoint.

This suggests point-cloud losses alone are now the bottleneck. They can move the tradeoff between boundary p95, coverage, and false positives, but they do not impose the sheet support structure strongly enough. The next useful architecture change should constrain local decoding itself:

- Decode local points in a scaffold-local tangent frame rather than free 3D offsets.
- Predict local patch scale, tangent axes, and boundary/crease type per scaffold node.
- Supervise `refinement_logits` or replacement occupancy logits so inactive scaffold neighborhoods stop emitting points.
- Move from unordered point reconstruction toward a CAE Mesh IR with explicit nodes, edges, boundary chains, and local patch connectivity.

In short: the current point-cloud AE has reached a useful diagnostic ceiling. Further progress likely needs topology-aware decoder structure, not only additional scalar losses.

## Follow-up Run: Typed Cross-Attention Lattice

Implemented an optional semantic interaction layer between the encoder streams:

```text
global/coarse tokens  = part-scale structural context
local/patch tokens    = local midsurface evidence
boundary tokens       = open-edge / CAE constraint evidence
```

The design intent is described in:

```text
docs/cae_typed_cross_attention_lattice_design.md
```

The important architectural change is that global, local, and boundary streams no longer meet only at
the final latent fusion. They pass through `TypedCrossAttentionLattice`, where each stream has a
typed role:

- global reads local and boundary evidence so the part-scale scaffold can see thin strips and edge constraints.
- local reads global and boundary evidence so refinement is less free to emit detached 3D points.
- boundary reads global and local evidence so edge constraints remain attached to the correct sheet patches.

The first formal implementation kept the decoder unchanged. This intentionally isolated the effect of
semantic token interaction before changing the output representation.

Validation:

```text
unit tests: 33 passed
CUDA train/eval/resume smoke: success
```

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13
```

Configuration:

- same q20-60 n24 seed13 split
- 300 epochs on CUDA
- `--use-boundary-feature`
- `--boundary-token-count 4`
- `--lattice-layers 1`
- `--lattice-heads 4`
- topology-boundary sampling/loss enabled
- split scaffold typed losses enabled
- prediction surface loss disabled
- primary checkpoint: `val_cae_score`

### Lattice Checkpoint Comparison

Metrics below are from the full 24-part evaluation outputs, using the validation split rows
(`count=6`) in each `aggregate_metrics.json`.

| checkpoint | epoch | Chamfer mean mm | recon p95 mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| topology boundary p95/boundary-best | 200 | 23.448 | 30.706 | 19.334 | 0.201 | 20.711 | 0.270 |
| boundary token CAE-score best | 80 | 24.255 | 33.658 | 18.506 | 0.197 | 23.248 | 0.215 |
| split scaffold CAE-score best | 50 | 25.134 | 36.340 | 18.520 | 0.192 | 20.486 | 0.143 |
| prediction-surface CAE-score best | 120 | 28.036 | 38.550 | 21.175 | 0.157 | 18.449 | 0.181 |
| lattice CAE-score best | 70 | 35.145 | 48.299 | 22.176 | 0.113 | 22.513 | 0.113 |
| lattice val-loss best | 100 | 28.027 | 38.401 | 22.924 | 0.158 | 28.707 | 0.151 |
| lattice target-p95 best | 90 | 30.432 | 42.157 | 21.084 | 0.136 | 25.847 | 0.115 |
| lattice last | 300 | 38.975 | 49.044 | 31.656 | 0.092 | 35.928 | 0.055 |

Artifacts:

- CAE-score eval: `runs/cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13_eval_all24_best`
- val-loss eval: `runs/cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13_eval_all24_val_loss`
- target-p95 eval: `runs/cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13_eval_all24_target_p95`
- last eval: `runs/cae_mesh_structured_q20_60_n24_lattice_boundary_tokens_seed13_eval_all24_last`

### Interpretation

Typed Cross-Attention Lattice did not improve the current structured point-cloud autoencoder.

- The best lattice checkpoint by CAE score has much worse Chamfer than previous candidates: 35.145 mm.
- The best lattice checkpoint by validation loss improves Chamfer relative to the CAE-score checkpoint, but still trails topology-boundary, boundary-token, split-scaffold, and prediction-surface baselines.
- Boundary p95 does not improve over topology-boundary, split-scaffold, or prediction-surface results.
- Target-within-5mm and boundary-within-5mm are both weak; the lattice did not make the decoder place dense local surface support more reliably.
- The last checkpoint is clearly overfit, with target p95 31.656 mm and boundary p95 35.928 mm.

The negative result is useful. It suggests that the bottleneck is not only shallow interaction between
global/local/boundary evidence. The model can give those streams a better semantic conversation, but
the unchanged decoder still emits unordered local point offsets. That decoder can still create false
sheet area, miss thin strips, and trade boundary accuracy against local density.

Therefore the next architecture change should move to the decoder:

- constrain local refinement in scaffold-local tangent frames;
- predict patch axes, patch scale, and patch type per scaffold node;
- supervise occupancy / `refinement_logits` so inactive scaffold neighborhoods stop emitting points;
- start representing CAE Mesh IR topology, especially nodes, edges, boundary chains, and patch connectivity.

In short: Lattice is a useful, optional encoder/fusion module, but not the next performance lever by
itself. The next performance lever is a topology-aware and frame-constrained decoder.

## Decoder Follow-up: Tangent-Frame Refinement

Implemented the next decoder-side candidate behind:

```powershell
--refinement-mode tangent
```

Design note:

```text
docs/cae_tangent_frame_decoder_design.md
```

The decoder now supports scaffold-local shell patch generation:

- per-scaffold normal prediction;
- in-plane tangent-axis rotation;
- per-scaffold tangent/normal patch scales;
- unsupervised patch type logits;
- per-generated-point refinement occupancy logits;
- weak occupancy supervision from generated-point nearest distance to the target midsurface.

New training flags:

```powershell
--tangent-offset-scale
--normal-offset-scale
--patch-type-count
--lambda-refinement-occupancy
--occupancy-positive-threshold-mm
--occupancy-negative-threshold-mm
```

Validation:

```text
unit tests: 35 passed
compileall: success
CUDA tangent train smoke: success
CUDA tangent eval smoke: success
CUDA tangent resume smoke: success
```

Smoke artifacts:

```text
runs/cae_mesh_smoke_tangent_decoder_v1
runs/cae_mesh_smoke_tangent_decoder_v1_eval
runs/cae_mesh_smoke_tangent_decoder_v1_resume
```

This is not yet a quality claim. The smoke run used only two small parts for one epoch. The result only
proves that the new decoder can train, save, evaluate, and resume. The next formal comparison should
use the same q20-60 n24 seed13 split and keep the lattice disabled initially, so the decoder change is
isolated from the encoder/fusion change.

### Formal q20-60 n24 Seed13 Tangent Result

Formal run:

```text
runs/cae_mesh_structured_q20_60_n24_tangent_decoder_seed13
```

Evaluation outputs:

```text
runs/cae_mesh_structured_q20_60_n24_tangent_decoder_seed13_eval_all24_best
runs/cae_mesh_structured_q20_60_n24_tangent_decoder_seed13_eval_all24_val_loss
runs/cae_mesh_structured_q20_60_n24_tangent_decoder_seed13_eval_all24_refinement_occupancy
runs/cae_mesh_structured_q20_60_n24_tangent_decoder_seed13_eval_all24_last
```

History selected early checkpoints:

| selection metric | epoch | value | validation notes |
|---|---:|---:|---|
| `val_cae_score` | 50 | 76.143 | same as target-p95 best |
| `val_loss` | 40 | 0.450927 | same as boundary-p95 best |
| `val_refinement_occupancy` | 10 | 0.258020 | geometry still immature |

Validation-split aggregate metrics:

| checkpoint | epoch | Chamfer mean mm | recon p95 mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| tangent CAE-score best | 50 | 44.288 | 61.379 | 31.371 | 0.075 | 36.983 | 0.061 |
| tangent val-loss best | 40 | 46.963 | 60.235 | 32.630 | 0.071 | 34.486 | 0.068 |
| tangent occupancy best | 10 | 63.864 | 83.190 | 37.926 | 0.029 | 44.389 | 0.020 |
| tangent last | 300 | 60.057 | 75.941 | 46.005 | 0.049 | 49.896 | 0.035 |

Prior stronger validation checkpoints for comparison:

| checkpoint | Chamfer mean mm | recon p95 mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| topology-boundary boundary-best | 23.448 | 30.706 | 19.334 | 0.201 | 20.711 | 0.270 |
| boundary-token CAE best | 24.255 | 33.658 | 18.506 | 0.197 | 23.248 | 0.215 |
| split-scaffold CAE best | 25.134 | 36.340 | 18.520 | 0.192 | 20.486 | 0.143 |
| prediction-surface CAE best | 28.036 | 38.550 | 21.175 | 0.157 | 18.449 | 0.181 |
| lattice val-loss best | 28.027 | 38.401 | 22.924 | 0.158 | 28.707 | 0.151 |
| tangent CAE-score best | 44.288 | 61.379 | 31.371 | 0.075 | 36.983 | 0.061 |

Interpretation:

Tangent-frame decoding is not beneficial in the current architecture.

- The train split is memorized well by epoch300, but the validation split collapses. Last checkpoint train Chamfer is 8.519 mm while validation Chamfer is 60.057 mm.
- The best validation checkpoint is much worse than the earlier topology-boundary and split-scaffold baselines.
- Worst case remains `A0072600002_AllCATPart:081`; visual inspection shows the decoder emits a large sheet-like patch cloud around a thin target strip instead of aligning to the strip.
- Occupancy logits are not reliable yet. The occupancy-best checkpoint deactivates too much before geometry forms; later checkpoints activate nearly all generated points.

The failure is useful because it isolates the next bottleneck: local patch parameterization alone is not
enough. The scaffold proposal itself must be better anchored to encoded geometry, and occupancy must
affect topology/export rather than remain an auxiliary scalar.

Next recommended decoder direction:

- anchor scaffold candidates to encoded fine/coarse centers and predict residuals;
- keep a learned scaffold query path only as a global proposal supplement;
- make active occupancy gate exported local neighborhoods;
- move local output from unordered points toward CAE Mesh IR nodes, edges, boundary chains, and patch connectivity.

## Data Follow-up: Train-only Mirror Augmentation

Implemented train-only mirror augmentation for scarce part diversity.

The key rule is that augmentation is applied after part-level train/validation split:

```text
records -> part-level split -> train dataset augmentation only
```

This avoids evaluation leakage. A validation part never appears in mirrored form in the training set.

New training flag:

```powershell
--train-mirror-axes y
```

Implementation behavior:

- `MidsurfacePointCloudDataset` expands only the dataset instance that receives `mirror_axes`.
- `--train-mirror-axes y` makes train samples `original + mirror_y`.
- validation and evaluation datasets remain unaugmented.
- points, normals, scaffold targets, boundary targets, crease targets, and corner targets are mirrored consistently in normalized coordinates.
- checkpoint args preserve `train_mirror_axes` for resume.

Validation:

```text
unit tests: 37 passed
compileall: success
CUDA mirror train smoke: success
```

Smoke run:

```text
runs/cae_mesh_smoke_mirror_aug_y_v1
```

The smoke confirmed:

```text
split: train=3 val=1
train_augmentation: mirror_axes=['y'] base_records=3 samples=6
preload_data: split=train items=6
preload_data: split=val items=1
```

### Formal q20-60 n24 Seed13 Mirror-y Result

Formal run:

```text
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13
```

Configuration:

- same q20-60 n24 seed13 split as previous formal runs
- train base parts: 18
- validation parts: 6
- train samples after `--train-mirror-axes y`: 36
- validation samples: 6 original parts only
- structured free decoder, lattice disabled, tangent decoder disabled
- topology-boundary sampling/loss enabled
- typed scaffold losses enabled

History selected:

| selection metric | epoch | value | notes |
|---|---:|---:|---|
| `val_cae_score` | 80 | 44.628 | also target/boundary p95 best |
| `val_loss` | 70 | 0.228592 | slightly worse CAE metrics |
| last | 300 | 58.772 CAE score | overfit relative to epoch80 |

Evaluation outputs:

```text
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_best
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_val_loss
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_target_p95
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_boundary_p95
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_last
```

Validation-split aggregate comparison:

| checkpoint | Chamfer mean mm | recon p95 mean mm | target p95 mean mm | target within 5mm | boundary p95 mean mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| mirror-y CAE best | 23.010 | 29.189 | 19.851 | 0.194 | 21.778 | 0.160 | `081` | 28.594 |
| mirror-y val-loss best | 23.876 | 30.256 | 21.062 | 0.189 | 22.605 | 0.169 | `081` | 32.257 |
| mirror-y last | 25.487 | 30.093 | 25.344 | 0.152 | 30.418 | 0.113 | `A0072601285:10` | 30.653 |
| topology-boundary boundary-best | 23.448 | 30.706 | 19.334 | 0.201 | 20.711 | 0.270 | `081` | 35.969 |
| boundary-token CAE best | 24.255 | 33.658 | 18.506 | 0.197 | 23.248 | 0.215 | `081` | 32.532 |
| split-scaffold CAE best | 25.134 | 36.340 | 18.520 | 0.192 | 20.486 | 0.143 | `081` | 34.785 |
| prediction-surface CAE best | 28.036 | 38.550 | 21.175 | 0.157 | 18.449 | 0.181 | `081` | 37.739 |
| tangent CAE best | 44.288 | 61.379 | 31.371 | 0.075 | 36.983 | 0.061 | `081` | 79.294 |

Interpretation:

Mirror-y augmentation is the first recent change that clearly improves the worst-case held-out part
without changing the model architecture.

- It improves validation Chamfer slightly versus topology-boundary best: 23.448 -> 23.010 mm.
- It improves reconstruction p95: 30.706 -> 29.189 mm.
- It dramatically improves the recurring worst case `A0072600002_AllCATPart:081`: 35.969 -> 28.594 mm versus topology-boundary best, and 79.294 -> 28.594 mm versus tangent best.
- It does not improve every metric. Boundary p95 and boundary-within remain weaker than topology-boundary best.
- Last epoch still overfits, so early stopping remains important.

Conclusion:

Data diversity was a real bottleneck. Train-only mirroring is worth keeping as a default augmentation
candidate for subsequent formal runs, but it should be treated as a geometry-preserving augmentation
whose axis must match the vehicle/CATIA coordinate convention. The next experiment should compare
`--train-mirror-axes y` against `x`, `z`, and possibly multi-axis augmentation, while keeping validation
unaugmented.

## fable5 Diagnostic Follow-up: Evaluation Calibration and Epoch Resampling

fable5 reviewed `docs/fable5_local_diagnostic_report.md` and returned
`docs/fable5_diagnostic_response.md`.

The key correction is that mirror-axis ablation should not be the immediate next step. Two lower-level
issues can distort every subsequent comparison:

1. The 512-point evaluation target has a non-trivial sampling floor. Even two independent point samples
   from the same midsurface can have several mm of Chamfer distance, so small experiment differences
   may be evaluation noise rather than model behavior.
2. Training samples were fixed for every epoch. Each train part exposed the same sampled points
   throughout a 300-epoch run, so part diversity and within-part point diversity were both limited.

Implemented P0 fixes:

- `evaluate_autoencoder.py` now separates model input point count from metric target point count.
- Use `--metric-target-n-points 4096` to evaluate a 512-point reconstruction against a denser target.
- Evaluation now estimates an independent target-sampling floor per part and reports:
  - `sampling_floor_chamfer_mm`
  - `sampling_floor_bbox_diag_pct`
  - `chamfer_bbox_diag_pct`
  - `chamfer_to_sampling_floor`
- `train_autoencoder.py` now supports train-only epoch resampling with:

```powershell
--resample-train-each-epoch
```

Important behavior:

- validation sampling remains fixed for historical comparability.
- `--resample-train-each-epoch` cannot be combined with `--preload-data`, because preloading freezes
  samples.
- if `num_workers > 0`, train DataLoader persistent workers are disabled under epoch resampling so
  worker datasets receive the updated epoch sample seed.

Validation:

```text
unit tests: 39 passed
compileall: success
calibrated eval smoke: runs/cae_mesh_eval_calibrated_smoke
resample train smoke: runs/cae_mesh_smoke_resample_epoch_v1
```

Calibrated evaluation smoke command:

```powershell
$env:PYTHONPATH="C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src"
python -m cae_mesh_generator.evaluate_autoencoder `
  --checkpoint runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13\best.pt `
  --output-dir runs\cae_mesh_eval_calibrated_smoke `
  --split val `
  --max-parts 1 `
  --metric-target-n-points 1024 `
  --no-html `
  --device cpu
```

Smoke result for `A0072600002_AllCATPart:026`:

```text
Chamfer: 17.332mm
sampling floor: 3.413mm
Chamfer/floor: 5.078x
Chamfer/bbox diagonal: 10.354%
```

Formal calibrated re-evaluation of mirror-y best:

```text
runs/cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_best_calibrated4096
```

Command:

```powershell
$env:PYTHONPATH="C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src"
python -m cae_mesh_generator.evaluate_autoencoder `
  --checkpoint runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13\best.pt `
  --output-dir runs\cae_mesh_structured_q20_60_n24_mirror_y_seed13_eval_all24_best_calibrated4096 `
  --split all `
  --max-parts 0 `
  --metric-target-n-points 4096 `
  --no-html `
  --device cuda
```

Aggregate results:

| split | count | Chamfer mean mm | Chamfer / bbox diag mean | sampling floor mean mm | floor / bbox diag mean | Chamfer / floor mean | target p95 mean mm | boundary p95 mean mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 24 | 12.791 | 5.314% | 2.159 | 0.930% | 5.979x | 12.985 | 14.342 |
| train | 18 | 9.676 | 4.408% | 2.107 | 0.964% | 4.604x | 10.671 | 11.877 |
| val | 6 | 22.137 | 8.032% | 2.315 | 0.829% | 10.107x | 19.927 | 21.736 |

Worst floor-normalized parts:

| part | split | Chamfer mm | Chamfer / bbox diag | sampling floor mm | Chamfer / floor |
|---|---|---:|---:|---:|---:|
| `A0072600002_AllCATPart:081` | val | 28.390 | 9.118% | 1.740 | 16.316x |
| `A0072600002_AllCATPart:077` | val | 22.962 | 8.343% | 2.107 | 10.898x |
| `A0072600002_AllCATPart:026` | val | 17.176 | 10.257% | 1.639 | 10.478x |
| `A0072601285_AllCATPart:10` | val | 24.310 | 7.649% | 2.716 | 8.951x |

Interpretation:

- Dense target evaluation reduces the absolute mean Chamfer versus the earlier 512-target report, but it
  makes the sampling floor explicit.
- The train/val gap remains clear after floor normalization: train is 4.604x floor while validation is
  10.107x floor.
- `081` is still the most serious failure once normalized by evaluation floor, so it is not merely a
  large/small part scale artifact.
- This supports fable5's recommendation: run epoch resampling next, then redo mirror-axis ablation under
  calibrated metrics.

Revised next formal sequence:

1. Re-test mirror axis ablation under epoch resampling, starting from x/z because y has now been tested.
2. Run scaffold placement diagnostic before implementing scaffold anchoring.
3. Move to scaffold anchoring only if the diagnostic confirms scaffold placement is the failure mode.

## Formal Follow-up: Epoch Resampling and Mirror-y Interaction

Two formal q20-60 n24 seed13 runs were completed after the P0 changes.

### Run A: Epoch Resampling without Mirror

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_resample_seed13
```

Evaluation directory:

```text
runs/cae_mesh_structured_q20_60_n24_resample_seed13_eval_all24_best_calibrated4096
```

Configuration delta from old mirror-y:

- `--resample-train-each-epoch` enabled.
- `--train-mirror-axes` omitted.
- `--preload-data` omitted because it would freeze samples.
- all model/loss/split settings otherwise matched the formal q20-60 n24 seed13 configuration.

Best training checkpoint:

| selection | epoch | val CAE score | val loss | val Chamfer normalized | val target p95 mm | val target within 5mm | val boundary p95 mm | val boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resample only | 50 | 52.488 | 0.2853 | 0.1452 | 23.752 | 0.127 | 23.416 | 0.142 |

4096-target calibrated evaluation:

| split | Chamfer mean mm | Chamfer / floor mean | recon p95 mean mm | target p95 mean mm | boundary p95 mean mm | target within 5mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| train | 14.515 | 6.883x | 16.666 | 14.495 | 15.039 | 0.347 | 0.340 | `A0072600002_AllCATPart:054` | 22.138 |
| val | 31.231 | 14.316x | 40.855 | 24.623 | 25.099 | 0.129 | 0.133 | `A0072600002_AllCATPart:081` | 52.089 |

Interpretation:

- Epoch resampling alone does not improve validation generalization in this configuration.
- It prevents the old fixed-sample memorization pattern from being the only training signal, but it also
  makes optimization harder without adding new part-level diversity.
- The difficult thin-strip part `081` becomes much worse: 52.089mm Chamfer and 29.935x the sampling floor.
- Therefore fable5 was correct that fixed sampling was a methodological issue, but resampling is a
  measurement/training hygiene fix rather than a sufficient augmentation by itself.

### Run B: Epoch Resampling plus Train-only Mirror-y

Run directory:

```text
runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13
```

Evaluation directory:

```text
runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13_eval_all24_best_calibrated4096
```

Configuration delta from Run A:

- `--train-mirror-axes y` added.
- train dataset uses original + mirror-y variants, both resampled each epoch.
- validation remains original, fixed, and unaugmented.

Best training checkpoint:

| selection | epoch | val CAE score | val loss | val Chamfer normalized | val target p95 mm | val target within 5mm | val boundary p95 mm | val boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resample + mirror-y | 30 | 38.359 | 0.2219 | 0.1050 | 17.832 | 0.223 | 19.260 | 0.212 |

4096-target calibrated evaluation:

| split | Chamfer mean mm | Chamfer / floor mean | recon p95 mean mm | target p95 mean mm | boundary p95 mean mm | target within 5mm | boundary within 5mm | worst Chamfer part | worst Chamfer mm |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| train | 12.624 | 6.030x | 14.188 | 12.683 | 14.182 | 0.400 | 0.322 | `A0072601285_AllCATPart:11` | 17.093 |
| val | 21.724 | 9.717x | 32.382 | 17.272 | 18.288 | 0.217 | 0.216 | `A0072601285_AllCATPart:10` | 27.707 |

### Calibrated Comparison

All rows use 4096-point metric targets and the same q20-60 n24 seed13 split.

| run | train augmentation | train resampling | best epoch | val Chamfer mm | val Chamfer/floor | val target p95 mm | val boundary p95 mm | val target within 5mm | val boundary within 5mm | worst val Chamfer |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| old mirror-y | mirror-y | fixed samples | 80 | 22.137 | 10.107x | 19.927 | 21.736 | 0.188 | 0.155 | `081` 28.390mm |
| resample only | none | epoch resample | 50 | 31.231 | 14.316x | 24.623 | 25.099 | 0.129 | 0.133 | `081` 52.089mm |
| resample + mirror-y | mirror-y | epoch resample | 30 | 21.724 | 9.717x | 17.272 | 18.288 | 0.217 | 0.216 | `1285:10` 27.707mm |

Key conclusions:

- Mirror-y remains valuable after epoch resampling, so its earlier benefit was not merely caused by
  providing the only non-identical point sample.
- Epoch resampling and mirror-y together give the current strongest calibrated result.
- The most important improvement is not raw Chamfer alone. The stronger signal is target p95
  `19.927 -> 17.272mm`, boundary p95 `21.736 -> 18.288mm`, and boundary within-5mm
  `0.155 -> 0.216`.
- The recurring worst part `081` is no longer the worst Chamfer part under resample + mirror-y:
  it improves from 52.089mm in resample-only and 28.390mm in old mirror-y to 25.178mm.
- However, validation still overfits early. Best epoch moves to 30, while epoch 300 degrades to
  val CAE score 55.174. This strengthens the case for early stopping, EMA, or checkpoint averaging.

Recommended next experiments:

1. Run the same epoch-resampling configuration with `--train-mirror-axes x`.
2. Run the same epoch-resampling configuration with `--train-mirror-axes z`.
3. If x/z are close to y, interpret mirror as generic geometry-preserving augmentation.
4. If y remains uniquely strong, treat it as a likely vehicle/CATIA left-right symmetry prior.
5. After axis ablation, run the scaffold placement diagnostic before implementing scaffold anchoring.

## Formal Follow-up: Mirror Axis Ablation under Epoch Resampling

The recommended x/z axis ablations were run under the same q20-60 n24 seed13 split, same model/loss
configuration, epoch train resampling enabled, and 4096-point calibrated evaluation.

Run directories:

```text
runs/cae_mesh_structured_q20_60_n24_resample_mirror_x_seed13
runs/cae_mesh_structured_q20_60_n24_resample_mirror_z_seed13
```

Evaluation directories:

```text
runs/cae_mesh_structured_q20_60_n24_resample_mirror_x_seed13_eval_all24_best_calibrated4096
runs/cae_mesh_structured_q20_60_n24_resample_mirror_z_seed13_eval_all24_best_calibrated4096
```

Training-time best checkpoints:

| axis | best epoch | val CAE score | val loss | val Chamfer normalized | val target p95 mm | val boundary p95 mm | val target within 5mm | val boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 50 | 52.488 | 0.2853 | 0.1452 | 23.752 | 23.416 | 0.127 | 0.142 |
| x | 100 | 44.816 | 0.2392 | 0.1139 | 19.465 | 22.831 | 0.206 | 0.173 |
| y | 30 | 38.359 | 0.2219 | 0.1050 | 17.832 | 19.260 | 0.223 | 0.212 |
| z | 110 | 39.470 | 0.2269 | 0.1110 | 18.232 | 19.266 | 0.201 | 0.206 |

4096-target calibrated validation aggregates:

| axis | val Chamfer mm | val Chamfer/floor | recon p95 mean mm | target p95 mean mm | boundary p95 mean mm | target within 5mm | boundary within 5mm | worst val Chamfer |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| none | 31.231 | 14.316x | 40.855 | 24.623 | 25.099 | 0.129 | 0.133 | `081` 52.089mm |
| x | 24.323 | 11.424x | 36.299 | 19.896 | 23.059 | 0.204 | 0.177 | `081` 42.251mm |
| y | 21.724 | 9.717x | 32.382 | 17.272 | 18.288 | 0.217 | 0.216 | `1285:10` 27.707mm |
| z | 23.381 | 10.874x | 35.013 | 18.244 | 20.291 | 0.194 | 0.194 | `081` 40.103mm |

Selected difficult validation parts:

| part | no mirror Chamfer | mirror-x Chamfer | mirror-y Chamfer | mirror-z Chamfer | note |
|---|---:|---:|---:|---:|---|
| `A0072600002_AllCATPart:081` | 52.089 | 42.251 | 25.178 | 40.103 | y is clearly best for the recurring thin-strip false-positive mode |
| `A0072600002_AllCATPart:077` | 25.607 | 30.584 | 20.017 | 18.907 | z is best, y close, x worsens |
| `A0072600002_AllCATPart:024` | 39.050 | 18.010 | 23.925 | 24.040 | x is best for this part |
| `A0072601285_AllCATPart:10` | 26.515 | 18.995 | 27.707 | 24.533 | x is best; y makes this the worst aggregate part |

Interpretation:

- Mirror augmentation is generally useful. Every mirrored axis beats the no-mirror resampling run on
  validation Chamfer, target p95, boundary p95, and within-5mm metrics.
- y remains the best aggregate axis and is uniquely strong on `081`, the most recurring thin-strip
  failure case.
- z is close to y by training CAE score and boundary p95, but its 4096-target validation Chamfer and
  `081` false-positive behavior remain meaningfully worse.
- x helps some parts, especially `024` and `1285:10`, but it leaves `081` badly overgenerated and has
  the weakest aggregate p95/boundary metrics among the mirrored axes.
- Therefore the current evidence supports both readings:
  - mirror itself is a useful geometry-preserving augmentation;
  - y likely carries an additional coordinate/family prior that protects thin-strip cases.

Updated recommendation:

1. Keep `--resample-train-each-epoch --train-mirror-axes y` as the current default single-axis
   augmentation.
2. Test multi-axis augmentation cautiously, starting with `--train-mirror-axes y z`; avoid assuming
   all axes are equally safe.
3. Add scaffold placement diagnostics next, because x/z/y differences are part-specific and likely
   tied to false-positive scaffold placement rather than encoder capacity alone.

## Scaffold Placement Diagnostic

The next diagnostic was added as:

```powershell
python -m cae_mesh_generator.diagnose_scaffold_placement `
  --checkpoint runs\cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13\best.pt `
  --output-dir runs\cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13_scaffold_diag4096 `
  --split all `
  --max-parts 0 `
  --metric-target-n-points 4096 `
  --device cuda
```

This evaluates the generated `scaffold_points` directly against high-resolution target surface,
boundary, crease, and corner samples. It is intentionally separate from reconstruction Chamfer,
because a locally refined point cloud can partially hide a bad coarse scaffold.

Key aggregate results for the current best learned-scaffold model:

| split | scaffold p95 mean | target-to-scaffold p95 mean | boundary-to-scaffold p95 mean | scaffold within 5mm | target within 5mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| train | 33.451mm | 22.813mm | 25.376mm | 0.228 | 0.065 | 0.068 |
| val | 70.497mm | 36.248mm | 35.734mm | 0.057 | 0.017 | 0.026 |
| all | 42.713mm | 26.172mm | 27.965mm | 0.185 | 0.053 | 0.057 |

Worst validation scaffold p95 parts:

| part | scaffold p95 | target-to-scaffold p95 | boundary-to-scaffold p95 |
|---|---:|---:|---:|
| `A0072600002_AllCATPart:081` | 104.118mm | 37.084mm | 38.280mm |
| `A0072600002_AllCATPart:024` | 74.612mm | 40.618mm | 40.063mm |
| `A0072601285_AllCATPart:10` | 71.741mm | 41.303mm | 32.723mm |
| `A0072600002_AllCATPart:077` | 70.431mm | 37.023mm | 36.902mm |
| `A0072600002_AllCATPart:074` | 61.536mm | 41.871mm | 44.515mm |

Correlation with calibrated 4096-target Chamfer:

| split | n | corr(scaffold p95, Chamfer) | scaffold p95 mean | Chamfer mean |
|---|---:|---:|---:|---:|
| train | 18 | 0.670 | 33.451mm | 12.624mm |
| val | 6 | 0.739 | 70.497mm | 21.724mm |
| all | 24 | 0.898 | 42.713mm | 14.899mm |

Interpretation:

- The learned free scaffold queries generalize poorly on validation parts. The train/val gap is much
  larger for scaffold placement than for final Chamfer.
- The recurring `081` false-positive mode is now visible one stage earlier: its scaffold is already
  far from the midsurface before local refinement.
- More encoder attention is unlikely to be the first fix. The immediate bottleneck is decoder
  geometry anchoring: scaffold candidates should start from actual encoded centers, then predict
  bounded residuals.

## Implementation Update: Anchored Scaffold Decoder

An anchor-based scaffold mode was implemented behind:

```powershell
--scaffold-mode anchored `
--scaffold-anchor-source coarse_fine `
--scaffold-anchor-residual-scale 0.08
```

The old behavior remains the default:

```powershell
--scaffold-mode learned
```

Model-level change:

1. `StructuredScaffoldAutoencoder.encode_memory()` already returns `coarse_centers`, `fine_centers`,
   and their corresponding tokens.
2. `select_scaffold_anchors()` selects candidate centers from `coarse`, `fine`, or `coarse_fine`.
   If there are enough candidates, FPS chooses `n_scaffold` anchors. If candidates are fewer than
   `n_scaffold`, anchors are repeated and differentiated by slot embeddings.
3. `AnchoredScaffoldDecoder` starts each scaffold from:

   ```text
   scaffold = clamp(anchor + tanh(residual_head(token)) * residual_scale, -0.75, 0.75)
   ```

4. The local refinement decoder is unchanged. It now receives scaffold points that are constrained
   to remain near encoder-derived midsurface evidence.

Checkpoint/config additions:

- `scaffold_mode`
- `scaffold_anchor_source`
- `scaffold_anchor_residual_scale`

Smoke validation:

| check | result |
|---|---|
| compile | passed |
| unit tests | 43 passed |
| CUDA train smoke | `runs/cae_mesh_smoke_anchored_scaffold_v1` |
| eval smoke | `runs/cae_mesh_smoke_anchored_scaffold_v1_eval` |
| scaffold diagnostic smoke | `runs/cae_mesh_smoke_anchored_scaffold_v1_scaffold_diag` |

Short smoke is not a performance claim, but it confirms that train, checkpoint save/load, evaluation,
and scaffold diagnostics all work with anchored checkpoints. The next formal experiment should compare
anchored scaffold decoding against the current learned-scaffold best under the same q20-60 n24 seed13
conditions, including epoch resampling, train-only mirror-y, boundary sampling, and 4096-target
calibrated evaluation.

## Formal Run: Resample + Mirror-Y + Anchored Scaffold

Formal run:

- `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13`
- checkpoint evaluated: `best.pt`
- best epoch: 290
- primary metric: `val_cae_score`
- same split and data contract as the learned-scaffold `resample_mirror_y` run

Main architecture delta:

```powershell
--scaffold-mode anchored `
--scaffold-anchor-source coarse_fine `
--scaffold-anchor-residual-scale 0.08
```

Training checkpoint metrics:

| checkpoint | epoch | val CAE score | val target p95 | val boundary p95 | val target within 5mm | val boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| best CAE | 290 | 4.441 | 9.002mm | 8.231mm | 0.653 | 0.745 |
| last | 300 | 5.173 | 9.189mm | 8.537mm | 0.638 | 0.736 |
| val loss best | 280 | 4.849 | 9.092mm | 8.353mm | 0.641 | 0.736 |

The run did not simply peak at the beginning. It improved substantially from epoch 180 through epoch
290, suggesting that anchor placement solved the coarse false-positive mode and gave local refinement
enough time to improve coverage.

### 4096-Target Evaluation

Evaluation output:

- `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13_eval_all24_best_calibrated4096`

Comparison against the previous current best:

| model | val Chamfer | val Chamfer / sampling floor | val target p95 | val boundary p95 | val target within 5mm | val boundary within 5mm | worst val Chamfer |
|---|---:|---:|---:|---:|---:|---:|---|
| learned scaffold + resample + mirror-y | 21.724mm | 9.717x | 17.272mm | 18.288mm | 0.217 | 0.216 | `A0072601285_AllCATPart:10`, 27.707mm |
| anchored scaffold + resample + mirror-y | 6.317mm | 2.733x | 9.416mm | 8.565mm | 0.548 | 0.687 | `A0072600002_AllCATPart:024`, 7.872mm |

All 4096-target splits:

| split | Chamfer | Chamfer / floor | target p95 | boundary p95 | target within 5mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| train | 5.639mm | 2.676x | 8.502mm | 7.841mm | 0.604 | 0.741 |
| val | 6.317mm | 2.733x | 9.416mm | 8.565mm | 0.548 | 0.687 |
| all | 5.808mm | 2.690x | 8.730mm | 8.022mm | 0.590 | 0.727 |

Former recurring worst part `A0072600002_AllCATPart:081` improved to:

| part | split | Chamfer | Chamfer / floor | recon p95 | target p95 | target within 5mm | boundary p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0072600002_AllCATPart:081` | val | 4.740mm | 2.72x | 2.615mm | 6.857mm | 0.766 | 7.063mm |

Current worst validation part is now `A0072600002_AllCATPart:024`:

| part | split | Chamfer | Chamfer / floor | recon p95 | target p95 | target within 5mm | boundary p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0072600002_AllCATPart:024` | val | 7.872mm | 2.69x | 3.358mm | 11.754mm | 0.380 | 10.893mm |

This is a qualitatively different failure. `024` no longer shows a far-away scaffold; instead it has
remaining target-miss coverage inside a correctly localized part envelope.

### Scaffold Diagnostic

Diagnostic output:

- `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13_scaffold_diag4096`

Direct scaffold placement comparison:

| model | val scaffold p95 mean | val target-to-scaffold p95 mean | val boundary-to-scaffold p95 mean | worst scaffold p95 |
|---|---:|---:|---:|---:|
| learned scaffold + resample + mirror-y | 70.497mm | 36.248mm | 35.734mm | `081`, 104.118mm |
| anchored scaffold + resample + mirror-y | 2.521mm | 10.384mm | 9.798mm | `024`, 3.274mm |

All anchored scaffold diagnostic splits:

| split | scaffold p95 mean | target-to-scaffold p95 mean | boundary-to-scaffold p95 mean | scaffold within 5mm |
|---|---:|---:|---:|---:|
| train | 2.363mm | 9.406mm | 8.663mm | 1.000 |
| val | 2.521mm | 10.384mm | 9.798mm | 0.999 |
| all | 2.403mm | 9.650mm | 8.947mm | 1.000 |

Interpretation:

- The previous free learned scaffold was the dominant generalization failure. Anchoring scaffold
  candidates to encoder centers almost eliminates the coarse false-positive mode.
- The model is now much closer to the sampling floor: validation Chamfer/floor dropped from 9.717x to
  2.733x.
- The remaining problem is no longer coarse placement. It is local density, point allocation, and
  topology: target-to-scaffold and target-to-reconstruction p95 are still higher than recon-to-target
  p95, especially on broad or boundary-rich regions.

Updated recommendation:

1. Treat `--scaffold-mode anchored --scaffold-anchor-source coarse_fine` as the default scaffold path
   for the current AE line.
2. Add evaluation outputs for `scaffold_anchor_points` and residual vectors so visual diagnostics can
   distinguish anchor selection failure from residual/refinement failure.
3. Replace fixed `points_per_scaffold=4` with learned or rule-based local point allocation per anchor.
4. Start moving from point reconstruction to CAE Mesh IR topology: scaffold nodes, boundary chains,
   local connectivity, and quality-gated shell element export.
