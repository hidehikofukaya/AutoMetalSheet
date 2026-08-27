# fable5向け進捗メモ: Anchored Scaffold AEから拘束点条件付き点群生成へ

Date: 2026-07-02
Author: Codex
Project: `C:\Users\hide2\IdeaBox\AutoMetalSheet`

## 0. 今回fableに聞きたいこと

当面の目標は、**中立面全体を入力せず、拘束点のみを条件として板金中立面点群を生成すること**です。

ここまでの実験では、中立面点群を入力するautoencoderとして、

1. 自由なlearned scaffold queryは検証部品で粗配置が外れやすい
2. encoder由来の中立面代表点にscaffoldをanchorすると、未学習validation部品の復元が大幅改善する

ことを確認しました。

ただし、anchored scaffoldは現状 **入力された中立面からanchorを選んでいる** ため、拘束点のみ生成へそのまま使えるわけではありません。fableには、次段階として **拘束点からどのようにanchor/scaffold priorを生成すべきか**、また現在の成果をどう解釈すべきかを診断してほしいです。

## 1. 現在の目的と到達点

### 最終寄りの目標

- 入力: 締結点、取り付け点、支持点、荷重点などの拘束点群
- 任意メタデータ: thickness/material/load/support/part class/size envelope
- 出力: CAEに使える板金中立面 shell mesh
- 短期MVP出力: まずは中立面点群、次に局所connectivity付きmesh IR

### 現在の実験段階

現在はまだ **中立面点群入力のautoencoder** です。

つまり、今のモデルが直接解いている問題は:

```text
filled midsurface point cloud -> reconstructed midsurface point cloud
```

であり、まだ:

```text
constraint points only -> generated midsurface point cloud
```

ではありません。

それでも今回の成果は重要です。板金の中立面点群生成では、点をbbox空間全体から自由生成するよりも、**粗い構造足場(scaffold)を中立面近傍に置く帰納バイアス** が非常に効くことが確認できたためです。

## 2. 実装済みの主要改善

### 2.1 fable前回診断を受けたP0修正

前回のfable診断で指摘された2点を修正済みです。

1. 評価の512点sampling floor問題
   - `evaluate_autoencoder.py` に `--metric-target-n-points` を追加
   - reconは512点のまま、metric targetだけ4096点化
   - `sampling_floor_chamfer_mm`
   - `chamfer_to_sampling_floor`
   - `chamfer_bbox_diag_pct`
   を評価出力に追加

2. train samplingが全epoch固定だった問題
   - `MidsurfacePointCloudDataset` に `resample_each_epoch` と `resample_step` を追加
   - `train_autoencoder.py --resample-train-each-epoch`
   - validationは固定samplingのまま
   - `--preload-data` との併用は禁止

関連コード:

- `cae_mesh_generator/src/cae_mesh_generator/data/fill_volume_dataset.py`
- `cae_mesh_generator/src/cae_mesh_generator/train_autoencoder.py`
- `cae_mesh_generator/src/cae_mesh_generator/evaluate_autoencoder.py`

### 2.2 mirror augmentation

train splitにのみmirror augmentationを入れました。

```powershell
--train-mirror-axes y
```

validationにはmirrorを入れていないため、validation leakageはありません。

axis ablationでは、同一q20-60 n24 seed13条件で:

| condition | val Chamfer | Chamfer/floor | target p95 | boundary p95 | worst |
|---|---:|---:|---:|---:|---|
| no mirror + resample | 31.231mm | 14.316x | 24.623mm | 25.099mm | `081`, 52.089mm |
| mirror-x + resample | 24.323mm | 11.424x | 19.896mm | 23.059mm | `081`, 42.251mm |
| mirror-y + resample | 21.724mm | 9.717x | 17.272mm | 18.288mm | `1285:10`, 27.707mm |
| mirror-z + resample | 23.381mm | 10.874x | 18.244mm | 20.291mm | `081`, 40.103mm |

mirror-yが最良でした。

### 2.3 scaffold placement diagnostic

自由なlearned scaffoldが本当に外れているかを直接見る診断スクリプトを追加しました。

```powershell
python -m cae_mesh_generator.diagnose_scaffold_placement `
  --checkpoint <checkpoint.pt> `
  --output-dir <diagnostic_dir> `
  --split all `
  --max-parts 0 `
  --metric-target-n-points 4096 `
  --device cuda
```

追加ファイル:

- `cae_mesh_generator/src/cae_mesh_generator/diagnose_scaffold_placement.py`

現行bestだった learned scaffold + resample + mirror-y の診断:

- checkpoint: `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13/best.pt`
- diagnostic: `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13_scaffold_diag4096`

結果:

| split | scaffold p95 mean | target-to-scaffold p95 mean | boundary-to-scaffold p95 mean |
|---|---:|---:|---:|
| train | 33.451mm | 22.813mm | 25.376mm |
| val | 70.497mm | 36.248mm | 35.734mm |
| all | 42.713mm | 26.172mm | 27.965mm |

worst scaffold p95:

```text
A0072600002_AllCATPart:081 = 104.118mm
```

この診断で、自由scaffold queryがvalidation部品で中立面から大きく外れることが明確になりました。

## 3. Anchored Scaffold Decoder

### 3.1 変更理由

従来のstructured AEでは、scaffold点はlearned queryから直接生成していました。

```text
latent/coarse/boundary memory -> learned scaffold query -> scaffold xyz
```

これは自由度が高く、validation部品でscaffoldが中立面から大きく外れました。

そこで、scaffoldをゼロから生成せず、encoderが実際に見た中立面代表点をanchorにして、そこから小さなresidualだけを予測する構造を追加しました。

### 3.2 現在の処理

```text
input midsurface point cloud
  -> encoder
     -> coarse centers
     -> fine patch centers
  -> FPS over coarse/fine centers
  -> anchor_points
  -> decoder predicts bounded residual
  -> scaffold = clamp(anchor + residual, -0.75, 0.75)
  -> local refinement decoder
  -> reconstructed point cloud
```

CLI:

```powershell
--scaffold-mode anchored `
--scaffold-anchor-source coarse_fine `
--scaffold-anchor-residual-scale 0.08
```

defaultは旧挙動維持です。

```powershell
--scaffold-mode learned
```

### 3.3 数式イメージ

入力点群を `X`、encoder tokenを `H`、coarse/fine中心候補を `C` とします。

1. anchor選択:

```text
A = FPS(C, n_scaffold)
```

2. anchor tokenをmemoryへcross-attention:

```text
T = CrossAttention(anchor_token(A), H)
```

3. residual予測:

```text
R = tanh(MLP(T)) * residual_scale
```

4. scaffold:

```text
S = clamp(A + R, -0.75, 0.75)
```

このため、scaffoldは基本的に入力中立面近傍から大きく飛びません。

### 3.4 実装箇所

- `cae_mesh_generator/src/cae_mesh_generator/model/hierarchical_ae.py`
  - `AnchoredScaffoldDecoder`
  - `StructuredScaffoldAutoencoder.select_scaffold_anchors`
  - `StructuredScaffoldAutoencoder.forward`

- `cae_mesh_generator/src/cae_mesh_generator/train_autoencoder.py`
  - CLI追加
  - checkpoint config保存
  - resume時のmodel args復元

- `cae_mesh_generator/src/cae_mesh_generator/evaluate_autoencoder.py`
  - anchored checkpointのload対応

- `cae_mesh_generator/tests/test_hierarchical_ae.py`
  - anchored output shape
  - bounded residual
  - candidate不足時repeat分岐
  - evaluate loaderのstrict restore

テスト:

```text
43 passed
```

## 4. Formal Run結果

### 4.1 run条件

run:

- `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13`

条件:

- q20-60 bbox diagonal
- max_parts=24
- train/val = 18/6
- split_seed=13
- `--resample-train-each-epoch`
- `--train-mirror-axes y`
- `--boundary-sample-fraction 0.25`
- `--use-boundary-feature`
- `--scaffold-mode anchored`
- `--scaffold-anchor-source coarse_fine`
- `--scaffold-anchor-residual-scale 0.08`
- epochs=300
- best metric: `val_cae_score`

best checkpoint:

```text
best.pt = epoch 290
```

training-time validation metrics at epoch 290:

| metric | value |
|---|---:|
| val CAE score | 4.441 |
| val target p95 | 9.002mm |
| val boundary p95 | 8.231mm |
| val target within 5mm | 0.653 |
| val boundary within 5mm | 0.745 |

### 4.2 4096-target evaluation

評価dir:

- `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13_eval_all24_best_calibrated4096`

learned scaffold current bestとの比較:

| model | val Chamfer | Chamfer/floor | val target p95 | val boundary p95 | val target within 5mm | val boundary within 5mm | worst val Chamfer |
|---|---:|---:|---:|---:|---:|---:|---|
| learned scaffold + resample + mirror-y | 21.724mm | 9.717x | 17.272mm | 18.288mm | 0.217 | 0.216 | `A0072601285_AllCATPart:10`, 27.707mm |
| anchored scaffold + resample + mirror-y | 6.317mm | 2.733x | 9.416mm | 8.565mm | 0.548 | 0.687 | `A0072600002_AllCATPart:024`, 7.872mm |

all splits:

| split | Chamfer | Chamfer/floor | target p95 | boundary p95 | target within 5mm | boundary within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| train | 5.639mm | 2.676x | 8.502mm | 7.841mm | 0.604 | 0.741 |
| val | 6.317mm | 2.733x | 9.416mm | 8.565mm | 0.548 | 0.687 |
| all | 5.808mm | 2.690x | 8.730mm | 8.022mm | 0.590 | 0.727 |

旧worstだった `081`:

| part | split | Chamfer | Chamfer/floor | recon p95 | target p95 | target within 5mm | boundary p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0072600002_AllCATPart:081` | val | 4.740mm | 2.72x | 2.615mm | 6.857mm | 0.766 | 7.063mm |

現在のworst validation:

| part | split | Chamfer | Chamfer/floor | recon p95 | target p95 | target within 5mm | boundary p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `A0072600002_AllCATPart:024` | val | 7.872mm | 2.69x | 3.358mm | 11.754mm | 0.380 | 10.893mm |

### 4.3 anchored scaffold診断

診断dir:

- `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13_scaffold_diag4096`

learned scaffoldとの直接比較:

| model | val scaffold p95 mean | val target-to-scaffold p95 mean | val boundary-to-scaffold p95 mean | worst scaffold p95 |
|---|---:|---:|---:|---:|
| learned scaffold + resample + mirror-y | 70.497mm | 36.248mm | 35.734mm | `081`, 104.118mm |
| anchored scaffold + resample + mirror-y | 2.521mm | 10.384mm | 9.798mm | `024`, 3.274mm |

anchored scaffold diagnostic:

| split | scaffold p95 mean | target-to-scaffold p95 mean | boundary-to-scaffold p95 mean | scaffold within 5mm |
|---|---:|---:|---:|---:|
| train | 2.363mm | 9.406mm | 8.663mm | 1.000 |
| val | 2.521mm | 10.384mm | 9.798mm | 0.999 |
| all | 2.403mm | 9.650mm | 8.947mm | 1.000 |

## 5. 解釈

### 5.1 何が解けたか

今回ほぼ解けたのは、**中立面点群が入力にある場合の粗scaffold配置** です。

自由learned scaffoldでは、validation部品でscaffoldが中立面から大きく外れていました。anchored scaffoldでは、scaffold p95がval平均で70.497mmから2.521mmまで下がりました。

これは、板金中立面生成では「bbox空間全体から点を直接生成する」よりも、**薄肉中立面近傍に粗足場を置く帰納バイアスが極めて有効** であることを示しています。

### 5.2 何がまだ解けていないか

現在のanchored scaffoldは、入力中立面点群からanchorを選びます。

したがって、現在の結果は:

```text
unseen midsurface point cloud -> reconstruction
```

の汎化であり、

```text
constraint points only -> generation
```

の汎化ではありません。

拘束点のみ生成では、中立面全体のcoarse/fine centersが存在しないため、現在のanchor選択:

```text
A = FPS(coarse/fine centers)
```

をそのまま使えません。

### 5.3 ただし次段階への示唆

今回の結果から、拘束点のみ生成でも「完全自由query」ではなく、何らかの **constraint-conditioned anchor/scaffold prior** が必要そうです。

候補:

1. 拘束点をanchor seedとして使う
2. 拘束点間に仮想rib/flange/bridge scaffoldを生成する
3. bbox/envelope/part classからglobal scaffoldを生成する
4. learned priorでanchor candidatesを生成し、そこからbounded residualを出す
5. 生成されたanchor/scaffoldから局所点密度とboundary chainを展開する

つまり、次のモデルは:

```text
constraint points + metadata
  -> constraint encoder
  -> scaffold prior generator
  -> anchor candidates
  -> residual refinement
  -> point cloud / mesh IR
```

のようにすべきだと考えています。

## 6. 拘束点のみ生成へ向けた設計課題

### 6.1 拘束点から何を生成すべきか

拘束点のみから直接全点群を出すより、まずは以下を生成するほうがよさそうです。

1. coarse scaffold nodes
2. boundary chains
3. local patch allocation
4. point density / refinement logits
5. optional topology edges

中立面点群は最終出力ではなく、CAE mesh IRへ向かう途中表現として扱うのが自然です。

### 6.2 条件情報として必要そうなもの

拘束点だけでは部品の外形自由度が大きすぎる可能性があります。追加条件候補:

- part bbox or design envelope
- thickness
- material
- load/support metadata
- fastener/mount type
- symmetry/mirror flags
- manufacturing constraints
- expected flange/rib/emboss class
- assembly/component family id

fableには、拘束点のみで十分か、どのmetadataが不可欠かを評価してほしいです。

### 6.3 生成多様性と評価

拘束点条件付き生成では、一つの拘束点配置から複数の妥当形状があり得ます。

そのためautoencoder評価のChamferだけでは不十分です。

必要そうな評価:

- constraint satisfaction
- generated surface coverage
- boundary plausibility
- thin-sheet manufacturability
- minimum bend radius / gradient
- local stiffness proxy
- node/element quality
- solver import smoke
- diversity under same constraints
- train family vs unseen family split

## 7. fableへの具体質問

1. 今回のanchored scaffold結果を、拘束点のみ生成へ向けた帰納バイアスの証拠としてどう評価しますか？
2. 中立面入力がない場合、anchor候補を何から作るべきだと思いますか？
3. 拘束点のみでは不良設定ですか？最低限必要なmetadataは何でしょうか？
4. 次のMVPは、点群を直接生成するべきですか、それともscaffold/boundary chain/mesh IRを先に生成するべきですか？
5. 現在のAE checkpointを使って、constraint-conditioned generatorの教師信号を作る方法はありますか？
6. 多様性を失わずに、板金らしい形状制約を入れるにはどの構造がよいでしょうか？
7. CAE用途として、点群生成段階で必ず入れるべき評価指標は何でしょうか？

## 8. 参照パス

主要実装:

- `cae_mesh_generator/src/cae_mesh_generator/model/hierarchical_ae.py`
- `cae_mesh_generator/src/cae_mesh_generator/train_autoencoder.py`
- `cae_mesh_generator/src/cae_mesh_generator/evaluate_autoencoder.py`
- `cae_mesh_generator/src/cae_mesh_generator/diagnose_scaffold_placement.py`
- `cae_mesh_generator/tests/test_hierarchical_ae.py`

主要レポート:

- `docs/cae_structured_q20_60_cov_scaf_seed13_report.md`
- `docs/fable5_diagnostic_response.md`
- `docs/fable5_local_diagnostic_report.md`

主要runs:

- learned scaffold current best:
  - `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13`
  - `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13_eval_all24_best_calibrated4096`
  - `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_seed13_scaffold_diag4096`

- anchored scaffold:
  - `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13`
  - `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13_eval_all24_best_calibrated4096`
  - `runs/cae_mesh_structured_q20_60_n24_resample_mirror_y_anchored_seed13_scaffold_diag4096`

## 9. 一文要約

中立面入力AEでは、自由queryでscaffoldを生成するより、入力中立面上のcoarse/fine centersにanchorしてbounded residualを出す構造が圧倒的に安定した。ただしこれは拘束点のみ生成の解ではなく、次は拘束点とmetadataからanchor/scaffold prior自体を生成するモデルへ移行する必要がある。
