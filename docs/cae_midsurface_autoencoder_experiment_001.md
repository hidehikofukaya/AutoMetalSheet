# 中立面オートエンコーダ実験 001

## 目的

板金のfilled midsurface STEPを点群化し、部品の大域構造と局所形状を潜在表現に落として復元できるかを最小構成で確認する。

この実験はCAEメッシュ生成そのものではなく、次段の「粗メッシュ + 局所細分化デコーダ」に進む前の構造理解チェックである。成功条件は、まず単一部品の固定サンプル点群を過学習できること、次に少数部品で破綻傾向を観察できることとした。

## 実装

実装場所:

- `cae_mesh_generator/src/cae_mesh_generator/data/step_tessellate.py`
- `cae_mesh_generator/src/cae_mesh_generator/data/fill_volume_dataset.py`
- `cae_mesh_generator/src/cae_mesh_generator/model/hierarchical_ae.py`
- `cae_mesh_generator/src/cae_mesh_generator/train_autoencoder.py`

入力形状は `C:\Users\hide2\IdeaBox\fill_volume\fill_mid_surf\<assembly>\fill\*.stp` のfilled midsurface STEPである。OpenCASCADEで三角形へテッセレーションし、面積重み付きで中立面点群をサンプリングする。

デフォルトの各点特徴量:

- 正規化座標 `x, y, z`
- 法線 `nx, ny, nz`
- 定数チャネル `d_joint=1`

`--use-joint-distance` を指定した場合のみ、`annotations/joints.json` から近傍拘束点距離を読み、`d_joint` に入れる。今回の正式なshape-only確認ではこのフラグを使っていない。

部品選定では、最小部品だけを選ぶと形状が単純すぎる可能性がある。そのため追加実験では、テッセレーション後のbbox対角長をサイズ指標とし、下位20〜40%の部品を選ぶ。`--max-parts` を併用した場合は、そのサイズ帯から均等にサンプリングする。

モデル構成:

- Coarse global path: farthest point samplingで粗い代表点を取り、Transformer encoderで大域構造を記憶する。
- Fine local path: farthest point sampling中心 + k近傍パッチをPointNet型MLPで局所トークン化する。
- Latent fusion: learnable latent queryがcoarse/fine tokenへcross-attentionする。
- Decoder: learnable output queryがlatentへcross-attentionし、unordered point cloudと法線を出力する。

損失:

- Chamfer distance
- 最近傍点の法線との整合損失
- bbox統計を安定させる弱いspread loss

## 実行環境メモ

この環境ではPyTorch/OCC利用時にOpenMP runtime重複が出るため、テストと学習の両方で以下を設定する。

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:PYTHONPATH='C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator\src'
```

## 再現コマンド

テスト:

```powershell
python -m pytest cae_mesh_generator\tests -q
```

shape-onlyスモーク学習:

```powershell
python -m cae_mesh_generator.train_autoencoder `
  --max-parts 1 `
  --max-file-mb 0.25 `
  --n-points 256 `
  --epochs 20 `
  --batch-size 1 `
  --token-dim 64 `
  --n-coarse 64 `
  --n-patches 32 `
  --k-neighbors 16 `
  --n-latents 32 `
  --output-dir runs\cae_mesh_ae_shape_only_smoke `
  --device cpu `
  --lr 0.001 `
  --log-every 10
```

サイズ下位20〜40%を狙うスモーク学習:

```powershell
python -m cae_mesh_generator.train_autoencoder `
  --max-file-mb 0.25 `
  --size-quantile-min 0.2 `
  --size-quantile-max 0.4 `
  --max-parts 3 `
  --n-points 128 `
  --epochs 1 `
  --batch-size 1 `
  --token-dim 32 `
  --n-coarse 32 `
  --n-patches 16 `
  --k-neighbors 8 `
  --n-latents 16 `
  --output-dir runs\cae_mesh_ae_q20_40_smoke `
  --device cpu `
  --lr 0.001 `
  --log-every 1
```

shape-only単一部品の過学習確認:

```powershell
python -m cae_mesh_generator.train_autoencoder `
  --max-parts 1 `
  --max-file-mb 0.25 `
  --n-points 256 `
  --epochs 150 `
  --batch-size 1 `
  --token-dim 64 `
  --n-coarse 64 `
  --n-patches 32 `
  --k-neighbors 16 `
  --n-latents 32 `
  --output-dir runs\cae_mesh_ae_shape_only_overfit1 `
  --device cpu `
  --lr 0.001 `
  --log-every 25
```

視覚評価:

```powershell
python -m cae_mesh_generator.evaluate_autoencoder `
  --checkpoint runs\cae_mesh_ae_shape_only_overfit1\best.pt `
  --output-dir runs\cae_mesh_ae_shape_only_overfit1\visual_eval `
  --max-parts 1 `
  --device cpu
```

評価スクリプトは、各部品ごとに以下を出力する。

- `comparison.png`: target、reconstruction、overlay、recon→target誤差、target→recon取りこぼし、距離ヒストグラム
- `projection_overlay.png`: XY/XZ/YZ直交投影でのtarget/reconstruction重ね合わせ
- `comparison.html`: Plotlyの回転可能な3D比較
- `target.ply`, `recon.ply`, `overlay_target_recon.ply`
- `recon_error.ply`, `target_miss_error.ply`
- `metrics.json`, `metrics.csv`
- `eval_manifest.json`: checkpoint引数、選択部品、STEP fingerprint照合結果

## 結果

ユニットテスト:

- `11 passed`

shape-only 20epochスモーク:

- epoch 1: loss 0.875300, Chamfer 0.834961, normal 0.070246
- epoch 10: loss 0.556921, Chamfer 0.510676, normal 0.340588
- epoch 20: loss 0.480584, Chamfer 0.443948, normal 0.173558

サイズ下位20〜40%スモーク:

- 候補: `max_file_mb=0.25` の17部品
- bbox対角長: min 71.380, p20 86.935, p40 114.835, max 227.910
- 選択部品: `A0072600002_AllCATPart:007`, `A0072600002_AllCATPart:011`, `A0072601285_AllCATPart:20`
- epoch 1: loss 0.779475, Chamfer 0.731875, normal 0.390077

shape-only単一部品150epoch:

- epoch 1: loss 0.875300, Chamfer 0.834961, normal 0.070246
- epoch 25: loss 0.519879, Chamfer 0.472524, normal 0.424399
- epoch 50: loss 0.512673, Chamfer 0.472232, normal 0.115578
- epoch 75: loss 0.423498, Chamfer 0.388675, normal 0.011686
- epoch 100: loss 0.238692, Chamfer 0.207953, normal 0.049267
- epoch 125: loss 0.153636, Chamfer 0.127278, normal 0.030977
- epoch 150: loss 0.098368, Chamfer 0.079311, normal 0.035721

shape-only単一部品の視覚評価:

- 評価対象: `runs/cae_mesh_ae_shape_only_overfit1/best.pt`
- Chamfer: 5.0923 mm
- recon→target p95: 3.7965 mm
- target→recon p95: 6.1374 mm
- recon within 5mm: 98.8%
- target within 5mm: 86.7%

この結果では、生成点自体は正解面の近傍に寄っている一方で、正解側の一部領域を取りこぼしている。直交投影では、底面側に点が寄り、立ち上がり側のcoverageが弱い傾向が見える。

注意: 古いcheckpointにはSTEP fingerprintが入っていないため、評価時targetは現在のSTEPから再テッセレーション・再サンプリングされる。新しいcheckpointでは `record_fingerprints` を保存し、評価時に現在のSTEPと照合して `eval_manifest.json` に結果を残す。また、点群サンプリングseedは部品ID由来に固定し、train/val分割後も同一部品は同じ固定サンプル点群になるようにした。

## 複数部品holdout診断

目的は、汎化性能を証明することではなく、複数部品でどこが壊れるかを観察することである。

条件:

- `max_file_mb=1.0`
- bbox対角長の下位20〜40%
- 選択部品: 9部品
- train: 7部品
- val: 2部品
- 点数: 256
- epoch: 120
- token/coarse/patch/latent: 96/96/48/48

実行:

```powershell
python -m cae_mesh_generator.train_autoencoder `
  --max-file-mb 1.0 `
  --size-quantile-min 0.2 `
  --size-quantile-max 0.4 `
  --max-parts 10 `
  --val-count 2 `
  --split-strategy even `
  --n-points 256 `
  --epochs 120 `
  --batch-size 1 `
  --token-dim 96 `
  --n-coarse 96 `
  --n-patches 48 `
  --k-neighbors 24 `
  --n-latents 48 `
  --output-dir runs\cae_mesh_ae_q20_40_holdout_stable_v2 `
  --device cpu `
  --lr 0.001 `
  --log-every 20
```

評価:

```powershell
python -m cae_mesh_generator.evaluate_autoencoder `
  --checkpoint runs\cae_mesh_ae_q20_40_holdout_stable_v2\best.pt `
  --output-dir runs\cae_mesh_ae_q20_40_holdout_stable_v2\visual_eval `
  --max-parts 0 `
  --device cpu
```

集約結果:

| split | count | Chamfer mean | recon p95 mean | target p95 mean | target within 5mm |
|---|---:|---:|---:|---:|---:|
| train | 7 | 12.110mm | 10.612mm | 13.963mm | 30.3% |
| val | 2 | 42.118mm | 37.270mm | 38.769mm | 4.7% |
| all | 9 | 18.779mm | 16.536mm | 19.476mm | 24.6% |

`all` は評価CLIで選択された評価対象のmacro averageである。今回のrunは `--max-parts 0` によりcheckpoint内の9部品すべてを評価している。

worst:

- val worst: `A0072601285_AllCATPart:21`, Chamfer 47.004mm, target p95 40.575mm
- train worst by Chamfer: `A0072601285_AllCATPart:31`, Chamfer 15.225mm
- train worst by target p95: `A0072600002_AllCATPart:018`, target p95 16.347mm

val worstでは、復元点が正解の縦リブ状/折れ線状構造に乗らず、別の塊として散る。これは単なる局所ノイズではなく、未知形状に対する構造的な崩れと見るべきである。

## 複数部品train-all過学習診断

holdoutの悪化が、モデル容量不足なのか、未知部品への汎化不足なのかを見るため、同じ9部品を全てtrainに入れて過学習させた。

条件:

- 選択部品: holdoutと同じ9部品
- train: 9部品
- val: 0部品
- 点数: 256
- epoch: 200
- token/coarse/patch/latent: 128/128/64/64

集約結果:

| split | count | Chamfer mean | recon p95 mean | target p95 mean | target within 5mm |
|---|---:|---:|---:|---:|---:|
| train | 9 | 17.064mm | 17.651mm | 20.834mm | 15.6% |

holdoutで大きく崩れたval部品は、train-allでは改善した。`A0072601285_AllCATPart:30` は Chamfer 37.232mm から 12.195mm、`A0072601285_AllCATPart:21` は 47.004mm から 16.431mm へ改善している。したがって、現在のAEは完全に表現不能というより、未知部品への補間・汎化が弱い。

このtrain-all比較はholdout runと同じ120epoch/token 96系で行った。ただし、全体平均ではholdout trainより悪い部品もあり、固定query点群デコーダの多部品過学習はまだ安定していない。改善分を単純に「trainに入れれば覚えられる」とは読まない。

train-allでも target within 5mm は平均15.6%であり、CAEメッシュ生成の前段としてはまだ不十分である。視覚的にも点が面全体を均一に覆うというより、特徴線や薄板面の一部を取りこぼし、散布点として崩れる傾向が残る。

出力:

- `runs/cae_mesh_ae_q20_40_trainall_matched_v2/best.pt`
- `runs/cae_mesh_ae_q20_40_trainall_matched_v2/history.json`
- `runs/cae_mesh_ae_q20_40_trainall_matched_v2/visual_eval/aggregate_metrics.csv`
- `runs/cae_mesh_ae_q20_40_trainall_matched_v2/visual_eval/index.html`

## 構造化scaffold decoder実装

固定query点群デコーダのholdout崩れを受け、同じdual encoderを使いながら、出力側だけを以下の2段構成へ変更した。

1. `ScaffoldDecoder`: latent tokenとcoarse tokenへcross-attentionし、部品全体の粗いscaffold点を復元する。
2. `LocalRefinementDecoder`: 各scaffold点から近いfine tokenをhard routingで集め、その局所memoryへattentionして複数のrefined point、法線、refinement logitを出す。

実装場所:

- `cae_mesh_generator/src/cae_mesh_generator/model/hierarchical_ae.py`
- `cae_mesh_generator/src/cae_mesh_generator/train_autoencoder.py`
- `cae_mesh_generator/src/cae_mesh_generator/evaluate_autoencoder.py`
- `cae_mesh_generator/tests/test_hierarchical_ae.py`

CLIでは `--model-kind structured` を指定する。主な追加パラメータは以下である。

- `--n-scaffold`: 粗構造点の数
- `--points-per-scaffold`: 1 scaffold点から生成する局所点数。未指定の場合は `--n-points` を覆うよう自動計算する。
- `--n-local-tokens`: 各scaffold点が参照する近傍fine token数

視覚評価では、structured checkpointの場合に `scaffold.ply` を出力し、`projection_overlay.png` へ緑色のscaffold点を重ねる。
`--n-points` が `--n-scaffold * --points-per-scaffold` で割り切れない場合でも、復元点と `refinement_logits` は要求点数へ揃える。scaffold側は `active_scaffold_mask` と `scaffold_point_counts` を返し、可視化では有効scaffoldだけを使う。

注意: 現在の `refinement_logits` は将来の局所細分化優先度/保護領域のためのplaceholderであり、まだ専用lossや評価指標では教師していない。CAE Mesh IRへ進める段階で意味づけと教師信号を固定する。

## 構造化decoder q20-40 holdout診断

条件:

- 入力: shape-only filled midsurface point cloud
- 選択部品: bbox対角長の下位20〜40%、9部品
- train: 7部品
- val: 2部品
- 点数: 256
- epoch: 80
- token/coarse/patch/latent: 96/96/48/48
- scaffold/local: 64 scaffold, 4 points per scaffold, 8 local tokens

実行:

```powershell
python -m cae_mesh_generator.train_autoencoder `
  --model-kind structured `
  --max-file-mb 1.0 `
  --size-quantile-min 0.2 `
  --size-quantile-max 0.4 `
  --max-parts 10 `
  --val-count 2 `
  --split-strategy even `
  --split-seed 13 `
  --n-points 256 `
  --epochs 80 `
  --batch-size 1 `
  --token-dim 96 `
  --n-coarse 96 `
  --n-patches 48 `
  --k-neighbors 24 `
  --n-latents 48 `
  --n-scaffold 64 `
  --points-per-scaffold 4 `
  --n-local-tokens 8 `
  --output-dir runs\cae_mesh_structured_q20_40_holdout_v1 `
  --device cpu `
  --lr 0.001 `
  --log-every 20
```

評価:

```powershell
python -m cae_mesh_generator.evaluate_autoencoder `
  --checkpoint runs\cae_mesh_structured_q20_40_holdout_v1\best.pt `
  --output-dir runs\cae_mesh_structured_q20_40_holdout_v1\visual_eval `
  --max-parts 0 `
  --device cpu
```

学習ログ:

- epoch 1: loss 0.650426, Chamfer 0.605497
- epoch 20: Chamfer 0.207144
- epoch 40: Chamfer 0.167293
- epoch 60: Chamfer 0.128697
- epoch 80: loss 0.110569, Chamfer 0.102566

集約結果:

| split | count | Chamfer mean | recon p95 mean | target p95 mean | target within 5mm | recon within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| train | 7 | 10.590mm | 9.105mm | 11.643mm | 51.1% | 56.8% |
| val | 2 | 30.687mm | 28.386mm | 34.294mm | 13.3% | 11.3% |
| all | 9 | 15.056mm | 13.390mm | 16.677mm | 42.7% | 46.7% |

固定query点群decoderのholdout stable v2と比べると、val Chamfer meanは 42.118mm から 30.687mm へ改善し、val target within 5mmは 4.7% から 13.3% へ改善した。80epochでこの改善が出ているため、粗scaffoldを介してから局所refinementする方向は、固定query点群decoderより有望である。

ただし、CAE-readyにはまだ遠い。val worstの `A0072601285_AllCATPart:21` は Chamfer 34.104mm、target p95 39.355mm であり、特徴線や縦方向構造へのcoverage不足が残る。視覚的にもscaffoldは正解構造へ寄り始めているが、refined pointが薄板面全体や端部を安定して覆う段階ではない。

現時点の読み:

- dual encoder自体は使い続ける価値がある。
- decoderは固定queryより、coarse scaffold + local refinementの方が構造を壊しにくい。
- hard nearest-token routingは実装が単純で速いが、境界・リブ・フランジ端の明示的な割当やtopologyはまだ学んでいない。
- Chamfer改善はCAEメッシュ品質の保証ではない。次はnode/element topology、境界保持、局所サイズ場、品質projectionへ進める必要がある。

出力:

- `runs/cae_mesh_structured_q20_40_holdout_v1/best.pt`
- `runs/cae_mesh_structured_q20_40_holdout_v1/history.json`
- `runs/cae_mesh_structured_q20_40_holdout_v1/visual_eval/aggregate_metrics.csv`
- `runs/cae_mesh_structured_q20_40_holdout_v1/visual_eval/index.html`

## 拡大検証に向けた学習スクリプト更新

検証部品数とepoch数を増やす場合、train lossだけでbest checkpointを選ぶと過学習を見落としやすい。そのため `train_autoencoder.py` に学習中validationを追加した。

追加内容:

- validation splitがある場合、各評価タイミングで `val_loss`, `val_chamfer`, `val_normal`, `val_spread` を `history.json` に記録する。
- `--best-metric auto` は、validation splitがあれば `val_loss`、なければ `train_loss` を使う。
- `--eval-every` でvalidationの間隔を指定できる。epoch 1と最終epochでは必ず評価する。
- checkpointに `best: {epoch, metric, value}` を保存する。

スモーク確認:

```powershell
python -m cae_mesh_generator.train_autoencoder `
  --model-kind structured `
  --max-file-mb 0.25 `
  --size-quantile-min 0.2 `
  --size-quantile-max 0.4 `
  --max-parts 3 `
  --val-count 1 `
  --n-points 64 `
  --epochs 2 `
  --token-dim 32 `
  --n-coarse 24 `
  --n-patches 16 `
  --k-neighbors 8 `
  --n-latents 12 `
  --n-scaffold 16 `
  --points-per-scaffold 4 `
  --n-local-tokens 4 `
  --output-dir runs\cae_mesh_structured_val_smoke `
  --device cpu `
  --lr 0.001 `
  --log-every 1
```

結果:

- epoch 1: train_loss 0.780599, val_loss 0.847058
- epoch 2: train_loss 0.651106, val_loss 0.688659
- best: epoch 2, metric `val_loss`, value 0.688659
- unit test: `12 passed`

次の本検証は、`max_file_mb=5.0`, bbox対角長 `q20-60`, `max_parts=24`, `val_fraction=0.25`, `epochs=300` 程度から始める。split seedは最低3種類回し、単一splitの偶然を避ける。

2026-06-28時点のローカル環境ではCUDAが使用可能である。

- PyTorch: `2.6.0+cu124`
- GPU: NVIDIA GeForce GTX 1660 SUPER 6GB
- CUDA smoke training: success
- `last.pt` からのresume: success

1000 epochで回す場合は、`--device cuda --epochs 1000 --eval-every 10 --save-every 25` を使う。1000 epochでは途中停止のリスクが高いため、`best.pt` だけでなく `last.pt` を必ず残す。

## 解釈

shape-onlyの単一部品では復元誤差が明確に下がったため、この構成は中立面STEPからサンプリングした固定点群を潜在表現に記憶して復元する最低限の経路を持つ。

一方で、これは「中立面そのものを連続面として復元できた」証拠ではない。固定seedでサンプリングした点群に対する過学習であり、境界線、穴、フランジ端、拘束点近傍のCAE上重要な誤差はまだ測っていない。

固定queryからunordered point cloudを直接吐くデコーダは、複数部品の構造差が出始めると平均化する可能性がある。ただし今回の実験だけでは、容量不足、epoch不足、サンプリング条件、テッセレーション条件でも説明できるため、断定はしない。

## 現時点の限界

- 出力は点群であり、CAEに使える節点・要素・集合ではない。
- ChamferだけではCAE品質を測れない。
- 法線は最近傍法線で弱く教師しているが、連続面の向きや要素法線品質を保証しない。
- 境界線、穴、フランジ端、拘束点周辺の幾何誤差を個別に評価していない。
- 単一部品の過学習は汎化性能の証拠ではない。
- 複数部品holdoutではvalが大きく崩れる。現在の固定query点群デコーダは、未知部品の構造補間に弱い。
- 構造化decoderでも、scaffold topology、境界接続、要素品質、局所サイズ場はまだ出力していない。
- STEPのテッセレーション密度により学習対象が変わるため、メッシュ品質目標とはまだ切り離されている。

## 次の実装方針

1. 点群AEを評価器として残しつつ、structured scaffold decoderをCAE mesh IRへ拡張する。
2. scaffold点を節点候補だけでなく、edge/face候補、局所サイズ場、boundary/protected-region logitへ拡張する。
3. 学習・検証splitは、最小部品ではなくbbox対角長の下位20〜40%を主対象にする。
4. 視覚評価では `recon->target` と `target->recon` を分け、ゴースト生成と取りこぼしを別々に確認する。
5. scaffold nodeごとに局所edge length、refinement priority、boundary/protected-region logitsを予測する。
6. filled midsurfaceへの投影と品質projectionをdeterministic post-processとして実装する。
7. target/reconstructionのPLYに加え、boundary distance、joint distance、normal consistency、surface coverageを評価に追加する。
8. CAE MVPでは最終出力を節点、三角/四角要素、node/element set、property、quality reportにする。

## 判定

「中立面形状データだけを入力した階層AEが、固定サンプル点群として部品を記憶し始めるか」という問いには Yes と言える。ただし「生成モデルとして汎化し、多様な板金部品をCAEメッシュとして出せるか」は未検証であり、このプロトタイプだけでは足りない。

次は、structured scaffold decoderをCAE Mesh IRへ拡張し、点群復元ではなく節点・要素・境界・品質projectionまで含む粗メッシュ + 局所細分化生成へ進める。
