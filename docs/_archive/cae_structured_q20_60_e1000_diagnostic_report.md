# Structured Scaffold AE q20-60 n24 Diagnostic Report

## この文書の位置づけ

この文書は、`docs/cae_structured_scaffold_autoencoder_architecture.md` の続編である。先の文書では、Structured Scaffold Autoencoderの目的、入出力、dual encoder、latent fusion、scaffold decoder、local refinement decoder、将来のCAE Mesh IR拡張を定義した。

本レポートでは、その同じアーキテクチャを実データ24部品へ拡大して学習し、以下を検証する。

1. epochを伸ばすことでvalidation性能が改善するか。
2. structured scaffold decoderは未知部品のcoverageを改善できるか。
3. 現在の点群AEからCAE Mesh IRへ進む前に、どの失敗モードを優先して潰すべきか。

記号と構成要素はアーキテクチャ文書と同じである。

| アーキ文書の記号 | 本レポートでの観察対象 |
|---|---|
| `P`, `F` | filled midsurface STEPから得た入力点群と特徴量 |
| `T_c`, `T_f` | coarse / fine encoder token |
| `Z` | latent fusion後の潜在token |
| `S` | 可視化上の緑点、coarse scaffold |
| `Y_hat` | 可視化上の橙点、復元点群 |
| target | 可視化上の青点、正解サンプル点群 |

したがって、このレポートの結論は「別モデルの評価」ではなく、前章で定義したstructured scaffold architectureの初回拡大診断である。

## 概要

対象run:

- `runs/cae_mesh_structured_q20_60_n24_e1000_seed13`
- structured scaffold + local refinement decoder
- bbox diagonal q20-60
- `max_file_mb=5.0`
- `max_parts=24`
- train / val = 18 / 6
- `n_points=512`
- `token_dim=128`
- `n_scaffold=128`
- `points_per_scaffold=4`
- `device=cuda`

学習はepoch 500まで到達した。epoch 500以降の `last.pt` 保存タイミングでWindows file lockにより停止したが、`best.pt` と `last.pt` は評価可能である。今後のため、checkpoint保存はatomic writeへ修正済み。

## 結論

モデルはepoch 100前後でvalidation性能がほぼ収束し、epoch 110で最良になった。その後はtrain lossが下がり続ける一方でvalidation lossは悪化しており、1000 epochまで回す価値は低い。

現状の主問題は、点が正解面の近傍へ寄ることではなく、未知部品の面全体・境界・細長い段付き構造を覆う能力が足りないことである。CAEメッシュ生成器としては、topology、boundary、coverage、projection、quality gateが未実装であることが支配的な弱点になっている。

## 学習履歴

| epoch | train loss | val loss | train chamfer | val chamfer |
|---:|---:|---:|---:|---:|
| 1 | 0.539640 | 0.321444 | 0.504020 | 0.295417 |
| 10 | 0.270806 | 0.272885 | 0.248470 | 0.246547 |
| 50 | 0.124191 | 0.189454 | 0.110469 | 0.168626 |
| 100 | 0.106619 | 0.147261 | 0.095900 | 0.128491 |
| 110 | 0.094380 | 0.139356 | 0.085364 | 0.124275 |
| 150 | 0.094457 | 0.139458 | 0.085767 | 0.123823 |
| 200 | 0.099045 | 0.147107 | 0.089021 | 0.129773 |
| 300 | 0.087344 | 0.149748 | 0.078258 | 0.134760 |
| 400 | 0.085571 | 0.154506 | 0.078010 | 0.138924 |
| 500 | 0.091529 | 0.154320 | 0.082637 | 0.139355 |

![loss curve](../runs/cae_mesh_structured_q20_60_n24_e1000_seed13/diagnostics/loss_curve.png)

読み取り:

- validation bestはepoch 110。
- epoch 110以降、train lossは下がるがvalidationは改善しない。
- early stopping patienceを入れるなら、patience 80〜120 epoch程度で十分。

## best epoch 110 評価

評価出力:

- `runs/cae_mesh_structured_q20_60_n24_e1000_seed13/visual_eval_best_epoch110`

| split | count | Chamfer mean | recon p95 mean | target p95 mean | target within 5mm | recon within 5mm |
|---|---:|---:|---:|---:|---:|---:|
| train | 18 | 14.299mm | 12.978mm | 18.110mm | 35.4% | 44.3% |
| val | 6 | 27.216mm | 31.178mm | 35.581mm | 16.0% | 18.4% |
| all | 24 | 17.528mm | 17.528mm | 22.478mm | 30.6% | 37.8% |

worst val:

| part | Chamfer | target p95 | target within 5mm |
|---|---:|---:|---:|
| `A0072600002_AllCATPart:081` | 37.402mm | 41.553mm | 11.5% |
| `A0072600002_AllCATPart:074` | 28.316mm | 40.833mm | 10.9% |
| `A0072600002_AllCATPart:024` | 28.307mm | 30.316mm | 11.3% |
| `A0072601285_AllCATPart:10` | 25.822mm | 38.151mm | 16.2% |

## epoch 500 との比較

| split/model | Chamfer mean | target p95 mean | target within 5mm | recon within 5mm |
|---|---:|---:|---:|---:|
| train best epoch110 | 14.299mm | 18.110mm | 35.4% | 44.3% |
| val best epoch110 | 27.216mm | 35.581mm | 16.0% | 18.4% |
| train last epoch500 | 13.938mm | 18.056mm | 40.0% | 51.0% |
| val last epoch500 | 30.262mm | 33.331mm | 12.9% | 13.2% |

![aggregate best vs last](../runs/cae_mesh_structured_q20_60_n24_e1000_seed13/diagnostics/aggregate_best_vs_last.png)

読み取り:

- epoch 500はtrain側のcoverageを改善するが、val側のcoverageを悪化させる。
- val Chamfer meanは 27.216mm から 30.262mm へ悪化。
- val target within 5mmは 16.0% から 12.9% へ悪化。
- epoch 110の `best.pt` を採用するべきで、epoch 500の `last.pt` は採用しない。

## Validation部品別coverage

![val coverage by part](../runs/cae_mesh_structured_q20_60_n24_e1000_seed13/diagnostics/val_coverage_by_part.png)

best epoch110でも、val部品のtarget coverageは低い。

| part | best target within 5mm | last target within 5mm | best target p95 |
|---|---:|---:|---:|
| `074` | 10.9% | 5.3% | 40.833mm |
| `024` | 11.3% | 10.9% | 30.316mm |
| `081` | 11.5% | 8.0% | 41.553mm |
| `077` | 13.5% | 12.7% | 31.683mm |
| `10` | 16.2% | 12.3% | 38.151mm |
| `026` | 32.6% | 27.9% | 30.948mm |

Chamferだけを見ると中程度に見える部品でも、target coverageは低い。これは、生成点が一部の面や線には近いが、正解面全体を覆っていないことを示す。

## 問題点の洗い出し

### 1. 過学習

epoch 110以降、train lossは下がるがval lossは改善しない。1000 epochを単純に回しても汎化性能は伸びにくい。

対策:

- early stoppingを導入する。
- best checkpointは必ずval loss基準にする。
- split seedを増やし、単一splitの偶然を避ける。

### 2. Coverage不足

val target within 5mmが平均16.0%しかない。これはCAE用途ではかなり不足している。

対策:

- Chamferに加えてcoverage lossを入れる。
- target-to-recon側を強くする重みを導入する。
- scaffold点の分散・面全体coverage regularizationを追加する。

### 3. Scaffold配置の破綻

worst部品 `081` ではscaffoldが正解の長い段付き構造へ乗らず、部品中央または片側へ偏る。局所refinement以前に粗構造が外れている。

対策:

- scaffoldを教師するpseudo targetを作る。
- FPS target scaffold、skeleton/medial graph、boundary-aware scaffoldを比較する。
- scaffoldにrepulsion / coverage / boundary attractionを入れる。

### 4. 境界・細長い構造の取りこぼし

`026`, `074`, `081` では、長い境界線、縦線状構造、段付き構造が欠落する。点は近傍に出るが、線や面の連続性を保証できていない。

対策:

- boundary distance metricを追加する。
- STEP tessellationから境界候補を抽出し、boundary tokenを追加する。
- boundary/protected-region logitsに教師を入れる。

### 5. 出力が点群でありtopologyがない

現出力は点群なので、CAE節点・要素・法線向き・接続品質を保証しない。CAEメッシュ生成の本質的な失敗をまだ評価していない。

対策:

- CAE Mesh IRへ拡張する。
- node / edge / face candidateを出す。
- deterministic projectionとquality repairを入れる。

### 6. `refinement_logits` が未教師

`refinement_logits` は出力されているが、lossも評価もない。現時点では信頼度や細分化優先度として使えない。

対策:

- local error、boundary proximity、joint proximity、curvature proxyを教師にする。
- logitsを評価CSVと可視化に出す。

### 7. 評価指標がまだCAE向けではない

現在の主評価はChamfer、nearest p95、within 5mmである。CAEメッシュのaspect ratio、warpage、Jacobian、normal consistency、solver importは未評価である。

対策:

- mesh quality metricsを導入する。
- solver import smoke testを評価gateへ入れる。
- target/recon point cloud評価は前段診断に限定する。

### 8. データsplitの偏り

val 6部品のうち、難しい部品が `A0072600002` 側に多く、splitごとの難易度差が大きい可能性がある。

対策:

- split seedを最低3つ以上回す。
- assembly/family/size帯でstratified splitを実装する。
- q20-60以外のサイズ帯でも別評価する。

### 9. checkpoint読み書きの競合

評価中にtraining checkpointを読むと、Windows file lockで `last.pt` 保存が失敗しうることが分かった。

対策:

- checkpoint保存をatomic writeへ修正済み。
- 評価時は `best.pt` をsnapshot copyしてから読む。

## 次の推奨実装

1. early stoppingを追加する。
2. coverage-oriented lossを追加する。
3. boundary-aware sampling / boundary metricを追加する。
4. scaffold pseudo targetを作り、scaffoldを直接評価する。
5. `refinement_logits` に教師と評価を入れる。
6. seed 13/29/47で同条件の短めrunを比較する。
7. CAE Mesh IRのnode/edge/face候補へ進む。

## 実装更新: coverage loss + scaffold loss

この診断を受けて、まず点群AEの範囲で直接効く改善として以下を実装した。

- `target_coverage_loss`
- `target_coverage_fraction`
- `scaffold_chamfer_loss`
- 学習CLI `--lambda-target-coverage`
- 学習CLI `--coverage-threshold-mm`
- 学習CLI `--lambda-scaffold`

目的:

- `target_coverage_loss` は、正解target点から復元点への最近傍距離が指定mm閾値を超えた部分に罰則を与える。
- `scaffold_chamfer_loss` は、target点群のFPS pseudo scaffoldを教師にして、coarse scaffoldが部品全体を覆うようにする。

スモーク確認:

```text
unit tests: 14 passed
CPU structured loss smoke: success
CUDA structured loss smoke: success
```

次の比較runでは、既存のq20-60 n24 seed13条件に以下を追加して、epoch 110 bestと比較する。

```powershell
--lambda-target-coverage 0.5 `
--coverage-threshold-mm 5.0 `
--lambda-scaffold 0.2 `
--batch-size 2 `
--preload-data `
--pin-memory
```

採否はtrain lossではなく、評価CLIの `val target within 5mm`、`val target p95`、worst部品のprojection overlayで判定する。

## 採用判断

このrunから、structured scaffold decoderは小規模holdoutより多い24部品でも学習自体は成立することが確認できた。一方で、汎化coverageはまだ低く、epochを伸ばしても解決しない。次の改善はモデルを長く学習することではなく、coverage、boundary、scaffold supervision、topologyを明示する方向で進めるべきである。
