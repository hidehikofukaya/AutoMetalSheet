# R0 PoC: 出力監査とGo/No-Go GUI

R0の第一弾は、既存または新規の中立面メッシュを同じ評価契約で監査し、
各ステージの成果物、正式な双方向point-to-surface指標、メッシュ品質、
hard gateとPhase判定をGUIで確認するための基盤です。

## 1. 既存ステージ出力の監査

```powershell
cd C:\Users\hide2\IdeaBox\AutoMetalSheet

python biw_poc\src\r0\audit_run.py `
  --profile biw_poc\configs\r0_audit.yaml `
  --reference path\to\gt_midsurface.ply `
  --stage coarse=path\to\coarse.ply `
  --stage refine1=path\to\refine1.ply `
  --output-dir biw_poc\runs\my_run `
  --part-id PART-001 `
  --part-family FAMILY-A
```

`run_r0_audit.bat`も利用できます。

出力:

```text
runs/{run}/
  run_manifest.json
  audit_events.jsonl
  stage_*_metrics.json
  stage_*_samples.npz
```

正式評価は各方向10万点の面積一様サンプリングと、
相手側三角形面への最近点距離を使います。評価点と距離も保存され、
SHA-256がmanifestに記録されます。

## 2. profile駆動のStage C実行

```powershell
python biw_poc\src\r0\reconstruct_run.py `
  --profile biw_poc\configs\r0_audit.yaml `
  --checkpoint path\to\best.pt `
  --data path\to\dataset.h5 `
  --reference path\to\gt_midsurface.ply `
  --run-dir biw_poc\runs\new_reconstruction
```

候補証拠が不足した場合、ungated gridへのfallbackは行わず
`InsufficientEvidenceError`で停止します。

## 3. GUI

```powershell
biw_poc\run_r0_viewer.bat biw_poc\runs\my_run\run_manifest.json
```

GUIで確認できる情報:

- referenceと各stage meshの表示切替
- referenceに対する距離ヒートマップ
- 頂点数、面数、面積、component、最小角、aspect ratio、辺長
- 双方向point-to-surface指標
- hard gateごとの実測値、閾値、合否
- `RefinementRunStatus`と`PhaseGateStatus`
- manifest、artifact、metrics、評価sample、AuditEvent chainの整合性

GUIはGo判定を再計算しません。`run_manifest.json`と保存済みgate結果を正本として
表示します。3Dヒートマップは視覚補助であり、Go判定値ではありません。

## 現在の完了範囲

R0の証拠・可視化基盤を開始できる状態です。Technical Goには今後、
最低8部品・3 family、nested grouped validation、family-cluster統計が必要です。
bounded adaptive baselineは比較用に実装済みですが、投影付きadaptive remeshingと
GNNはまだGo判定対象ではありません。

## 4. 複数部品の集約

15部品の対象とfamilyは`configs/r0_cohort.yaml`に事前固定しています。

```powershell
biw_poc\run_r0_aggregate.bat `
  biw_poc\runs `
  biw_poc\runs\r0_cohort_manifest.json `
  coarse `
  adaptive

biw_poc\run_r0_cohort_viewer.bat `
  biw_poc\runs\r0_cohort_manifest.json
```

集約は部品単位のpaired improvementを計算し、family平均を等重みで扱う
family-cluster bootstrapを使用します。欠落部品、重複part ID、family不一致、
profile/evaluator/hashの混在、power analysis未達があればTechnical Goにしません。

## 5. Bounded adaptive baseline

```powershell
biw_poc\run_adaptive_baseline.bat `
  path\to\coarse_midsurface.ply `
  biw_poc\runs\adaptive_part001 `
  8.0
```

このbaselineは次を実装します。

- `length > 4/3 * target_edge`の長辺だけを選択
- 共有辺を両側で同時更新するconforming split
- 1面1splitに限定した非競合batch
- 全生成子辺の`h_floor`検査
- 1 sweepのsplit率・頂点増加率・総頂点数のhard limit
- component、boundary component、短辺数、面積、最小角、aspect p95の単調gate
- candidate mesh hash付きtransaction、失敗時rollback

現時点では既存入力に含まれる極短辺をcollapseしません。残存時は
`STALLED_SAFE`として記録し、次段階の入力正規化collapseへ引き継ぎます。
