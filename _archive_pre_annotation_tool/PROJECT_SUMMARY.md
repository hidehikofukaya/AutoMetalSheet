# AutoMetalSheet — プロジェクトサマリー（他PC引き継ぎ用）

> 作成日: 2026-06-18
> 目的: 別環境のClaude Codeにこのプロジェクトの文脈を渡すための要約。
> 詳細な意思決定履歴（Round 1〜10、判断ID付き）はすべて `CLAUDE.md` に記録されている。
> このファイルは「今どこにいて、何に詰まっているか」を素早く掴むためのナビゲーション層であり、
> `CLAUDE.md` の代替ではない。深掘りする際は必ず `CLAUDE.md` の該当セクションを参照すること。

---

## 1. プロジェクトの全体像

**最終ゴール**: 板金設計（自動車ボディ・BIW = Body In White）の自動化システムを作ること。
CADデータ（CATPart/STEP）から板金の慣例・DFM（製造性）制約・材料力学を踏まえた設計を
半自動生成し、設計者の工数を削減する（詳細: `CLAUDE.md` §1〜12、エージェントアーキテクチャ
Round 4/5の確定設計判断 R4-01〜R5-15）。

このフルスコープは非常に大きく、Phase 1〜3・Stage 0〜4の段階的ロードマップが
`CLAUDE.md` §9, §12 に確定済み。

**現在進行中のPoC（このリポジトリの実体）**は、上記の壮大な構想とは別建ての、
もっと焦点を絞った技術検証である：

> **板金STPファイルから中立面（mid-surface）点群を抽出し、それをAIで
> 幾何形状（中立面メッシュ）に復元できるか？** という単一の技術的問いの検証（`CLAUDE.md` §13〜17）。

これは「将来このAIが使い物になるなら、パラメトリックなフィーチャー復元・板金CAD自動生成の
土台になりうる」という仮説のもとに進めている、Stage 1のPoC。今のスコープは
**メッシュ復元のみ**（曲げ角度やフランジ長を編集可能パラメータとして出力する
パラメトリック復元はやらない、`CLAUDE.md` R6-05で確定）。

---

## 2. PoCのアーキテクチャ（3ステージ）

```
Stage A: 決定論的データ生成（AI不使用）
  body###_filled.stp（穴埋め済みソリッド）
    → テッセレーション → 中立面投影（壁面除外つき） → 中立面メッシュ構築
    → ノイズ入り学習用点群 + GT UDFクエリサンプリング
    → dataset/<part>.h5

Stage B: VecSet Autoencoder（ニューラルネット、UDF=符号なし距離場を学習）
  points[8192,3] → FPS+kNN → mini-PointNet → tokens
    → Self-Attn Encoder → Cross-Attn(learnable queries) → 潜在ベクトルZ
  query_xyz → Fourier位置エンコーディング → Cross-Attn Decoder → { UDF, normal, confidence }

Stage C: 幾何復元（決定論的後処理）
  学習済みモデル + 密グリッド評価 → 決定論的ゲート（confidence/距離）で候補点フィルタ
    → UDFメッシュ抽出（VTK reconstruct_surface） → メッシュ分割+再投影で高密度化
    → prune_far_vertices()で外挿頂点除去 → reconstructed_midsurface.ply
```

なぜUDF（符号なし距離場）でSDF（符号付き）ではないか: 中立面は厚みゼロの真の開多様体
（inside/outsideが定義できない）であるため。これがStage1全体の設計の出発点
（`CLAUDE.md` §13、R6-01〜R6-06）。

### 実装ファイル
| ファイル | 役割 |
|---|---|
| `biw_poc/src/preprocess/hole_filler.py` | 穴検出・分類（jig/bolt/design_feature）・自動穴埋め |
| `biw_poc/src/preprocess/midsurface_sampler.py` | Stage A一式（中立面投影〜h5データセット生成） |
| `biw_poc/src/model/vecset_ae.py` | Stage Bモデル本体（VecSetAE, QueryDecoder, FourierFeatures） |
| `biw_poc/src/model/dataset.py` | 学習データセット（正規化・正規化スケール保存） |
| `biw_poc/src/model/train.py` | 学習ループ（truncated-distance loss, Eikonal正則化） |
| `biw_poc/src/model/reconstruct.py` | Stage C本体（eval_grid, reconstruct_midsurface, prune_far_vertices, subdivide_and_project） |
| `biw_poc/src/model/validate_refine.py` | refine_rounds等のパラメータスイープ検証スクリプト |
| `biw_poc/src/model/diagnose_outliers.py` | Recon→GT方向（突起・オーバーシュート）の診断 |
| `biw_poc/src/model/diagnose_holes.py` | GT→Recon方向（被覆穴＝「虫食い」）の診断 |
| `biw_poc/src/viewer/stp_viewer.py` | STPビューワー（穴の手動編集・fill実行UI） |
| `biw_poc/run_*.bat` | 各スクリプトの起動バッチ |

---

## 3. 現状（このサマリー作成時点）でどこまで進んでいるか

### 確定・検証済み（body002の単一部品overfitサニティチェックで検証）
- Stage A〜Cの全パイプラインが動作し、最終的な復元精度は実用域に到達:
  - GT→Recon（内部形状の再現精度）: mean 1.62mm（目標1〜2mm達成）
  - Recon→GT（不要メッシュの少なさ）: mean 8.57mm, max 47.27mm
- 主要な技術的バグを2つ特定・解決済み:
  1. **TD-16「突起」問題**: UDFの境界（cut locus、切断端面）での外挿によりGT外に
     ゴーストメッシュが生成される既知のOCCT/UDF系統の問題。R7-11+R7-12の決定論的ゲート
     （距離ベースの候補点フィルタ + 頂点剪定）で緩和済み（mean 53.83mm→8.57mm、6.3倍改善）。
     **max=47.27mmは未解決のまま残存**（根治にはMeshUDF/CAP-UDF/DUDF等の専用手法が必要、
     未着手）。
  2. **「虫食い」問題**（GT→Recon方向の被覆穴）: メッシュ密度の天井とprune閾値のトレードオフが
     原因と特定。`refine_rounds`を3→4、`prune_dist_threshold_mm`を12→20に変更して緩和
     （穴>5mm率 5.3%→1.1%）。**1.1%の残差は許容済み**（これ以上はメッシュサイズが
     不釣り合いに増大するため）。
- Deep Ensembles方式によるOOD検出は**検証の上で不採用**と結論（系統バイアスにより
  disagreement signalが機能しないことを実証、`CLAUDE.md` Round 8）。
- トポロジーパッチ（GT境界の断片化を補修する前処理）は実装したが、**Stage Cの最終精度には
  統計的に有意な効果なしと判明し棄却**（n=5 seed × 2条件のt検定、Round 9, R9-01）。

### 未着手・保留中（ユーザー判断待ち）
- DUDF（CVPR 2024、cut locus非微分可能性をネットワーク学習時に修正する手法）の導入検討。
  「虫食い」問題の調査を優先したため一時保留中。再開はユーザーの明示的指示があった場合のみ
  （`CLAUDE.md` §17冒頭）。
- max誤差47.27mmの根治。

---

## 4. 直近のセッションで対応した内容（このサマリー作成のきっかけ）

ユーザーから「fillをしたstpを見るとまだ穴が複数残っている。ビュワーでfillしようとすると
"defeaturing completed but some holes may remain"と出る」というバグ報告があり、調査・修正した。

**根本原因は2つ判明**:
1. **設計通りの動作（バグではない）**: 直径15mmを超える穴は `design_feature`（実設計フィーチャー）
   として分類され、意図的に自動穴埋めの対象外（R6-03の不変条件）。検出された穴の約46%が
   これに該当しており、「穴が残っている」という見え方の主因だった。
2. **本物のバグ（修正済み）**: OpenCASCADEの `BRepAlgoAPI_Defeaturing.Build()` が特定の
   穴形状に対して無限ループしてフリーズする既知のOCCT欠陥（dev.opencascade.orgのフォーラムで
   報告あり、本プロジェクトでも実機で20分以上のハングを再現・確認）。

**修正内容**（`hole_filler.py`）: 各穴を独立したサブプロセスで処理し、60秒のwall-clock
タイムアウトを設定。1つの穴がハングしても他の穴・他のボディ・パイプライン全体をブロック
しないようにした。ビュワー側 (`stp_viewer.py`) も穴ごとの成否を表示するよう更新済み。

**現在のステータス**: 修正版での全アセンブリ再処理バッチがバックグラウンドで実行中
（直前の進捗確認時点: 279ボディ処理済み、`ok`230 / `no_holes`34 / `defeaturing_partial`10 /
`defeaturing_failed`5。穴単位では `ok`522 / `timeout`5 / `failed`25 で、いずれも個別の穴の
スキップのみで全体停止は発生していない）。**このバッチが完走しているかどうかは
このサマリーを読んでいる時点で要確認**（進行中だった場合は完了を待つこと）。

---

## 5. リポジトリ構成・git運用に関する注意

- **このリポジトリにモデルチェックポイント（`biw_poc/src/model/checkpoints*/`、合計約36GB）は
  含まれていない**（`.gitignore`で除外）。再学習は `run_train.bat` で可能。
- **fill済みSTPファイルの全量（約9.3GB、Mesh_Generater/output配下）もこのリポジトリには
  含まれていない**。`biw_poc/data/filled/` に、動作確認・参照用として無作為抽出した
  **15個のサンプルのみ**を同梱している（ファイル名は `<assembly>_<body>_filled.stp` 形式）。
- 生成物（`*.ply`, `*.h5`, `*.log`, `*.stp`/`*.step`〈サンプル除く〉）は `.gitignore` で除外済み。
- 学習・推論を再現するには、別途 `C:\Users\hide2\IdeaBox\Mesh_Generater\output\` 配下の
  元データ（STPアセンブリ群）にアクセスできる環境が必要（このリポジトリには同梱されていない）。

---

## 6. このプロジェクトを引き継いだ場合に最初に読むべきもの

1. `CLAUDE.md`（このリポジトリのルート） — 全設計判断の確定履歴（R1〜R10、判断ID・根拠・
   依存関係グラフ付き）。特に §13〜17が現在進行中のPoCの直接の文脈。
2. `fable/stage1_poc_plan.md` — PoCの当初計画書。
3. このファイル（`PROJECT_SUMMARY.md`） — 現在地と直近の詰まりどころのナビゲーション。
