# 計画: Wireframe Flow AE — ワイヤーフレーム学習+拘束点条件付き部品復元

Date: 2026-07-08
Author: fable5(ユーザー承認済みアイデアの実行計画)
Status: 実装計画(Phase 0 から着手可能)

## 0. 一文サマリ

型付きワイヤーフレーム(wireframe v4.4)を教師とし、「bbox 内のランダム点を transformer が反復移動させて
ワイヤーフレーム点群に収束させる」flow matching デコーダを持つオートエンコーダーを構築する。
学習時に latent をランダムに落とす(conditioning dropout)ことで、**同一モデルが「自部品復元(診断)」と
「拘束点のみからの生成(本命)」の両方を学び**、anchored scaffold で起きた「入力ショートカットで易化した
数字を実力と誤認する」問題を構造的に回避する。

## 1. タスク定義

```text
教師:   型付きワイヤーフレーム点群(外周/穴/曲げ線/曲面フレーム、+接線/半径)
条件:   拘束点(joints.json の締結点: 座標+軸)+ bbox/スケール
入力:   bbox 内一様ランダム点群(flow matching の source)
モデル:  encoder E(wireframe) -> latent z(AE 枝)
        flow decoder F(点群, t | z(dropout), 拘束点, bbox) -> 各点の速度
出力:   ワイヤーフレーム点群 + 点ごとのタイプ
```

- **z あり復元** = オートエンコーダー診断(デコーダ容量・表現の検証)
- **z なし生成** = 拘束点のみからの部品復元(本命タスク)
- 両者を**必ず別々に報告**する。z あり数字だけを見て前進判断をしない(anchored scaffold の教訓)

## 2. なぜこの設計か(過去の教訓との対応)

| 教訓 | 本計画での対応 |
|---|---|
| 点群AEは非順序点群の構造欠如が天井 | 教師を密な中立面ではなく**構造的なワイヤーフレーム**に変更(ユーザー方針) |
| anchored scaffold のオラクルリーク | encoder 入力→出力の直通経路なし。latent dropout で z なし性能を常時測定 |
| 512点サンプリング床 | 評価は高解像度 target(ポリラインから4096点)+自己Chamfer床併記 |
| epoch毎サンプリング凍結 | ポリライン上の点サンプリングを**毎epoch再抽選**(実装済みパターンを踏襲) |
| mirror-y が有効 | part-level split 後の train-only mirror を初日から適用 |
| 65部品で生成は検索に負けうる | **train-NN 検索ベースラインを必ず併走**(拘束点配置の最近傍部品を返す) |

## 3. データパイプライン(Phase 0)

### 3.1 ワイヤーフレーム教師

ソース: `fill_volume/fill_mid_surf/<asm>/wireframes/*.json`(schema v4.4、140部品)

- ポリラインから**長さ比例+タイプ層化**でN点サンプリング(既定 N=1024)
  - タイプ配分の既定: outer 30% / hole 15% / bend 30% / surface_frame 25%(seam/crease は除外・微量)
  - 点特徴: xyz(部品中心+最大辺で正規化、既存 fill_volume_dataset と同方式)、タイプ one-hot、
    ポリライン接線ベクトル、log(face_radius)(隣接面の小さい方、無限は0)
- **手動 overrides を必ずマージ**(現在 server 側でのみ適用されている。dataset builder にも
  `apply_overrides` 相当を実装 — ユーザーの修正が教師に反映される経路)
- 毎epoch再抽選: sample_seed に epoch を混ぜる

### 3.2 拘束点条件

ソース: `annotations/joints.json`(A0072600002: 459 / A0072601285: 464)

- 部品ごとに per_part の hole_center_xyz / contact_xyz + axis direction を抽出 → (xyz, 軸ベクトル, joint種別 one-hot)
- **A0072600529 は joints.json 未整備**(65部品)→ 拘束トークンなし(=条件 dropout 状態)で学習に参加。
  データ課題: 529 のアノテーションを annotation_app で整備すれば拘束条件付きデータが約2倍になる(推奨・並行作業)
- bbox: 正規化後の half-extent 3値+log(scale_mm) を1トークンに

### 3.3 分割

- part-level split、seed 13(過去実験と比較可能に)、train/val ≈ 110/30
- 層化: アセンブリ+bbox対角+曲げ線本数
- mirror-y は split 後 train のみ(点・接線・軸すべて反転)

## 4. モデル仕様(Phase 1-2)

小さく始める。全体で ~10M param 級。

- **Encoder E**: ワイヤーフレーム点群(1024点×特徴)→ transformer 4層 → **latent 32トークン×256次元**
- **Flow decoder F**: DiT 構成 6〜8層×256次元×8頭
  - トークン = 生成中の点 M=1024(xyz 埋め込み+時刻 t 埋め込み)
  - cross-attention 先: [z(**dropout p: 0.3→0.7 カリキュラム**), 拘束トークン, bbox トークン]
  - 出力: 速度(3次元)+タイプ logits(補助ヘッド、CE 損失)
- **Flow matching**: 線形補間経路、速度回帰 MSE。source = bbox 内一様分布
  (flow matching は source 任意 — ここがユーザーのアイデアと理論が噛み合う点)
- **ペアリング**: v1 は独立カップリング。ぼやけるようなら minibatch-OT(Hungarian/Sinkhorn)に変更
- サンプリング: Euler 20〜50 step。z なし時は classifier-free guidance(条件 dropout の副産物)
- 拘束充足の構造保証: サンプリング中、穴タイプの一部の点を拘束点位置に**インペイント固定**(オプション、Phase 2 後半)

## 5. 評価(Phase 共通)

1. **タイプ別 Chamfer**(生成→GT / GT→生成、mm、較正: GTポリラインから4096点+自己Chamfer床併記)
   - 曲げ線と外周を重視(生成の骨格)。surface_frame は参考値
2. **拘束充足距離**: 各拘束点 → 生成された穴タイプ点群の最近傍距離(max/mean)
3. **z-on vs z-off の乖離**: AE 復元と拘束のみ生成を常に並記。z-off が本命の成績
4. **train-NN 検索ベースライン**: 拘束点配置が最も近い訓練部品のワイヤーフレームをそのまま返した場合の
   同指標。**これに勝てない設定は「生成」と呼ばない**
5. タイプ分類精度(補助ヘッドの confusion)
6. **可視化: 生成結果を wireframe v4.4 互換 JSON で出力し、既存ビュワーでGTと並べて表示**
   (server に `?generated=<run>` 読み込みを足すだけ。目視検証インフラを再利用)

## 6. フェーズ計画

| Phase | 内容 | 完了条件 |
|---|---|---|
| **0. データ** | dataset builder(3.1-3.3)+統計+検索ベースライン実装 | 140部品のサンプル可視化がビュワーで確認できる |
| **1. AE flow** | z あり復元のみ(dropout 0)。デコーダ容量の検証 | val Chamfer が床の~2倍圏内、タイプ精度>90% |
| **2. 条件付き** | latent dropout + 拘束トークン + CFG。z-off 評価解禁 | z-off が検索ベースラインと勝負になる |
| **3. 強化** | OT カップリング / インペイント固定 / 529 joints 整備 / Layer 2 パラメトリック教師 | — |

Phase 1 と 2 は同一コードで dropout スケジュールが違うだけ。Phase 0 が最大の実装量。

## 7. 実装配置

- `cae_mesh_generator/src/cae_mesh_generator/wireflow/` に新規モジュール
  (dataset.py / model.py / train.py / evaluate.py / export_viewer_json.py)
- 正規化・split・mirror・評価較正のユーティリティは既存 AE コードから流用
- wireframe JSON の読み込み+overrides マージは wireframe_app/extract.py に依存せず単独実装(JSON を読むだけ)

## 8. リスクと正直な見通し

- **140部品**: z-off 生成が検索に勝つのは容易でない。Phase 2 の判定は「勝つ」ではなく
  「検索と同等+補間の兆候(訓練にない拘束配置で妥当な形状)」でよしとする
- **拘束条件の被覆**: joints 付きは75部品のみ。529 のアノテーション整備が効く投資
- **surface_frame の支配**(33k本): サンプリング層化で抑制するが、タイプ重みは要チューニング
- ワイヤーフレーム→中立面(密な面)への復元は本計画の範囲外。Phase 3 以降に
  「ワイヤーフレーム+凍結 refinement decoder」または UDF で面を張る
