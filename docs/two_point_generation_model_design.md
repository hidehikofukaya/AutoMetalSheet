# 設計: 締結点2点 → 板金中立面 生成モデル(synthetic_parts 1,300部品)

Date: 2026-08-26
Author: fable5
Data: `C:\Users\hide2\IdeaBox\PartMaker\synthetic_parts`(batch02 100 + prod01 1,200 = 1,300部品、
flange 331 / bead 969、mid STEP + params JSON + joints.json)

## 0. 設計原則(本セッションの実験結果から)

| 教訓(出典) | 本設計への反映 |
|---|---|
| 大域レイアウトの分布は百件規模の自由空間生成では学べない(wtok 3世代) | **生成対象を低次元パラメトリック集合に圧縮**(骨格+詳細で〜15次元+離散1) |
| 「あり得ないフレーム」の根源は面拘束の欠如(curve v2 診断) | ワイヤー/面は生成せず**決定論的に実現** — 面から浮いた形状が表現上存在しない |
| 拘束充足はソフト制約でなく構成的に(CIRCLE_C の教訓) | 締結穴は実現器が p1/p2 に**構成的に配置**(充足誤差ゼロ) |
| 同一拘束に複数の妥当形状(多様性)が本質(roadmap §1) | 点推定回帰でなく**条件付き flow matching**(+CFG)で多モード分布を生成 |
| FM損失と実品質は乖離しうる / 検索ベースライン必須(wireflow) | 評価は min-of-N Chamfer+多様性+**検索・回帰の2ベースライン**併走 |
| 条件由来の正準化(final_wireframe_theory 定義4) | フレームは条件のみから決定(p1原点、p1→p2方向、n1 で決まる相似変換) |

## 1. 問題の定式化

**条件 C**(生成時に既知): p1, n1, p2, n2(締結点座標+軸)、板厚 t、穴径 d、座面半径。
**出力**: 中立面形状。本データ族では形状は次の自由度ベクトルで完全に決まる:

- 連続 θ(〜13次元): half_width, bend_radius, fold1_slack, fold2_slack,
  fold1_tilt_perturbation、feature連続パラメータ(flange: height, root_radius /
  bead: depth, top_width, wall_angle, ridge_radius, corner_radius)
- 離散 c: feature class {flange, bead}+flange の side/direction(±1)

つまりタスクは **p(θ, c | C) の条件付き生成**であり、幾何は θ,c から決定論的に実現される。
「Transformer か diffusion か」への答え: **この段では拡散系(flow matching)が適切**
(出力が短い連続ベクトル+少数離散で、系列構造がないため AR Transformer は過剰。
Transformer は Track 2 の幾何空間生成で使う)。

## 2. アーキテクチャ(Track 1: パラメトリック flow、本命)

```text
条件エンコーダ: g_C 正準化(原点p1、x軸=(p2-p1)/|p2-p1|、y軸=n1の直交化、
               スケール=|p2-p1|)→ [p2', n1', n2', |p2-p1|, t, d, 座面半径] ≈ 12次元
離散ヘッド:    p(c|C) — 小型MLP分類器(flange/bead、flange時 side/direction)
連続生成:      conditional flow matching(rectified flow)
               v_φ(θ_t, t | C, c) — FiLM条件付き MLP(幅512×4層、~1M params)
               θ は分位正規化(パラメータごとに train 分布で [-1,1] へ)
サンプリング:   c ~ p(c|C) → θ ~ flow(Euler 20step、CFG w=1〜3)→ N本サンプルで多様性
```

学習: 1,300部品、val 10%(130、法線内積クラスで層化)。θ 教師は params JSON から直読
(教師ラベル取得コストゼロ)。数分/学習の規模なのでアブレーションが自由。

## 3. 実現器(Stage B、学習なし)

θ, c → 中立面を Python で決定論的に構築(CATIA 不要の軽量版):

1. 骨格: p1→p2 を3平面セグメント+2円筒曲げ(fold_tilts は n1, n2, perturbation から
   生成器と同じ規則で導出)で接続、幅 half_width の帯として展開
2. 締結穴: p1/p2 に径 d の穴を**構成的に**配置
3. 詳細: flange(縁の立ち上げ)/ bead(リッジ掃引)を θ で付加
4. 出力: サンプル点群+型付きワイヤー(既存 wtok/viewer 形式) → 既存評価・可視化基盤を流用

精度検証: 実現器出力 vs GT mid STP の Chamfer(実現器の系統誤差 = 評価の床。
PartMaker 生成器と同一規則の再実装なので小さいはず。> 2mm ならフィレット等を追加)。

## 4. 評価プロトコル(セッション標準)

1. **拘束充足**: 構成的にゼロ(実現器が保証)— 報告のみ
2. **min-of-N Chamfer**(N=8): 生成8案の最良 vs GT mid。多モード分布の正しさを測る
3. **多様性**: 同一条件の8案のペアワイズ Chamfer と feature class 分布
4. **ベースライン**: (a) 回帰MLP(θ点推定 — 多モードを平均化するはず)、
   (b) 検索(train から条件最近傍の θ を流用)。**flow は (a) に min-of-N で、
   (b) に補間性(未見条件での妥当形状)で勝って初めて採用**
5. パラメータ別誤差(θ の次元ごと)— どの自由度が条件から決まり、どれが自由かの分析

## 5. Track 2(併走): wtok カーブ主導 AR の 10倍データ再検証

1,300部品の mid STP を既存パイプライン(wireframe_app v4.4 → wtok 変換)に通し、
カーブ主導 AR(トラバース順 v2)を再学習する。目的:

- 「126部品では大域一貫性が学べない」の結論が、**10倍・単純族**でどう変わるかの科学的検証
- Track 1(族に閉じた表現事前分布)との定量比較 = 表現事前分布の価値の測定
- 実部品への転移経路は Track 2 側にしかない(Track 1 は族ロックイン)ため、
  合成事前学習→実140部品微調整の布石

## 6. フェーズ計画

| Phase | 内容 | ゲート |
|---|---|---|
| P0 | データローダ+正準化+層化split+2ベースライン | ベースライン数値の確定 |
| P1 | flow matching 学習+パラメータ空間評価 | min-of-N で回帰に勝つ |
| P2 | Python 実現器+Chamfer/レンダリング評価 | 実現器床 <2mm、生成 min-of-N Chamfer 報告 |
| P3 | Track 2: 1,300部品の wtok 変換+AR 再学習 | v2(140部品)との比較表 |
| P4 | 転移設計: 合成事前学習→実データ微調整(将来) | — |

P0-P2 が本命で、モデルは小さく学習は数分規模。P3 は GPU 数時間(投入前確認)。
