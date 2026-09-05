# 拘束点ベース板金形状生成モデル（Neural SDF + Constraint Transformer）

## 1. 概要

本手法は、板金構造における**締結点（拘束点）**を起点とし、
それらの幾何拘束を利用して3次元形状（暗黙関数）を再構成するモデルである。

従来のSDF/UDF単独学習における以下の問題：

* floater（外れ値面）
* 虫食い（hole）
* 薄板構造の不安定性

に対して、**構造的拘束を導入することで安定化**を図る。

---

## 2. 拘束点の定義

### 2.1 拘束点の構成

各拘束点 ( C_i ) は以下の情報を持つ：

[
C_i = (\mathbf{x}_i, \mathbf{n}_i, t_i)
]

* (\mathbf{x}_i \in \mathbb{R}^3)：位置
* (\mathbf{n}_i \in S^2)：法線（単位ベクトル）
* (t_i)：拘束タイプ（ボルト・溶接など）

---

### 2.2 制約条件

* 拘束点は **最低2点以上必要**
* 法線情報は必須（面の向きを拘束するため）

---

### 2.3 拘束点の拡張（局所平面化）

各拘束点はスカラー点ではなく、以下の局所平面として扱うことが可能：

[
P_i = \left{
\mathbf{x} \mid
(\mathbf{x} - \mathbf{x}_i) \cdot \mathbf{n}_i = 0,
\ |\mathbf{x} - \mathbf{x}_i| \leq r
\right}
]

* 半径 (r = 10 \text{ mm} )（直径 φ20、可変）
* 拘束点を「微小パッチ」として扱う

---

## 3. モデル構成

### 3.1 全体構造

```
拘束点集合 C
      ↓
Transformer / GNN
      ↓
拘束特徴 {h_i}
      ↓
空間クエリ x
      ↓
Decoder
      ↓
SDF d(x), Normal n(x)
```

---

### 3.2 拘束エンコーダ

[
h_i = \phi_{\text{enc}}(\mathbf{x}_i, \mathbf{n}_i, t_i)
]

---

### 3.3 幾何統合（Transformer）

[
h_i' = \text{Transformer}({h_j})
]

#### Attention設計：

[
\text{Attention}_{ij} \sim
|\mathbf{x}_i - \mathbf{x}_j|

* \mathbf{n}_i \cdot \mathbf{n}_j
* \text{type relation}
  ]

---

### 3.4 デコーダ（暗黙関数）

任意点 (\mathbf{x}) に対して：

[
d(\mathbf{x}), \mathbf{n}(\mathbf{x}) =
\phi_{\text{dec}}(\mathbf{x}, {h_i'})
]

---

## 4. 学習目標

### 4.1 距離損失

[
\mathcal{L}_{dist} = |d(\mathbf{x}) - d^*(\mathbf{x})|
]

---

### 4.2 法線損失

[
\mathcal{L}_{normal} =
1 - \mathbf{n}(\mathbf{x}) \cdot \mathbf{n}^*(\mathbf{x})
]

---

### 4.3 勾配整合性

[
\mathcal{L}_{grad} =
|\nabla d(\mathbf{x}) - \mathbf{n}(\mathbf{x})|
]

---

### 4.4 Eikonal制約

[
\mathcal{L}_{eikonal} =
(|\nabla d(\mathbf{x})| - 1)^2
]

---

### 4.5 拘束損失（重要）

#### (1) 拘束点上での一致

[
d(\mathbf{x}_i) = 0
]

---

#### (2) 法線一致

[
\mathbf{n}(\mathbf{x}_i) = \mathbf{n}_i
]

---

#### (3) 局所平面一致（拡張）

[
d(\mathbf{x}) \approx 0 \quad (\mathbf{x} \in P_i)
]

---

## 5. サンプリング戦略

### 5.1 空間サンプル

* 表面近傍
* ランダム空間点

---

### 5.2 拘束周辺サンプル

* 各拘束点の局所平面上
* 法線方向オフセット点

---

## 6. 本手法の特徴

### 6.1 メリット

* floater抑制（拘束未接続点の排除）
* 虫食い防止（拘束からの補間）
* 薄板安定化（法線拘束）
* 面の一貫性向上（Transformer）

---

### 6.2 従来法との違い

| 項目    | 従来SDF | 本手法  |
| ----- | ----- | ---- |
| 入力    | 点     | 拘束構造 |
| 幾何拘束  | 弱い    | 強い   |
| 薄板    | 不安定   | 安定   |
| トポロジー | 不定    | 半拘束  |

---

## 7. 拡張方向

### 7.1 物理拘束

* bending energy
* stretching energy

---

### 7.2 曲率導入

[
\kappa = \nabla \cdot \mathbf{n}
]

---

### 7.3 パネル単位トークン

* 面クラスタリング
* 面単位生成

---

## 8. まとめ

本手法は、

**「拘束点から形状を生成する」構造的アプローチ**

であり、

* 暗黙関数（SDF）
* 法線推定
* Transformerによる大域整合
* 板金特有の締結拘束

を統合したモデルである。

特に板金のような**構造物形状生成**において高い有効性が期待される。
