# アーキテクチャ再検討: 2025-2026 最新技術サーベイと路線への含意

Date: 2026-08-27
目的: 案B(graph transformer)実装着手前に、最新の CAD/ワイヤーフレーム生成研究を
調査し、当プロジェクトの3路線(C=純AR / AB=計画+AR / B=グラフ反復精緻化)を
外部の知見と突き合わせて再評価する。

## 1. 調査で判明した最重要事実

**当プロジェクトが独自に辿り着いた設計判断の多くが、2025-2026 の frontier と一致していた。**
逆に、frontier が既に解決策を持つ課題(溶接・離散拡散の非効率)も見つかった。

### 1.1 CLR-Wire (SIGGRAPH 2025) — 「AB案の大規模版」が既に存在する

[arXiv:2504.19174](https://arxiv.org/abs/2504.19174) /
[GitHub](https://github.com/qixuema/CLR-Wire)

まさに「3D曲線ワイヤーフレーム生成」を扱う論文。構成:

| 要素 | CLR-Wire | 当方 AB案 |
|---|---|---|
| 曲線表現 | 正規化+256点サンプル → Curve VAE latent (4×3) | wtok 量子化列 |
| 全体表現 | Wireframe VAE → 固定長 latent Z∈R^(64×16) | PlanEncoder → 8 latent tokens |
| 生成 | **latent flow matching** (DiT 12層×768, AdaLN-Zero) | PlanFlow (FiLM-MLP) |
| 条件 | PointNet++ 特徴を AdaLN-Zero 注入。**疎な1K点で従来法の20K点相当** | FIX点+包絡を prefix |
| トポロジ | **BFSソート隣接リストの差分列** を latent に同梱 | AR デコーダが暗黙に生成 |
| データ | ABC 130,473 個 | 合成 2,300 個 |

結果: 無条件生成で 3DWire を上回り、点群条件付き再構成 CD 8.26(RFEPS 13.79 /
NerVE 12.73 に大差)。**ただしトポロジ整合率は 81.79% 止まり**(自己交差・
接続誤り・余剰曲線が残る)と正直に報告している。

含意:
- AB案(VAE latent + FM prior + 条件付け)は方向として正しい。世界最高会議で
  同型のアーキテクチャが SOTA を取っている
- ただし彼らは **130K サンプル**で学習。当方の 2.3K では同じ強さは出ない
- 彼らですらトポロジ 18% を落とす。「latent 経由の暗黙トポロジ」の限界も見えている
- 条件注入は prefix より **AdaLN(-Zero) 注入**が定番化している(FiLM の発展形。
  当方の PlanFlow/twopoint で実証済みの機構と同系)

### 1.2 Flatten The Complex (2026-01) — 案Bの「溶接リスク」への構成的解答

[arXiv:2601.17733](https://arxiv.org/abs/2601.17733)

B-Rep を **k-cell 粒子の集合**(頂点=0-cell、辺=1-cell、面=2-cell)として平坦化し、
**multi-modal flow matching** でトポロジと幾何を同時生成。決定的な工夫:

> 隣接するセルは境界インターフェースで **同一の latent を共有** する
> (adjacent cells share identical latents at their interfaces)

つまり「辺Aの端点」と「辺Bの端点」が同じ頂点なら、生成対象として **同じ変数**
を指す。decode 後に座標を突き合わせて溶接するのではなく、**表現の時点で接続が
成立している**。非多様体構造(=ワイヤーフレームそのもの)の合成も自然に扱えると明記。

含意: 案B設計のリスク1位「端点溶接の失敗」は、**エッジが端点座標を内包する**
(HouseDiffusion 方式)のをやめ、**頂点トークン+エッジからの共有参照**にすれば
構成的に消せる。これは当方の CIRCLE_C fix_ref(参照は座標より強い)と同じ思想の一般化。

### 1.3 トポロジ→幾何の2段分離が2025年の共通解

- **TG-Diff** ([arXiv:2607.21928](https://arxiv.org/abs/2607.21928)):
  離散拡散でトポロジ(面隣接)を先に生成 → トポロジ条件付きで幾何拡散。82M で SOTA
- **DTGBrepGen** ([arXiv:2503.13110](https://www.alphaxiv.org/abs/2503.13110)):
  同じくトポロジ/幾何のデカップリングでトポロジ妥当性を強制
- **BrepForge** ([arXiv:2605.19411](https://arxiv.org/abs/2605.19411)):
  face-aware AR でトポロジ完全な骨格を作り、幾何は境界ループから**学習フリー**で
  インスタンス化。「トポロジ=高エントロピーの決断、幾何=境界に従属」という非対称性を利用

含意: 案B の MVP 計画(型・要素数を先に一括サンプル → 座標のみ FM)は簡略化の
つもりだったが、**frontier ではそれ自体が本命構成**。「まず離散骨格、次に条件付き
座標」は正式な2段設計に格上げしてよい。

### 1.4 DeFoG (ICML 2025 oral) — 離散もマスク拡散より flow matching

[OpenReview](https://openreview.net/forum?id=KPRIwWhqAZ) /
[GitHub](https://github.com/manuelmlmadeira/DeFoG)

グラフ生成向け**離散 flow matching**。グラフの置換対称性を保ちつつ、
**拡散系の 5-10% のステップ数**で同等以上の品質。学習とサンプリングの
スケジュールを分離できる(後から反復数・スケジュールを変えられる)。

含意: 案B の離散成分(type / fix_ref)は D3PM 系マスク拡散でなく **離散FM** に
すると、連続座標(FM)と同一フレームワークに揃い、実装も推論も軽くなる。

### 1.5 AR 系の近況 — 死んではいないがデータを食う

- **BrepARG** (2026): 幾何+トポロジ統合トークン列の decoder-only AR。学習 1.2日/
  推論 1.5s で BrepGen 比 2-5 倍高速
- **BrepGPT** (2025): Voronoi Half-Patch 単位の AR
- **CMT** (2025): MAR(マスク AR)でエッジ/面 latent を生成し cross-attention で隣接予測

含意: AR は 100K 規模データ+強いトークン設計なら今も第一線。しかし当方の失敗
(2.3K データでの exposure bias)は規模の問題であり、**小データでは反復精緻化+
強い帰納バイアスが有利**という当方の判断と矛盾しない。

## 2. 3路線の再評価

| 観点 | C (純AR) | AB (計画+AR) | B (グラフ精緻化) |
|---|---|---|---|
| frontier との整合 | BrepARG 系だがデータ50倍必要 | CLR-Wire と同型(裏付け強) | Flatten/TG-Diff/DeFoG と同型(裏付け最強) |
| 順序問題 | 直撃(単発67.9mmで収束) | AR部分に残存 | **概念ごと消滅** |
| トポロジ保証 | 文法マスクのみ | latent 任せ(CLR-Wireでも81.8%) | 2段分離+共有頂点で構成的 |
| 小データ適性 | 弱(exposure bias) | 中 | **強(置換不変=順序を学習不要、参照=座標を学習不要)** |
| 実装コスト | 済 | 済(評価中) | 新規(ただし FM/OT/count head 資産流用) |

**結論: 案B を本命として実装に進む判断は最新動向に照らして正しい。**
ただし調査を受けて設計を3点アップグレードする。

## 3. 案B設計へのアップグレード(plan_b_graph_design.md への修正)

1. **頂点トークンの導入(溶接の構成化)** — Flatten The Complex 方式。
   生成対象 = 頂点集合 V(K_v≤64、座標のみ)+ エッジ集合 E(K_e≤192、
   型+**端点は V への離散参照**+ARC中間点)。端点共有は参照の一致として
   表現の時点で成立し、decode 溶接が不要になる。fix_ref と同じ機構で実装可
   (参照の離散生成は 3. の離散FMで扱う)
2. **2段分離の正式化** — MVP の「型を先に」を格上げし、
   Stage-T: 条件 → (K_v, K_e, 型列, 接続参照) を離散FMで生成 /
   Stage-G: 骨格条件付きで頂点座標+中間点を連続FMで生成。
   Stage-T が正しければトポロジは 100% 保証(TG-Diff の思想)
3. **離散はマスク拡散でなく離散FM(DeFoG)** — 連続と同一フレームワーク、
   ステップ数 5-10%、学習/サンプリング分離

リスク再評価: 旧リスク1(溶接失敗)は構成的に解消。新リスクは「参照の離散生成が
座標なしで正しく出るか」— ただし Stage-G が骨格条件付きなので、参照が多少
不自然でも座標側が吸収する余地がある。Stage-T 単体の妥当率を最初に測る。

## 4. 残る当方固有の課題(論文が答えてくれないこと)

- 全 frontier 論文は 100K+ サンプルで学習している。**2.3K での成立性は当方で
  実証するしかない**。置換不変+参照化で「学習しなくてよいこと」を最大化したのが
  当方の対抗策
- 条件が「締結点2つ」という極端な疎条件である点は CLR-Wire の 1K 点群より遥かに
  情報が薄い。多様性(同条件から複数の正解)を許す評価系を維持すること
- 板金固有の拘束(曲げ半径・フランジ角)はどの論文にもない。将来の条件追加は
  cross-attention トークン追加で対応(全案共通)

## 5. 出典一覧

- CLR-Wire: https://arxiv.org/abs/2504.19174 (SIGGRAPH 2025)
- Flatten The Complex: https://arxiv.org/abs/2601.17733
- TG-Diff: https://arxiv.org/abs/2607.21928
- DTGBrepGen: https://www.alphaxiv.org/abs/2503.13110
- BrepForge: https://arxiv.org/abs/2605.19411
- BrepGiff (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_BrepGiff_Lightweight_Generation_of_Complex_B-rep_with_3D_GAT_Diffusion_CVPR_2025_paper.pdf
- DeFoG (ICML 2025 oral): https://openreview.net/forum?id=KPRIwWhqAZ
- BrepGen: https://arxiv.org/abs/2401.15563 (基準点)
- 生成パラダイム統一の理論: https://arxiv.org/pdf/2504.06416 (AR=拡散の特殊ケース)
