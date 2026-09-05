# 文献サーベイと提案メモ

> 統合日 2026-09-05。以下は元文書を順に**そのまま**収録(見出し番号は元のまま。`KB 21.24` のような参照はこのファイル内を検索)。
> 収録元: `architecture_survey_202608.md`, `fable5_alternative_generation_paradigms.md`, `constraint_geometric_attention_wireframe_proposal.md`, `mechanical_attention_data_collection_proposal.md`


---

<!-- 元文書: architecture_survey_202608.md -->

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


---

<!-- 元文書: fable5_alternative_generation_paradigms.md -->

# 板金部品形状生成 — 全く別のアプローチによる生成思想の提案

Date: 2026-07-02
Author: fable5 (claude-fable-5)
Status: 設計思想の提案(現行 scaffold AE 路線とは独立した代替案の棚卸し)

## 0. この文書の位置づけ

現行路線(点群 AE → scaffold prior → mesh IR)とは**別の生成思想**で「拘束点 → 板金中立面」タスクに挑む場合の選択肢を、既存の transformer / diffusion / LLM 系アーキテクチャの系譜を踏まえて整理する。各案は「思想」「アーキテクチャ系譜」「板金適合性」「データ効率」「リスク」「最小PoC」で評価する。

### 評価の前提条件(このプロジェクト固有)

1. **データが極端に少ない**: 2アセンブリ、使用可能パーツ最大65。ImageNet 時代の感覚は通用せず、**事前学習済み priors をどれだけ借りられるか**が支配的な選択基準になる
2. **出力の最終要件は CAE mesh**: mm精度、boundary chain、要素品質、拘束点の厳密充足
3. **板金というドメインの特殊性**: ①ほぼ可展面(developable)の集合、②「平板 + 曲げ列」という自然なプログラム表現を持つ、③bbox内の材料占有率が極小(voxel系が非効率)、④左右勝手違い・パーツファミリという構造的類似性
4. 拘束点(締結/取付/支持)+ envelope + 板厚が条件

### 結論の先出し

| 案 | 一言 | 推奨度 |
|---|---|---|
| A. 展開図空間生成 | 板金を2D問題に還元し、製造可能性を構造で保証 | ★★★ 最有力の対抗馬 |
| B. CAD プログラム合成(LLM) | 事前学習済みコード LLM の priors を最大限借りる | ★★★ データ最小で唯一成立しうる |
| C. Latent set diffusion + UDF | 3D生成の現代標準。既存 softmin UDF 資産と接続 | ★★ 中期の本命候補 |
| D. 検索 + 神経変形 | 最強のベースライン。生成と呼べる下限を定義する | ★★★ 必ず併走(投資は最小) |
| E. 最適化駆動生成(FEM in loop) | 「分布を学ぶ」のをやめ「拘束から最適化する」 | ★ 思想として重要、工数大 |
| F. AR mesh token 生成 | mesh IR を直接出せる唯一の案だがデータが桁不足 | ★ 長期のみ |

---

## A. 展開図空間生成 — 「板金は3D形状ではなく、2D輪郭+折りプログラムである」

### 思想

板金部品の本質は3D曲面ではない。**平板の展開図(2D輪郭+穴)と、曲げ線・曲げ角・曲げRの列**である。生成を3D空間で行うから難しいのであって、板金の自然座標系=展開図空間で生成すれば:

- 問題が **2D生成 + 折りプログラム生成** に分解される
- 生成物は**構成的に製造可能**(展開図として成立している=平板から作れる)
- 3D形状は展開図+曲げ列から決定論的に folding して得られる(微分可能 folding も実装可能)
- 現行路線が苦しんでいる「薄板が3D空間で占める測度ゼロ性」が消える — 展開図空間では板は面積を持つ普通の2D領域

### アーキテクチャ系譜

- 2D輪郭生成: 2D diffusion(U-Net/DiT)を SDF ラスタまたはポリライン列に適用。ControlNet 型の条件付けで「展開図上に投影した拘束点マップ」を条件画像として注入
- 曲げプログラム: 小さな autoregressive transformer(曲げ線パラメータ列は数トークン〜数十トークン)
- 系譜: SkexGen / DeepCAD の sketch-and-extrude 分解の板金版。「sketch = 展開図、operation = 曲げ」

### 板金適合性

- **最高**。ドメイン知識をデータではなく表現に焼き込む唯一の案。フランジ・リブ・エンボスは展開図上の領域+曲げ線として明示的に表現される
- 拘束点の扱い: 拘束点を展開図上へ逆投影して2D条件マップにする。曲げ後の3D位置と拘束点の一致は folding が決定論的なので厳密に検証可能

### データ効率

- 2D表現は増強が効く(回転・スケール・輪郭の局所摂動)。65パーツでも2D なら現実的な射程に入る
- 事前学習済み 2D diffusion バックボーンの転用も可能(形状SDF画像への fine-tune)

### リスク・課題

1. **展開図の教師データが現状ない**。中立面 STEP からの自動展開(unfolding)パイプラインが必要 — 可展面近似+曲げ線検出。これ自体が板金では確立技術(LVD/Amada系CAMが日常的にやっている)だが、実装工数はかかる
2. 絞り(draw)成形など非可展要素を持つパーツは表現できない(BIWでは一定数存在する)。「可展パーツのみ」への適用範囲限定を最初に受け入れる必要がある
3. 曲げ順・干渉は当面無視してよい(中立面幾何のみが目的なら折り順は不要)

### 最小PoC(2〜3週間規模)

1. 既存65パーツから crease 検出(既に typed scaffold で crease を扱っている)→ 平面領域分割 → 単純パーツ10個を手動/半自動で展開図化
2. 展開図 SDF ラスタ(例 256×256)+ 曲げ線チャネルの表現を設計
3. 小型 2D diffusion を拘束点マップ条件付きで訓練し、folding して Chamfer を現行 AE と比較

---

## B. CAD プログラム合成 — 「形状を生成するな、形状を作るコードを生成せよ」

### 思想

65パーツしかないなら、幾何分布をゼロから学ぶのは原理的に無理筋。代わりに、**すでに数兆トークンで訓練された code LLM の事前知識を借り**、板金部品を小さな DSL プログラム(またはCadQuery/FreeCAD Python)として生成する。

```text
拘束点 + envelope + 板厚 (テキスト/JSON)
  -> LLM (fine-tuned or few-shot + agentic loop)
  -> 板金DSL プログラム (base_plate, flange, rib, hole, bend...)
  -> 決定論的実行 -> 中立面 B-Rep/mesh
  -> 拘束充足チェッカー -> 不合格なら誤差情報を添えて再生成 (execute-verify-repair)
```

### アーキテクチャ系譜

- DeepCAD / SkexGen / HNC-CAD(CAD操作列の AR transformer)、Text2CAD、および近年の LLM+CAD カーネルの agentic 合成
- 決定的な違い: 専用モデルを訓練するのではなく、**汎用 code LLM + 実行フィードバックループ**を使う。65パーツは fine-tune データではなく few-shot 例+評価セットとして使う

### 板金適合性

- 板金 DSL は驚くほど小さい(平板・フランジ・曲げ・穴・エンボスで実用部品の大半を張れる)。DSL が小さいほど LLM 合成の成功率は上がる
- 出力がパラメトリックなので、**mm精度・板厚整合・曲げRが構成的に正確**。CAE mesh 化は既存メッシャに委譲できる(生成問題からメッシング問題を分離)
- 拘束充足は実行→測定→修正のループで厳密に到達可能

### データ効率

- **全案中最高**。訓練データほぼゼロで開始できる。65パーツは DSL で再現できるかの検証セットに回す
- 「既存65パーツを DSL で書き直せるか」自体が DSL の表現力テストになり、失敗パーツが DSL の拡張要件を教えてくれる

### リスク・課題

1. 有機的な自由曲面(絞り形状、複雑なビード)は DSL で表現しにくい — A案と同じ適用範囲の限定
2. LLM の幾何推論は空間的に粗い。実行フィードバックループの設計(誤差の言語化)が品質を決める
3. 推論コスト(生成1回あたり複数回の LLM 呼び出し)
4. 「学習」が蓄積しにくい — ループの改善はプロンプト/DSL/チェッカーの改善であり、勾配ではない

### 最小PoC(1週間規模で判定可能)

1. 板金 DSL v0 を定義(5プリミティブ程度、実行系は build123d/CadQuery)
2. 既存パーツ10個を人手で DSL 化 → few-shot 例に
3. val パーツの拘束点+envelope を入力し、frontier LLM に execute-verify-repair ループで生成させ、実パーツとの Chamfer と拘束充足距離を測定
4. **この PoC は訓練ゼロなので、現行路線と完全に並行して1人日単位で試せる**

---

## C. Latent set diffusion + 陰関数(UDF)デコーダ — 3D生成の現代標準に乗る

### 思想

現行 AE の「点群を直接出す」代わりに、3D生成研究の現在の主流である **latent set + diffusion + 陰関数場** に乗り換える:

```text
形状 -> latent token set (数百 token) への VAE 符号化
拘束点 + metadata -> conditional diffusion/flow が latent set を生成
latent set -> UDF (unsigned distance field) デコーダ -> 中立面抽出
```

### アーキテクチャ系譜

- 3DShape2VecSet / Michelangelo / CLAY / TripoSG 系の latent vecset diffusion
- 中立面は**開曲面**なので SDF ではなく **UDF** が正しい場の選択 — そして本リポジトリには `softmin guidance field` の資産(UDF 系の検証・可視化コード)が既にあり、feat/softmin-guidance-poc ブランチの知見がそのまま接続する
- ノイズスケジュールは flow matching を推奨(現行 diffusion の後継としてほぼ上位互換)

### 板金適合性

- UDF は薄板・開境界を自然に表現(occupancy/SDF の「内外」が定義できない問題を回避)
- 表面抽出(UDF→点群/メッシュ)は勾配追跡か marching が必要で、境界チェーンの明示性は現行 scaffold 路線より劣る — boundary は別ヘッドで補う必要がある
- 多様性・条件付けは diffusion の土俵なので自然に扱える

### データ効率

- **最大の弱点**。vecset diffusion 系は数千〜数百万形状で訓練されるのが常識であり、65パーツではVAE段から過学習する
- 現実解は2つ: ①公開CADコーパス(ABC dataset 等の薄肉サブセット)で事前学習して板金 fine-tune、②現行 AE の latent を流用して diffusion だけ小さく訓練(= 現行路線の Q5-2 と実質合流)

### リスク・課題

- 事前学習コーパスと BIW 板金のドメインギャップ
- mm 精度への到達は場の解像度に依存し、latent 容量とのトレードオフ

### 最小PoC

- ABC dataset から薄肉パーツ 5k〜10k を抽出し、UDF-VAE を事前学習 → 65パーツで fine-tune → 再構成品質が現行 AE を超えるかをまず AE 土俵で比較(diffusion はその後)

---

## D. 検索 + 神経変形 — 「65パーツしかないなら、生成ではなく編集をせよ」

### 思想

データが2桁不足しているとき、最も誠実なアプローチは分布学習を放棄することである:

```text
拘束点配置 -> 拘束コンフィグ埋め込みで最近傍の既存パーツを検索
  -> 拘束点対応を取り、cage/RBF/神経変形場でワープ
  -> 拘束点を厳密充足するよう射影
```

### アーキテクチャ系譜

- 古典: RBF / cage-based deformation。神経系: Neural Cages、deformation field MLP
- 検索埋め込みは拘束点集合の PointNet 程度で足りる

### 板金適合性・データ効率

- 変形は既存パーツの製造妥当性を大部分保存する(大変形しない限り)
- **訓練データ要求が最小**で、新パーツ追加が即座に検索プールの改善になる(継続的に強くなる運用特性)

### リスク

- 位相の異なる形状は原理的に出せない(穴の数・フランジ本数が変わる拘束には無力)
- 「生成」としての学術的な面白さはないが、**実務MVPとしては最短**

### 位置づけの提案

これは対抗馬というより **全案共通の必須ベースライン**である。fable5 前回レスポンスの「train-NN 検索ベースライン」の強化版であり、どの生成モデルもまず「検索+変形に勝てるか」で審査されるべき。3〜5人日で実装できる。

---

## E. 最適化駆動生成 — 「分布を学ぶな、拘束から最適化せよ」

### 思想

生成モデルの発想を捨て、拘束点・荷重・envelope から**形状最適化**として中立面を解く。学習済み生成 prior は「板金らしさ」の正則化項としてのみ使う:

```text
パラメトリック板金表現 (展開図 or scaffold) を最適化変数に
目的関数 = 拘束充足 + コンプライアンス(FEM) + 質量 + 板金らしさ(diffusion prior の score)
  -> 勾配ベース最適化 (微分可能 folding / 微分可能 FEM or 随伴法)
```

### アーキテクチャ系譜

- トポロジー最適化(SIMP)の板金版 + DreamFusion 系の score distillation(2D/3D diffusion prior を最適化の正則化に使う発想)
- 微分可能シミュレータ(shell FEM)との結合

### 評価

- 思想として最も「設計自動化」の最終目標に近い(既存部品の模倣ではなく、力学的に正しい新形状を出す)
- ただし工数・数値安定性・微分可能 shell FEM の整備コストが大きく、**現段階の MVP には過大**。A案(展開図表現)が成立した後に、その上の最適化層として載せるのが正しい順序

---

## F. Autoregressive mesh token 生成 — mesh IR を直接吐く

### 思想

MeshGPT / MeshXL / PolyGen 系: メッシュの頂点・面を語彙化し、LLM と同じ next-token prediction で **connectivity 込みのメッシュを直接生成**する。CAE mesh IR という最終要件に最短距離で並ぶ唯一の案。

### 評価

- 系譜的には magnetic だが、これらは数万〜数十万メッシュで訓練される。65パーツでは論外であり、公開コーパス事前学習をしても「BIW板金の quad-dominant CAE メッシュ」というドメインからの距離が遠い
- **今は選ばない。** ただし「メッシュをトークン列として表現する」語彙設計(頂点量子化、面の順序正規化)は、将来 mesh IR を設計する際の参考として一級の資料群である

### アンチ推奨の明記: voxel / occupancy 系

dense voxel diffusion、occupancy transformer 系は、板金の材料占有率(bbox の数%)に対して計算・容量とも壊滅的に非効率。元レポート §2 の判断通り、今後も除外してよい。

---

## G. 横断的な提言

### G.1 どれを試すか — 推奨ポートフォリオ

データ量が生成路線全体の律速である以上、**「priors を借りる度合い」の異なる案を小さく並走**させるのが正しい:

1. **今週から**: B案 PoC(訓練ゼロ、1人日〜)+ D案ベースライン(3〜5人日)。この2つは現行 scaffold 路線と一切競合しない
2. **今月**: A案の展開図化パイプライン調査(既存 crease 検出資産の流用可否)。10パーツ展開できたら PoC 続行
3. **現行路線は継続**: scaffold 蒸留 + アンカー疎化(前レスポンス参照)は上記と独立に価値がある
4. **C案は待機**: ABC 等の事前学習コーパス整備ができた時点で着手。E/F は当面凍結

### G.2 判定基準を先に固定する

全案を同じ土俵で審査するため、次の3指標を共通評価にする:

- 拘束充足距離(max/mean)
- 実パーツとの較正済み Chamfer(4096点 target)
- **D案(検索+変形)との差分** — これに勝てない案は棄却

### G.3 本質的な認識

現行路線もここでの全代替案も、最後は同じ壁に当たる: **65パーツでは「生成」は成立しない**。中期的に最も投資対効果が高いのは、モデルではなく:

1. 未処理3アセンブリ(A0072600081 / 367 / 529)の mid/fill 抽出完了
2. 展開図・DSL・scaffold などの「学習不要で作れる中間表現」パイプラインの整備
3. 公開CADコーパスからの薄肉パーツ採掘

である。アーキテクチャの選択は、この3つが進んだ後のほうが遥かに安全に行える。

## 一文要約

板金は「2D展開図+曲げプログラム」という自然なプログラム表現を持つ稀有なドメインであり、65パーツという制約下では、3D分布をゼロから学ぶ現行路線の対抗馬として、(A) 展開図空間での2D生成と (B) 事前学習済み code LLM による DSL プログラム合成 — すなわち priors を表現とモデルの両面で「借りる」思想 — が最も勝ち目のある別アプローチである。


---

<!-- 元文書: constraint_geometric_attention_wireframe_proposal.md -->

# Proposal: Constraint Geometric Attention for Wireframe Generation

Date: 2026-07-09
Author: Codex, for Claude/Fable handoff
Status: Proposal for the next experiment after the current wireflow baseline

## 1. Summary

The final target is:

```text
constraint points + nearby assembly context
  -> predicted part wireframe
  -> shell mesh
  -> eventually B-Rep
```

The current wireflow experiment asks a model to move random points inside a bbox until they match a typed wireframe. That is useful as a diagnostic, but the next structural step should be:

```text
constraint points
  -> constraint relation graph
  -> geometric wireframe graph
```

This proposal adds a first-stage **geometric attention model**. It predicts how geometrically close or related two constraint points should be on the eventual sheet-metal part, using only information available at generation time.

The first MVP intentionally avoids CATIA CAE stress/strain fields because they may require high manual/setup cost. Stress, strain, load-path, and topology-optimization-style signals should be future extensions. The first experiment should start with **geometric attention** only.

## 2. Minimal Input And Output

### Input

For each constraint point `i`:

```text
p_i: 3D coordinate in part or assembly frame
n_i: local surface normal at the constraint point
optional later:
  joint type, bolt/weld/clip/mount type
  hole/washer radius
  support/load tag
  nearby part occupancy or clearance field
  assembly-level part identity/context
```

For the first experiment, keep the model as small and clean as possible:

```text
node_i = [normalized xyz, normal xyz]
```

Normalize coordinates by part/condition bbox center and scale. Preserve the world-to-part transform metadata so later joints and generated geometry remain consistent.

### Output

A symmetric pairwise attention matrix:

```text
A_ij in [0, 1]
```

Interpretation:

```text
A_ij high  = constraint points i and j are geometrically close/related on the final sheet-metal surface
A_ij low   = points are likely separated by long surface distance, different regions, or weak direct geometric coupling
```

This is not the transformer's internal attention. It is an explicit predicted artifact: a **constraint relation graph** that can be evaluated, visualized, and passed into the wireframe generator.

## 3. Supervision Targets

The target score should be derived from completed parts where the true midsurface/wireframe is known.

Preferred target if practical:

```text
d_geo(i, j) = shortest path distance between constraint points on the midsurface mesh
score(i, j) = exp(-d_geo(i, j) / tau)
```

If direct midsurface geodesic distance is slow or noisy, use staged approximations:

1. **Wireframe graph distance**
   - Project each constraint point to the nearest typed wireframe entity.
   - Compute shortest path distance along the wireframe graph.
   - Good for a first implementation because the current wireframe identification is already strong.

2. **Patch/region relationship**
   - Same patch or same loop neighborhood: positive.
   - Across one bend line: medium.
   - Across many bends or disconnected feature regions: low.

3. **Euclidean + normal heuristic teacher**
   - Use as a baseline only, not as final evidence.
   - Example: nearby points with compatible normals get higher score.

The model should report both regression and ranking metrics. The real question is not exact score calibration at first; it is whether the model ranks geometrically relevant point pairs above irrelevant ones.

## 4. Why This Helps Wireframe Generation

Constraint points alone are underdetermined. A bbox and a few holes/mounts do not specify where loops, bend lines, and surface frames should go. The geometric attention matrix provides the missing relational prior:

```text
which constraints likely belong to the same sheet region
which constraints should be connected by material
which constraints are separated by long folded paths
which groups define local loops, ribs, or mounting regions
```

Then the wireframe generator can condition on:

```text
nodes: constraint points with normals
edges: predicted A_ij scores
global: bbox / scale / optional surrounding parts
```

This changes the problem from:

```text
generate a wireframe from independent points
```

to:

```text
generate a wireframe from a constraint relation graph
```

That is a much better fit for sheet metal because the final wireframe is itself a geometric graph/cell complex.

## 5. Recommended Model

Start with a pairwise geometric graph model:

```text
Input constraint nodes:
  xyz, normal

Pair features:
  delta xyz
  distance
  normal dot product
  projected distance along normals/tangent plane
  relative normal-angle features

Encoder:
  small Transformer or GNN over constraint nodes

Pair head:
  h_i, h_j, pair_features_ij -> score A_ij
```

Keep invariance in mind:

- Translation invariance: subtract bbox center.
- Scale robustness: divide by bbox diagonal or max extent.
- Rotation robustness: use relative pair features and normal dot products.
- Do not overfit to global vehicle coordinates unless assembly orientation is intentionally part of the condition.

Candidate losses:

```text
L_distance = SmoothL1(predicted_distance, target_geodesic_distance_normalized)
L_score    = BCEWithLogits(predicted_score, target_close_pair)
L_rank     = InfoNCE / pairwise ranking loss for top-k related neighbors
```

A practical first version can train only a score head with BCE/ranking labels:

```text
positive pairs = top-k nearest pairs by true geodesic or wireframe distance
negative pairs = far pairs sampled from the same part
```

## 6. Evaluation

Evaluate the attention model before integrating it into wireframe generation.

Metrics:

```text
top-k recall:
  For each constraint point, does predicted top-k include the true geodesic-nearest neighbors?

Spearman / Kendall rank correlation:
  Does predicted score order match true geometric distance order?

AUC / average precision:
  Can the model separate close-pair vs far-pair labels?

distance bucket accuracy:
  near / medium / far classification
```

Baselines:

```text
Euclidean distance only
Euclidean distance + normal dot
nearest-neighbor by bbox-normalized coordinate
oracle score from true geodesic/wireframe distance
```

The geometric attention model is useful only if it beats Euclidean+normal baselines on held-out parts. If it does not, feed the heuristic score directly to the wireframe generator and avoid adding a weak learned component.

## 7. Integration Into Wireframe Generation

After the attention model is validated, use `A_ij` in the wireframe generator in three ways.

### 7.1 Attention Bias

Pass `A_ij` as a bias into cross-attention between generated wireframe tokens and constraint tokens.

```text
attention_bias_ij = alpha * logit(A_ij)
```

This is the least invasive integration path for the current wireflow work.

### 7.2 Candidate Edge Prior

Use high-score pairs as candidate material paths or patch-local relationships.

```text
high A_ij -> likely same local sheet region or connected feature group
low A_ij  -> avoid direct generated connections unless other evidence supports them
```

This is especially useful when moving from point generation to geometric wireframe graph generation.

### 7.3 Patch Graph Prior

Cluster constraints by predicted relation score:

```text
constraint relation graph
  -> local groups
  -> likely patch/loop anchors
  -> outer/hole/bend/surface_frame generation
```

This supports the longer-term direction:

```text
constraint relation graph -> geometric wireframe -> patch graph -> shell mesh
```

## 8. Future Mechanical Attention

Do not start with full stress/strain fields. Keep them as planned extensions:

1. **CATIA CAE stress/strain attention**
   - Use FEA responses to derive pairwise compliance, load transfer, or stress propagation scores.
   - More physically meaningful, but higher setup cost.

2. **Topology optimization load-path attention**
   - Run coarse topology optimization or simplified shell/truss optimization.
   - Extract material/load-path skeletons as soft constraints.

3. **Hybrid score**
   - Combine:

```text
A_total = w_geo * A_geometric
        + w_mech * A_mechanical
        + w_context * A_surrounding_parts
```

The first geometric attention model should be designed so these additional channels can be added later without changing the downstream wireframe interface.

## 9. Proposed First Implementation Plan

### Phase 0: Dataset Builder

Create a dataset from assemblies with reliable joints and wireframes:

```text
input:
  constraint point xyz + local normal

target:
  pairwise geometric score from wireframe graph distance or midsurface geodesic distance
```

Start with `A0072600002_AllCATPart` and `A0072601285_AllCATPart`. Exclude assemblies without reliable annotations for the first training run.

### Phase 1: Baseline And Learned Attention

Implement:

```text
baseline_geometric_score.py
train_constraint_attention.py
evaluate_constraint_attention.py
```

Report:

```text
Euclidean baseline
Euclidean+normal baseline
learned score
oracle score
```

### Phase 2: Wireflow Conditioning

Use the learned score as an attention bias or extra pair feature in the current wireflow decoder. Compare:

```text
wireflow without A_ij
wireflow with heuristic A_ij
wireflow with learned A_ij
oracle A_ij upper bound
```

### Phase 3: Geometric Wireframe Graph Generator

Move away from unordered random point convergence and generate an explicit geometric graph:

```text
vertices
curves/edges
typed loops
patch incidence
```

The constraint attention graph becomes the natural conditioning structure for this generator.

## 10. Key Guardrails

- Do not call the score "mechanical" until it uses real mechanical supervision or simulation.
- Keep `A_ij` inspectable. Save it as JSON/CSV and visualize it as colored edges between constraint points.
- Compare against Euclidean+normal before claiming learning value.
- Treat exact geodesic regression as secondary; top-k relation recovery is the first success criterion.
- Avoid leakage: compute target distances only from the training target geometry, never from validation geometry at inference time.
- Preserve part-local/world transforms for all constraint coordinates and normals.

## 11. First Success Criterion

The first milestone is not improved wireframe generation. It is:

```text
Given only constraint point coordinates and local normals,
the model predicts geometrically related constraint pairs
better than Euclidean+normal baselines on held-out parts.
```

If this succeeds, the predicted relation graph should be integrated into wireframe generation. If it fails, the project still gains a useful conclusion: current constraint-point information is insufficient and needs surrounding part context, joint types, or mechanical signals.


---

<!-- 元文書: mechanical_attention_data_collection_proposal.md -->

# Proposal: Mechanical Attention Data Collection for Constraint Graphs

Date: 2026-07-09
Author: Codex, for Claude/Fable handoff
Status: Proposal for future extension after geometric attention

## 1. Summary

The current direction is:

```text
constraint points + local normals
  -> geometric attention / constraint relation graph
  -> geometric wireframe
  -> shell mesh
  -> eventually B-Rep
```

This document proposes the next extension: **mechanical attention**.

The key idea is not to use stress/strain fields as direct inference-time inputs. At inference time the target part shape does not exist yet, so a true stress field cannot be computed. Instead, existing completed parts should be used to generate mechanical supervision:

```text
completed part + constraints + CAE
  -> per-load response fields
  -> pairwise mechanical attention matrix M_ij
  -> train a model to predict M_ij from constraint geometry/context
```

Then the generation pipeline becomes:

```text
constraint points + normals + surrounding context
  -> geometric attention G_ij
  -> predicted mechanical attention M_ij
  -> wireframe graph generation
```

Start with geometric attention first. Mechanical attention should be added after the geometric attention dataset/model is inspectable.

## 2. Why Pairwise Mechanical Attention

Raw stress fields are high-dimensional, mesh-dependent, boundary-condition-dependent, and expensive to generate. For wireframe generation, the most useful signal is usually not the full field itself, but the relation between functional points:

```text
If load enters at constraint i,
which other constraints and sheet regions participate in carrying it?
```

This can be stored as:

```text
M_ij = mechanical relatedness score between constraint i and constraint j
```

High `M_ij` means:

- force or displacement response transfers strongly between the two constraints
- the same material path or patch likely participates in both load cases
- the generated wireframe should probably preserve a strong sheet connection between their regions

Low `M_ij` means:

- the two constraints are mechanically weakly coupled
- direct material/feature connection is less likely unless required by geometry or packaging

## 3. First Data To Collect

For each completed part with reliable constraints and mesh:

```text
constraint i:
  xyz position
  local surface normal
  optional local tangent basis t1, t2
  joint type / washer radius / support type if available
```

For each constraint `i`, run unit load cases.

### Minimal Load Cases

Start with one load case per constraint:

```text
case i_N:
  apply unit force at constraint i along local normal n_i
```

This is the lowest-effort first dataset. It is enough to test whether mechanical attention adds any information beyond geometric attention.

### Preferred Load Cases

If CATIA CAE automation cost is acceptable, use three orthogonal directions:

```text
case i_N:
  unit force along normal n_i

case i_T1:
  unit force along tangent t1_i

case i_T2:
  unit force along tangent t2_i
```

Sheet metal often carries important in-plane/shear loads, so tangent loads will eventually be important.

### Optional Later Cases

Later, add:

```text
unit moment around n_i / t1_i / t2_i
pressure-like local patch load
realistic assembly load cases
thermal or vibration cases if relevant
```

Do not start here. First prove value with unit force cases.

## 4. Boundary Condition Protocol

Boundary conditions must be standardized. Otherwise the learned signal will mostly reflect inconsistent CAE setup choices.

Recommended first protocol:

```text
For load at constraint i:
  apply unit force at i
  all other constraints use the same standardized support model
```

Candidate support models:

1. **Fixed supports at all other constraints**
   - easiest to set up
   - stable
   - may be too stiff and overemphasize local peaks

2. **Spring supports at all other constraints**
   - preferred if practical
   - use the same translational/rotational stiffness for all parts
   - closer to assembly-like compliance

3. **Hybrid**
   - fixed normal direction, spring tangential directions
   - useful if rigid-body modes are troublesome

The support protocol must be written into the dataset manifest.

## 5. Fields To Export

For each load case, save both full fields and reduced summaries.

### Full Fields

Save these if CATIA/export allows it:

```text
node displacement vector u
element or node von Mises stress
principal stress values/directions
strain energy density
shell membrane/bending stress components if available
```

The most valuable full-field quantity is usually:

```text
strain energy density
```

Stress peaks can be noisy near point loads, holes, and mesh singularities. Strain energy density is a better load-path signal because it indicates where the structure is doing mechanical work.

### Reduced Per-Constraint Responses

For every load case at `i`, collect responses near every constraint `j`:

```text
u_j:
  displacement vector at/near constraint j

u_j_projected:
  displacement projected onto n_j, t1_j, t2_j

reaction_j:
  reaction force/moment at supported constraint j if available

energy_near_j:
  sum/mean/max strain energy density in a local neighborhood around j

stress_near_j:
  robust p95 von Mises stress near j, not raw max
```

Use local neighborhoods such as washer radius, fixed mm radius, or k-nearest shell elements. Record which definition is used.

## 6. Constructing Mechanical Attention Scores

Several pairwise scores should be generated and compared.

### 6.1 Compliance Response

```text
C_ij = norm(displacement at j under unit load at i)
```

or direction-aware:

```text
C_ij_ab = displacement of j along direction b under unit load at i along direction a
```

This approximates how much motion at one constraint influences another. It is close to a boundary compliance/influence matrix.

### 6.2 Reaction Transfer

```text
R_ij = norm(reaction at j under unit load at i)
```

This asks where the applied load is supported. It is often more intuitive for fixture/fastener relationships.

### 6.3 Strain Energy Near Target Constraint

```text
E_ij = strain energy near j under unit load at i
```

This is a local participation score.

### 6.4 Field Overlap / Shared Load Path

For each unit load case `i`, normalize the strain energy density field:

```text
e_i(x) = normalized strain energy density under load at i
```

Then:

```text
O_ij = cosine_similarity(e_i, e_j)
```

This is likely the best load-path relatedness score. It says whether two constraints activate similar material regions, even if they are not adjacent in Euclidean space.

### 6.5 Final Mechanical Attention

Do not choose one formula too early. Save multiple channels:

```text
M_ij = {
  compliance_response,
  reaction_transfer,
  target_neighborhood_energy,
  strain_energy_field_overlap
}
```

The wireframe generator can later use one channel, a learned combination, or a multi-channel edge feature.

## 7. Normalization

Mechanical values must be normalized for cross-part learning.

Recommended:

```text
coordinate scale: bbox diagonal or max extent
force scale: unit force, fixed for all parts
thickness: record explicitly; normalize stiffness/energy by thickness if needed
material: record E, nu, density; start with a single material if possible
energy fields: normalize each load case by total strain energy
reaction scores: divide by applied force magnitude
compliance scores: divide by part scale / force
```

Also save raw values. Normalization choices will change.

## 8. Dataset Schema

Suggested artifact per part:

```text
mechanical_attention/<assembly>/<part_id>/
  constraints.json
  cae_manifest.json
  load_cases.json
  pair_scores.npz
  fields/
    case_000_energy.vtk or npz
    case_000_displacement.vtk or npz
    ...
```

### constraints.json

```json
{
  "part_id": "...",
  "coordinate_frame": "part_local",
  "constraints": [
    {
      "id": 0,
      "xyz": [0.0, 0.0, 0.0],
      "normal": [0.0, 0.0, 1.0],
      "tangent_1": [1.0, 0.0, 0.0],
      "tangent_2": [0.0, 1.0, 0.0],
      "joint_type": "bolt_or_unknown",
      "radius_mm": null
    }
  ]
}
```

### cae_manifest.json

```json
{
  "solver": "CATIA_CAE",
  "element_type": "shell",
  "mesh_size_mm": 5.0,
  "thickness_mm": 1.2,
  "material": {"E": 210000.0, "nu": 0.3},
  "support_protocol": "other_constraints_fixed",
  "load_protocol": "unit_normal_force",
  "notes": ""
}
```

### pair_scores.npz

Store arrays:

```text
compliance[n_constraints, n_constraints]
reaction_transfer[n_constraints, n_constraints]
energy_near_target[n_constraints, n_constraints]
energy_field_overlap[n_constraints, n_constraints]
valid_mask[n_constraints, n_constraints]
```

If directional loads are included:

```text
compliance[n_constraints, n_constraints, n_load_dirs, n_response_dirs]
reaction_transfer[n_constraints, n_constraints, n_load_dirs]
```

## 9. First Experiment

Before generating large data, run a small calibration study.

Recommended size:

```text
10 to 20 parts
normal-load-only cases
same material/thickness if possible
fixed or spring support protocol
```

Questions to answer:

1. Are the matrices stable and interpretable?
2. Does `M_ij` correlate with geometric attention `G_ij`?
3. Where does `M_ij` disagree with `G_ij`?
4. Do disagreements explain meaningful structural relationships?
5. Can `M_ij` predict same-patch, same-load-path, or wireframe graph distance better than Euclidean distance?

If the answer is no, do not scale CAE yet.

## 10. Integration With Current Geometric Attention Work

Claude is currently working on the constraint relation graph. The clean integration path is:

```text
Stage A:
  geometric attention only

Stage B:
  mechanical attention teacher from CAE on a small subset

Stage C:
  train a mechanical attention predictor:
    input: constraint xyz + normal + optional context
    target: M_ij from CAE

Stage D:
  wireframe generator conditions on:
    G_ij
    predicted M_ij
    optional oracle M_ij upper bound for analysis only
```

Important: oracle `M_ij` from actual target geometry/CAE must never be used at inference or validation generation except as an upper-bound diagnostic.

## 11. User Work Items

The user is best positioned to prepare CAE-side data and validate engineering assumptions.

### Must Do First

1. Select 10 to 20 representative parts with reliable constraint points and wireframes.
2. Confirm whether CATIA CAE can automate:
   - shell mesh generation
   - assigning thickness/material
   - applying unit load at a selected constraint point
   - fixing or spring-supporting other constraint points
   - exporting displacement, stress, and strain energy density
3. Decide the first support protocol:
   - fixed other constraints, or
   - spring-supported other constraints
4. Decide the first load protocol:
   - normal-only, or
   - normal + two tangent directions
5. Run 1 to 2 pilot parts manually and export results.

### Nice To Have

1. Record CATIA screenshots/results for one easy-to-understand part.
2. Confirm whether strain energy density can be exported directly.
3. Confirm whether shell membrane/bending stress components are available.
4. Check how point loads are applied: exact node, washer patch, RBE/spider, or distributed local patch load.
5. Prefer distributed washer/patch loads over singular point loads if setup cost is reasonable.

### Avoid At First

1. Do not run hundreds of parts before validating the schema.
2. Do not mix inconsistent boundary conditions.
3. Do not use raw maximum stress as the main target.
4. Do not start with complicated real-world combined loads.
5. Do not make mechanical attention a required input for the first geometric attention generator.

## 12. First Success Criterion

The first milestone is:

```text
From completed parts and standardized CAE unit load cases,
construct inspectable pairwise mechanical attention matrices
that reveal plausible load-transfer or shared-load-path relationships
between constraint points.
```

The second milestone is:

```text
Train a predictor that estimates those matrices from constraint point coordinates,
normals, and optional context better than geometry-only baselines.
```

Only after these pass should mechanical attention be integrated into wireframe generation.

## 13. References For Direction

- Graph neural networks have been used to predict displacement, stress, and strain fields on mesh-like structures, which supports the idea of mesh/CAE-derived mechanical supervision.
- Static condensation and reduced stiffness/compliance matrices provide a classical mechanical analogy: reduce the structure to relationships between boundary/interface degrees of freedom.
- Topology optimization and load-path design commonly use compliance and strain energy density, supporting strain-energy-based mechanical attention as a useful signal.
