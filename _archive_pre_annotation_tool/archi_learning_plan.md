# GHMR (archi.md) 技術要素整理 + 個人学習計画

> 作成日: 2026-06-20
> 対象文書: `archi.md` §0〜§0I（規範範囲のみ。§1以降の旧素案は非規範のため対象外）
> 対象コード: `biw_poc/src/**`
> 目的: ユーザー自身が GHMR アーキテクチャを「AIの要約を読んで分かった気になる」のではなく、
> 自力で説明・再導出・実装トレースできる状態まで理解すること。

---

## 0. この文書の使い方

- Part 1 は「何を理解する必要があるか」の棚卸し（技術要素マップ）。
- Part 2 は「どの順番で学ぶと前提知識が積み上がるか」の依存関係。
- Part 3 が本体の学習計画。Stage 0〜6 に分割し、各 Stage に
  **ゴール / 読む箇所（archi.md行番号 + コードファイル) / 自分の手を動かす演習 / 自己チェック問題**
  を持たせている。自己チェック問題に「コードを見ずに」答えられない場合はそのStageを終えたとみなさない。
- Part 4 は進捗記録用チェックリスト（このファイルに直接チェックを入れていく運用を想定）。

design.mdの方針（ユーザーの判断を尊重し、意思決定を支える情報整理を優先する）に従い、
本書はどのStageを選ぶか・どの順で進めるかを強制しない。状況に応じて並べ替えてよい。

---

## Part 1. 技術要素マップ

archi.md §0〜§0I に登場する技術要素を、性質ごとに6カテゴリに分類した。
各要素には archi.md 内の主な記載箇所（見出し）を付記する。

### A. 幾何処理・メッシュ数学の基礎

| 要素 | archi.md 該当箇所 | 既存コードでの対応物 |
|---|---|---|
| UDF (Unsigned Distance Field) と SDF の違い、なぜ中立面でUDFが必要か | §0.3, §0I.1 | `vecset_ae.py: QueryDecoder`, `dcudf_extract.py` |
| Marching Cubes / DCUDF抽出 | §0G | `dcudf_extract.py: DCUDFExtractor` |
| point-to-surface距離 vs point-to-vertex距離 | §0E.1 | `validate_refine.py`(頂点NN) vs `diagnose_holes.py`/`dcudf_reconstruct.py: surface_distances_true`(真の面距離) |
| メッシュのトポロジー記述（non-manifold, 自己交差, Euler標数, genus, connected component, boundary loop） | §0A.3-E, §0E.1 | `diagnose_outliers.py`, `verify_topology_patch.py` |
| メッシュ品質指標（最小角, aspect ratio, 反転, 面積比） | §0B.4 | （旧）R7-14面積膨張実測 |
| chord deviation（弦偏差）による誤差評価 | §0A.3-G「split選択」 | 新規（未実装） |
| 局所座標フレーム（接線/法線分解 `R_patch=[t1,t2,n]`） | §0A.3-D | `annotator.py: project_to_midsurface()` の法線/接線概念を拡張 |
| sizing field（局所目標辺長 `h_target(x)`）、growth ratio | §0A.3-G | 新規（`reconstruct.py`の一括`refine_rounds`を置き換える対象） |
| remeshing操作（split/collapse/flip）と前後条件・ヒステリシス | §0A.3-F, G | `reconstruct.py: subdivide_and_project()`（現行は一括split、これを置換する） |
| cut locus（UDFの非微分可能な特異点） | §0.3, §14のTD-16 | `dcudf_reconstruct.py`のコメント、CLAUDE.md §14根本原因分析 |

### B. 深層学習・幾何学習（GNN / Transformer / 表現学習）

| 要素 | archi.md 該当箇所 | 既存コードでの対応物 |
|---|---|---|
| PointNet系エンコーダ（FPS + kNN + mini-PointNet） | §0A.2 (Evidence Sampler の前段イメージ) | `vecset_ae.py: furthest_point_sampling, knn_gather_idx, MiniPointNet, PointTokenizer` |
| Self-Attention Encoder / Cross-Attention Decoder（VecSet方式） | §0A.3-A (FieldEvidenceProvider が包む対象) | `vecset_ae.py: TransformerEncoder, LatentCompressor, QueryDecoder, VecSetAE` |
| Fourier特徴によるNeRF式位置エンコーディング、spectral bias対策 | （archi.mdでは前提知識として暗黙、CLAUDE.md §14 R7-05が直接の出典） | `vecset_ae.py: FourierFeatures, PositionalMLP` |
| Edge-based Message Passing GNN（LocalMeshRefinerの中核） | §0A.3-D | 新規（未実装。最初の実装対象、§0H手順9） |
| 回転同変性 (SO(3) equivariance) のテスト設計 | §0A.3-D | 新規。archi_audit.md 第3版2.Aで「SO(2)面内回転も要検証」と指摘済み |
| truncated/clamped distance loss（DeepSDF/NDF方式） | （archi.mdでは§0B.2の損失設計の前提） | CLAUDE.md §14 R7-03, `train.py: weighted_losses()` |
| パッチ分割学習（overlap, halo, 継ぎ目整合） | §0A.3-C | 新規 |
| ランキング学習（唯一解ではなく操作後品質改善量を教師にする） | §0C.1 | 新規 |
| DAgger型データ収集（multi-round rollout） | §0C.1 | 新規 |
| OOD検出 / novelty detection / risk-coverage曲線 | §0A.3-B2 | 新規。CLAUDE.md §15 Deep Ensembles実験（不採用と判明済み）が反面教師 |

### C. 安全工学・状態機械設計

| 要素 | archi.md 該当箇所 |
|---|---|
| 辞書式最適化（lexicographic optimization） vs 重み付き総和 | §0B.1, §0A.3-G `rank(e)` |
| ハードゲート / モノトニックガード / fail-closed設計 | §0A.3-B, §0A.3-E |
| トランザクション的commit/rollback、原子的ロールバック | §0A.3-E |
| 状態機械と終端状態の設計（8状態、再開禁止） | §0A.5 |
| 監査ログのハッシュチェーン（append-only event sourcing） | §0A.4 |
| MANUAL_REVIEWの扱い（自動再開禁止、新規run必須） | §0A.5 |
| ReconstructionProfile（設定の単一情報源化） | §0.3.1, §0I.5 |
| 文書のnormative/historical区分（ガバナンス設計） | 冒頭 yaml frontmatter, §0I.6 |

### D. 統計的評価設計

| 要素 | archi.md 該当箇所 |
|---|---|
| paired bootstrap 95% CI、比較単位を part にする理由（seedではない） | §0E.2 |
| part-family holdoutとデータリーク防止（同一部品の近傍パッチを分割しない） | §0C.3 |
| 最小サンプル数とその妥当性（5 held-out parts / 3 families） | §0E.2。archi_audit.md第3版2.Cで統計的過小の懸念を指摘済み |
| 評価器自体のバージョン管理・再現性（sampling seed等もmanifestへ） | §0E.1 |

### E. CAEドメイン知識

| 要素 | archi.md 該当箇所 |
|---|---|
| シェル要素品質指標（aspect ratio, warpage, Jacobian等）とソルバー依存性 | §0B.4 |
| implicit/linear/explicit解析でのサイズ要件の違い | §0B.4 表 |
| 「CAE-ready mesh」と「CAE geometry mesh candidate」の呼称区別 | §0I.1 |
| 板金特有の3つの「中立面」概念の区別（geometric / shell reference / forming neutral） | §0I.1 |

### F. ソフトウェア工学 / MLOps

| 要素 | archi.md 該当箇所 |
|---|---|
| run_manifest.json による完全再現性（git commit, hash, config snapshot） | §0I.5 |
| Strategy パターン的な責務分離（Provider/Builder/Kernel/Remesher/Controller） | §0A.3 全体構成 |
| Go/No-Go条件を伴うフェーズ導入（R0〜R5ロードマップ） | §0D |
| 段階的本番導入ゲート（G0〜G5） | §0I.4 |

---

## Part 2. 前提知識の依存関係

学習の難所は「B（GNN/学習）」と「A（幾何数学）」が両方必要な箇所（LocalMeshRefiner, SizingAndBudgetController）である。
依存関係をおおまかに図示する。

```text
[A: メッシュ基礎・point-to-surface・トポロジー]
        │
        ├──→ [A: sizing field / chord deviation の数式]
        │            │
        │            └──→ [F: DeterministicRemesher の前後条件・ヒステリシス]
        │
        ├──→ [B: VecSetAEのPointNet/Attention構造（既存コード）]
        │            │
        │            └──→ [B: Fourier特徴 / spectral bias]
        │                       │
        │                       └──→ [B: LocalMeshRefinerのGNN設計・同変性]
        │
        └──→ [C: 安全ゲート設計（hard reject / monotonic guard）]
                     │
                     ├──→ [C: 状態機械・終端状態]
                     │            │
                     │            └──→ [C: 監査ログ・hash chain]
                     │
                     └──→ [B: OOD/novelty gate]

[D: 統計的評価] は A/B どちらの成果物を測るためにも独立して必要（並行して学べる）
[E: CAEドメイン知識] はB.4の品質指標を読むための前提知識として最初に軽く触れておくと後が楽
[F: MLOps/run_manifest] は実装時に毎回触れるので、Stage 0で一度概念だけ押さえれば十分
```

実装ロードマップ（§0H, 手順1〜11）は、ちょうどこの依存関係の順番（A→C→F→B→統合）で並んでいる。
これは偶然ではなく、「測定・安全機構を先に作ってからAIを足す」という§0Aの設計原則そのものである。
したがって学習順序も実装順序になぞらえるのが自然——というのが本計画の設計方針。

---

## Part 3. 学習計画（Stage 0〜6）

各Stageの目安所要時間は「集中して読む+手を動かす」前提のセッション数。厳密な期限ではなく目安。

### Stage 0: 現状把握 — 「何が既にあり、何がまだないか」

**ゴール**: archi.mdが「現行Stage Cの延長」と主張している根拠を、自分のコードベースに対して再現できる。

**読む箇所**:
- archi.md §0.1〜§0.4（結論と実測根拠）— 行1-216
- CLAUDE.md §13〜§17（R6〜R9、Stage A/B/Cの実装履歴）

**コード**:
- `biw_poc/src/model/reconstruct.py`（特に`reconstruct_midsurface()`, `prune_far_vertices()`, `subdivide_and_project()`）
- `biw_poc/src/model/dcudf_reconstruct.py`（archi.md §0Gの代替UDF抽出器比較の実験コード）

**演習**:
1. `reconstruct.py`のCLIデフォルト・`process()`関数デフォルト・`configs/model.yaml`の3者が食い違っていること（archi.md §0.3.1の表）を、実際にコードを `grep` して自分の目で確認する。
2. 表0.3.1の `LEGACY_R713 / COARSE_GATED / ADAPTIVE_BASELINE` の定義を見ずに、現行コードの引数だけから自分で同じ表を再構成してみる。

**自己チェック問題**（コードを見ずに回答）:
- Q1. なぜGT中立面に対して「SDF」ではなく「UDF」を使う必要があるのか、1段落で説明せよ。
- Q2. `refine_rounds=4`がなぜ「全体Transformerは非現実的」という結論の直接的根拠になるのか、頂点数の数値を使って説明せよ。
- Q3. point-to-vertex距離とpoint-to-surface距離はなぜ違う結論を出しうるのか、具体例（穴の誤検出）で説明せよ。

---

### Stage 1: 証拠設計（FieldEvidenceProvider / IndependentGeometryEvidence / NoveltyAndSupportGate）

**ゴール**: 「ニューラル場をオラクル扱いしない」という設計原則を、なぜそれが必要かという実測根拠込みで説明できる。

**読む箇所**: archi.md §0A.3 A・B・B2（行260〜341）

**コード**:
- `biw_poc/src/model/diagnose_outliers.py`（Recon→GT方向の外れ値診断。原因分析の実例）
- `biw_poc/src/model/diagnose_holes.py`（GT→Recon方向の被覆穴診断）
- `biw_poc/src/model/ensemble_compare.py`（Deep Ensembles不採用の実験。なぜ「学習済み確信度」だけでは不十分かの一次資料）

**演習**:
1. CLAUDE.md §15「Deep Ensembles比較」を読み、`ensemble_std`がなぜOOD検出に使えなかったかを自分の言葉で要約する（「全メンバーが同じ損失関数を共有しているため系統誤差は捉えられない」という結論を、自分で言い換えられるか確認）。
2. archi.md §0A.3-Bの`minimum_required = max(2*neighborhood_size, 0.25*target_count)`という fail-closed 条件を、現行`reconstruct.py`の「候補不足時に全グリッドへフォールバックする」挙動と対比し、なぜ後者が危険なのかを具体的な失敗シナリオで書き出す。

**自己チェック問題**:
- Q1. `FieldEvidence`と`IndependentEvidence`を別の型に分けている理由を、もし分けなかった場合に起きる失敗モードと共に説明せよ。
- Q2. NoveltyAndSupportGateの「学習embedding距離単独では安全判定しない」という制約は、Stage 0で読んだどの実験結果に基づくか。

---

### Stage 2: パッチ分割と局所GNN（MeshPatchBuilder / LocalMeshRefiner）

**ゴール**: VecSetAEの既存実装（global点群→global潜在表現）と、archi.mdが要求する局所パッチGNNの違いを、アーキテクチャ図として自分で描ける。

**読む箇所**: archi.md §0A.3 C・D（行343〜441）

**コード**（既存の類似実装として読む。LocalMeshRefiner自体は未実装）:
- `biw_poc/src/model/vecset_ae.py` 全体。特に
  - `furthest_point_sampling` / `knn_gather_idx`（パッチ中心選択・halo構築の参考になる近傍探索の基礎）
  - `MiniPointNet` / `PointTokenizer`（局所特徴抽出の既存実装パターン）
  - `TransformerEncoder` / `LatentCompressor`（global attentionの実装例。なぜこれを「局所パッチ+粗大域トークン」に分解する必要があるかを対比する）
  - `FourierFeatures` / `PositionalMLP`（spectral bias対策。LocalMeshRefinerでも座標エンコーディングに転用できる設計）

**外部資料**（archi.md §0Gが直接引用しているもの。原典を読む）:
- Neural Subdivision (SIGGRAPH 2020) — `https://arxiv.org/abs/2005.01819` — LocalMeshRefinerの直接の参照先。「決定論的細分化トポロジーと局所パッチ条件付き頂点位置予測の分離」という設計思想の原典。
- MeshCNN (SIGGRAPH 2019) — `https://arxiv.org/abs/1809.05910` — 辺ベース畳み込みの根拠（補助）。
- SubdivNet — `https://arxiv.org/abs/2106.02285` — subdivision connectivity階層表現（補助、採用しない理由も含め読む）。

**演習**:
1. `VecSetAE.encode()`が「全点群→単一潜在表現」を作っているのに対し、archi.mdの`R_patch = [t1,t2,n]`による局所フレーム変換（行373-377）を、自分で1パッチ分だけ手計算またはミニスクリプトで再現してみる（適当な3頂点パッチを用意し、法線・接線基底を求め、ワールド座標→局所座標へ変換する）。
2. equivariance regression test（行381-382）の意味を理解するため、`FourierFeatures`にランダムなSO(3)回転を施した入力を通し、出力の対称性が崩れることを確認する小実験を書いてみる（Fourier位置エンコーディングは座標非同変であることを体感する）。

**自己チェック問題**:
- Q1. なぜLocalMeshRefinerはワールド座標を直接入力しないのか。
- Q2. archi_audit.md第3版2.Aが指摘する「SO(2)面内回転の同変性」とは何か。SO(3)回転テストだけでは検出できない失敗モードを具体的に説明せよ。
- Q3. `delta_tangent`と`delta_normal`を分離出力する設計上の理由を、§0B.1の辞書式最適化方針と結びつけて説明せよ。

---

### Stage 3: サイズ場とリメッシュ数学（SizingAndBudgetController / DeterministicRemesher）

**ゴール**: `h_target(x)`の数式を、なぜその形なのか（幾何学的导出）を含めて自力で再導出できる。

**読む箇所**: archi.md §0A.3 F・G（行473〜822）— このStageが最も数式密度が高い

**コード**:
- `biw_poc/src/model/reconstruct.py: subdivide_and_project()`（置換対象の現行実装。一括1→4分割のコスト構造を体感する）
- `biw_poc/src/model/validate_refine.py`（`--refine-rounds`スイープの実測データ。CLAUDE.md §14の表と対応）

**演習**（このStageは数式の手計算が中心）:
1. `h_geometry(x) = sqrt(8 * δ_geo / κ(x))`（行616）を、円弧の弦と矢高（sagitta）の関係から自分で導出する。半径Rの円弧をδで近似する標準幾何の問題に帰着することを確認する。
2. `h_budget ≈ sqrt(A / (0.866 * V_max))`（行707）の`0.866`がどこから来るか（正三角形の面積公式 `(√3/4)a²` との関係）を導出する。
3. CLAUDE.md §17の表（R7-13、`refine_rounds`を3→4にした際の頂点数約6倍増、穴率1.1%への改善）の数値を、archi.mdの`MeshBudget`初期値（行683-697、`hard_max_vertices=500,000`等）と突き合わせ、もし§17の実験をGHMRのbudget gateに通したら何が起きるか（INFEASIBLE_MESH_BUDGETになるか等）を考える。

**自己チェック問題**:
- Q1. `h_edge_min`と`d_nonlocal_min`を分離する理由を、板金特有の形状（折り返し・近接シート）を例に説明せよ。
- Q2. `split_ratio=4/3, collapse_ratio=4/5`にヒステリシスを設ける理由を、振動（oscillation）という言葉を使わずに説明せよ。
- Q3. `rank(e)`が単一スカラーpriorityではなく辞書式tupleである理由を、archi_audit.md第3版2.Bの指摘（量子化されていない連続値が決定性要求と矛盾しうる）と関連付けて説明せよ。

---

### Stage 4: 安全状態機械と監査ログ（DeterministicSafetyKernel / 状態モデル）

**ゴール**: 8つの終端状態を、すべて「何が起きたらそこに到達するか」を自分の言葉で言える。

**読む箇所**: archi.md §0A.3-E（行442〜471）、§0A.4（行823〜884）、§0A.5（行886〜966）

**演習**:
1. 8終端状態（PASS, PASS_WITH_WARNINGS, INFEASIBLE_MESH_BUDGET, INFEASIBLE_GEOMETRY, INSUFFICIENT_EVIDENCE, OUT_OF_DISTRIBUTION, STALLED_SAFE, OSCILLATION_DETECTED, FAILED_NUMERICALLY）を見ずに紙に書き出し、その後archi.mdと突き合わせて漏れを確認する。
2. 自分で具体的なシナリオ（例:「拘束点が少なすぎる部品を入力した」「budget上限に達したが必須違反が残っている」）を3つ作り、それぞれどの終端状態に落ちるべきかをarchi.mdの分岐条件（行891-907）だけを根拠に判定する。
3. `AuditEvent`のハッシュチェーン（`sequence_no`, `previous_event_hash`）が何を検出するためのものか、git commitのハッシュチェーンとの類比で説明してみる。

**自己チェック問題**:
- Q1. `no accepted operation` や `max_rounds到達`が、なぜそのままPASSの理由にならないのか（行922-924）。
- Q2. MANUAL_REVIEWからの再開がなぜ「新しいrun」を要求し、同一runの再開を禁止するのか。
- Q3. rollback後のhashが一致しない場合に`FAILED_NUMERICALLY`へ倒す設計は、どの一般原則（fail-closed等）の具体化か。

---

### Stage 5: 学習データ設計と統計的評価

**ゴール**: 「なぜ6部品のPoC評価結果だけでGoを判定してはいけないのか」を統計的に説明できる。

**読む箇所**: archi.md §0C（行1078〜1206）、§0E（行1330〜1420）

**演習**:
1. paired bootstrap 95% CIが何を解決するための手法か（part間のばらつきとseed間のばらつきを混同しないため）を、自分の言葉で1段落にまとめる。可能なら numpy で適当な2群の擬似データに対しpaired bootstrap CIを実装してみる。
2. archi_audit.md第3版2.C（「5 held-out parts/3 families」が73倍のサイズ分散を考えると過小ではないか、という指摘）について、自分なりの結論（妥当/要拡大/要power analysis）を持つ。これは"答えのある問題"ではなく、ユーザー自身の設計判断を養う演習。
3. `SyntheticCorruptionCorpus`と`PipelineFailureReplayCorpus`を分ける理由を、CLAUDE.md §14のTD-16（cut locus外挿）やR7-12（prune由来の断片化）といった実際の失敗モードと対応付けて説明する。

**自己チェック問題**:
- Q1. 頂点パッチをランダムにtrain/test分割してはいけない理由（§0C.3）を、「リーク」という言葉の意味を込めて説明せよ。
- Q2. 学習時の`L_cad_cover`/`L_cad_overshoot`（CAD_GTを使う）と、推論時に使える証拠（input_evidence, field_evidence）が違う、ということが実務上何を意味するか説明せよ。

---

### Stage 6: 統合 — 全体パイプラインのトレースと自己レビュー

**ゴール**: archi.mdを見ずに、GHMRの全体フロー図（§0A.2のmermaid図、行232-258）を自力で再現できる。さらに、archi_audit.md第3版の指摘事項のうち最低2件について、自分なりの追加検証または反論を持てる。

**演習**:
1. 白紙に Neural Field / Input Cloud / Constraints / Current Mesh から PASS/INFEASIBLE/MANUAL_REVIEW 等の終端状態までの矢印を自力で描く。終わったら§0A.2のmermaid図と突き合わせ、抜けた分岐を確認する。
2. §0Hの実装順序（行1487-1503、11ステップ）を見ずに、自分なりの実装順序を考えてみる。その後§0Hと比較し、順序が異なる箇所があれば「なぜarchi.mdはこの順番を選んだか」を自分の言葉で説明できるか確認する（できなければStage 0-5のどこかに戻る）。
3. `archi_audit.md`第3版（最新版）を読み、2.A〜2.Eの指摘のうち2件以上について「自分はこの指摘に同意するか」を一段落で書く。これは監査結果を鵜呑みにしないための練習でもある。

**自己チェック問題（最終確認）**:
- Q1. GHMRが「初版素案の全機能を一度に実装しない」と判断した4点（§0.1）を、それぞれ「もし一度に実装したら何が壊れるか」とセットで説明できるか。
- Q2. Phase R0からR5までのGo条件（§0D）を一つも見ずに、各Phaseで何を測るべきかをおおまかに言えるか。
- Q3. このアーキテクチャの中で、自分が最も懐疑的な（さらに検証が必要だと感じる）設計判断はどれか。理由とともに1つ挙げられるか。

---

## Part 4. 進捗チェックリスト

進めながらこのファイルを直接編集し、`[ ]` を `[x]` にしていく運用を推奨する（reproducibility重視 = いつ・どこまで理解したかを記録に残す）。

- [ ] Stage 0: 現状把握（自己チェックQ1-3に回答できた日付: ____）
- [ ] Stage 1: 証拠設計（同上: ____）
- [ ] Stage 2: パッチ分割と局所GNN（同上: ____）
- [ ] Stage 3: サイズ場とリメッシュ数学（同上: ____）
- [ ] Stage 4: 安全状態機械と監査ログ（同上: ____）
- [ ] Stage 5: 学習データ設計と統計的評価（同上: ____）
- [ ] Stage 6: 統合・自己レビュー（同上: ____）

---

## 補足: 本書の位置づけ

本書は archi.md の規範性を持たない（学習用の補助資料）。技術的判断・実装方針は常に
`archi.md §0〜§0I`（規範） → `AGENTS.md`（決定履歴） → `archi_audit.md`（監査入力） の優先順位に従うこと。
本書が古い記述を含む場合は archi.md 側を正とする。
