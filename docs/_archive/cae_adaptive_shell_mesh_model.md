# CAE用適応シェルメッシュ生成モデル 詳細設計

状態: 初期設計案  
作成日: 2026-06-27  
対象データ: `C:\Users\hide2\IdeaBox\fill_volume`

## 1. 目的

本MVPの目的は、板金部品の正準中立面サーフェスから、CAEでそのまま使える中立面シェルメッシュを生成することである。ここではCAD/B-Rep復元を主目的にしない。最終出力は、節点、シェル要素、板厚、材料、法線、拘束点/荷重点/接触候補のnode setおよびelement setを持つ解析用メッシュである。

本設計では、巨大Transformerで全頂点列や全三角形列を直接生成するのではなく、板金中立面の構造を利用して、以下の階層に分解する。

```text
中立面STEP
  -> 階層メッシュ/点群特徴
  -> 粗い大域構造の記憶
  -> 局所パッチ詳細の注意選択
  -> 粗CAEメッシュ
  -> 適応細分化
  -> 品質投影/平滑化
  -> CAE用シェルメッシュ
```

重要な方針は、ニューラルモデルに幾何のすべてを暗記させないことである。全体形状は粗く、細部は局所共有デコーダで扱い、CAE品質保証は決定論的な投影・検証器に分担させる。

## 2. 利用可能データ

`fill_volume/HANDOFF.md` 時点の正準データは、穴フィル版中立面STEPと締結点アノテーションである。

| アセンブリ | fill STEP | mid STEP | CATParts | Edit | joints.json | 学習での扱い |
|---|---:|---:|---:|---:|---|---|
| `A0072600002_AllCATPart` | 43 | 43 | 43 | 43 | あり | すぐ使える主データ |
| `A0072601285_AllCATPart` | 32 | 0 | 0 | 32 | あり | すぐ使える主データ |
| `A0072600081_AllCATPart` | 0 | 0 | 102 | 0 | なし | mid/fill抽出後に使用 |
| `A0072600367_AllCATPart` | 0 | 0 | 18 | 0 | なし | mid/fill抽出後に使用 |
| `A0072600529_AllCATPart` | 0 | 0 | 152 | 0 | なし | mid/fill抽出後に使用 |

MVPで使う形状入力は、まず `fill_mid_surf/<assembly>/fill/*.stp` とする。`mid/` は穴検知・穴境界復元には有効だが、MVPの正準中立面は穴フィル版で統一する。`joints.json` は拘束点、締結点、穴中心、締結軸、部品間グラフとして使う。板厚、材料、解析タイプ、目標要素サイズなどのメタデータは別途付与できる前提とする。

## 3. MVPのスコープ

### やること

- OPEN_SHELLの板金中立面STEPを入力にする。
- 中立面からCAE用シェルメッシュを出力する。
- quad優先、triangle許容のquad-dominant meshを目指す。
- 締結点、穴中心、取り付け点、荷重点をnode setまたはelement setとして出力する。
- 板厚、材料、部品ID、feature label、refinement levelを属性として保持する。
- solver importを前提に、非多様体、自己交差、反転面、極端なaspect/skew/warpageを検出する。

### まだやらないこと

- B-RepまたはCATIA feature treeの生成。
- 完全な新規部品形状をメタデータだけから生成すること。
- 閉断面、重なり板、ヘム、複雑な絞り、厚み付きソリッドの直接生成。
- クラッシュ解析用のスポット溶接・接触・破断モデルまで含む完全デック生成。

MVPは「既存中立面形状をCAE-ready meshへ変換/補完するモデル」である。将来、完全な条件付き新規生成を行う場合は、このエンコーダ/デコーダを使い、別途 `metadata/constraints -> latent` のprior modelを追加する。

MVPの最初の成功条件は、ニューラルモデルが決定論baselineを超えることではない。まず、決定論baselineがデータ契約、set契約、quality gate、solver smoke testを通ることを成功条件にする。その後、MLは次の順に導入する。

```text
ML role 0:
  使わない。決定論baselineのみ。

ML role 1:
  target edge length / refinement priority / protected regionを予測する補助器。

ML role 2:
  局所パッチの変位・平滑化重みを予測する補助器。

ML role 3:
  coarse scaffoldや接続候補を提案する生成器。
```

現データ量では、最初からMLに最終メッシュ全体を任せない。

## 4. 入出力契約

### 入力

```text
part_surface:
  fill STEP open shell

optional_observation:
  mid STEP open shell
  raw tessellation at multiple deflections

assembly_context:
  assembly_id
  part_id
  canonical_part_id
  part transform / normalization
  joints.json

metadata:
  thickness_mm
  material_id
  target_edge_length_mm
  min_edge_length_mm
  analysis_type
  load_direction
  forbidden_volume if available
```

### 出力

```text
mesh:
  nodes: xyz
  elements: quad4 preferred, tria3 allowed
  normals

properties:
  thickness per element or property region
  material id
  part id
  element feature labels

sets:
  fastener node sets
  mounting node/edge sets
  load application sets
  contact candidate sets
  boundary edge sets
  high refinement regions

quality_report:
  solver import assumptions
  non-manifold count
  inverted face count
  self-intersection count
  aspect/skew/warpage/min_angle statistics
  constraint matching error
```

出力形式は実装段階で `Abaqus .inp`、Nastran `BDF`、LS-DYNA keyword、または中間の `npz/json/vtk` を選ぶ。MVP初期は学習と評価を優先し、まず `npz + json quality report + vtk/ply preview` を正本にする。

### 4.1 締結点・穴フィル面のCAE表現

`fill/` は穴が塞がれた正準中立面であるため、穴を幾何として必ず再開口するとは限らない。MVPでは、締結点周辺を次の2モードで扱う。

```text
mode A: virtual washer patch
  穴を再開口せず、filled midsurface上に円形または楕円形のwasher領域を切る。
  node set / element set / local refinement regionを作る。
  初期MVPの既定。

mode B: reopened hole patch
  mid STEPまたはjoints.jsonのhole_center/diameterから穴境界を復元し、
  穴周りにリング状quad-dominant patchを作る。
  bolt/mounting_holeで必要な場合に使う。
```

mode Aを既定にする理由は、穴フィル面を正準形状として保ちつつ、CAEの拘束・荷重伝達に必要な局所メッシュ密度とsetを確保できるからである。mode Bは穴縁応力やボルト穴周辺応力を評価したい場合に有効だが、穴再開口と境界整合の難度が上がる。

washer patchは `joints.json` の `axis` と `per_part.contact_xyz` または `hole_center_xyz` を使って作る。

```text
washer_patch:
  center_xyz
  axis_direction
  inner_radius if reopened hole
  outer_radius
  radial_divisions
  circumferential_divisions
  node_set_name
  element_set_name
  coupling_policy
```

MVP初期値:

```text
outer_radius:
  max(2.0 * hole_radius, 8.0mm) for hole joints
  6.0mm for weld/contact point if diameter is unknown

radial_divisions:
  2-3

circumferential_divisions:
  12-24

coupling_policy:
  node set only in neutral export
  RP + distributing coupling in solver-specific export
```

この仕様により、締結点を単なる最近傍nodeに落とさず、CAEで使える局所領域として扱う。

A0072601285のように `mid/` が無いアセンブリでは、実穴境界は形状から復元しない。`joints.json` のhole center/diameterがある場合だけ仮想washerまたはreopened patchを構成する。hole center/diameterが無い溶接点はcontact washerとして扱う。

### 4.2 node/element setの意味論

setは名前だけでなく、生成方法と許容誤差をmanifestに残す。

```text
set_contract:
  set_name
  set_type: node | element | edge | reference_point
  source_joint_id
  source_part_id
  construction: nearest_node | washer_patch | boundary_band | projected_point
  search_radius_mm
  max_projection_error_mm
  solver_semantics: neutral | abaqus_coupling | nastran_rbe | lsdyna_constrained
```

中間形式ではsolver非依存のneutral semanticsを持つ。Abaqus/Nastran/LS-DYNA固有の実体化はexport層で行う。

## 5. モデル全体像

モデルは以下の5段で構成する。

```text
1. Geometry Pyramid Builder
2. Hierarchical Midsurface Encoder
3. Coarse CAE Mesh Decoder
4. Adaptive Refinement Decoder
5. Quality Projection and Export
```

大域形状と局所詳細を分離することが中心である。

```text
中立面全体:
  粗い階層で低トークン化して読む

穴、自由縁、締結点、曲率部:
  局所パッチとして高解像度で読む

潜在表現:
  少数latent tokenが粗情報と局所情報をcross-attentionで読む

出力:
  粗メッシュを作り、必要箇所だけ細分化する
```

## 6. Geometry Pyramid Builder

入力STEPを一度だけテッセレーションして終わりにしない。複数解像度と局所特徴を持つピラミッドを作る。

### 6.1 基本テッセレーション

`fill_volume/annotation_app/mesh.py` と同様に、OCCTでSTEPを読み、`BRepMesh_IncrementalMesh` で三角形化する。MVPでは以下を追加する必要がある。

- face idの保持
- face orientationの保持
- boundary edgeの抽出
- edge length分布の保存
- 法線の向きの一貫性検査
- テッセレーション設定のmanifest保存

### 6.2 多解像度化

```text
Level 0:
  入力の高解像度テッセレーション。

Level 1:
  境界と高曲率を保った中解像度メッシュ。

Level 2:
  geodesic FPSまたはdecimationによる粗メッシュ。

Level 3:
  拘束点、自由縁、主要曲率線、部品bboxを含む骨格グラフ。
```

粗化では、単純なランダムサンプルではなく、以下を優先して残す。

- 自由縁
- 穴中心/締結点近傍
- 高曲率領域
- 面積の大きい代表領域
- bboxの極値付近
- `joints.json` の接触点・穴中心

### 6.3 節点/面特徴

各点またはface tokenには以下を付与する。

```text
geometry:
  xyz normalized
  normal
  area
  local edge length
  curvature proxy
  boundary flag
  face id

context:
  distance to nearest boundary
  distance to nearest joint
  distance to nearest hole center
  distance to bbox faces
  part-local coordinates

mesh_quality:
  local aspect proxy
  local sampling density
  connected component id
```

座標正規化は、アセンブリ座標の絶対値が大きいため必須である。各部品ごとに `center` と `scale` を保存し、`joints.json` の接触点も同じ正規化で扱う。

### 6.4 正規化とjoint round-trip契約

manifestには、部品IDと座標変換を必ず保存する。

```text
part_manifest:
  assembly_id
  part_id_raw
  canonical_part_id
  stp_path
  source_kind: fill | mid
  center_xyz
  scale
  normalized_bounds
  transform_world_to_part
  transform_part_to_world
  joints_used
```

`joints.json` はアセンブリ座標である。したがって、正規化後のjoint点を再びworld座標へ戻したとき、元の座標に戻ることをgateにする。

```text
joint_round_trip_gate:
  max_error_mm <= 1.0e-6 for pure normalization
  max_projection_error_mm <= configured tolerance after surface projection
```

part_idはアセンブリ間で命名規則が違うため、学習では `canonical_part_id` を使う。元のIDは必ず `part_id_raw` として残す。

## 7. Hierarchical Midsurface Encoder

エンコーダは、全体を粗く読む経路と細部を読む経路を分ける。

```text
Midsurface
  -> Coarse Global Encoder
  -> Fine Patch Encoder
  -> Latent Cross-Attention Fusion
```

### 7.1 Coarse Global Encoder

目的は、部品の大域構造を少数トークンに圧縮することである。

入力はLevel 2/3の粗メッシュまたは骨格グラフ。

```text
coarse node token:
  xyz
  normal
  boundary flag
  curvature proxy
  local area
  nearest joint distance
  bbox-relative coordinate
  metadata embedding

coarse edge token:
  relative vector
  edge length
  dihedral/normal difference
  boundary edge flag
  geodesic distance class
```

実装候補:

```text
GraphSAGE / EdgeConv / MeshGraphNet block
  + sparse graph attention
  + latent query pooling
```

全点間attentionは使わない。隣接エッジ、geodesic近傍、拘束点近傍だけを見る。出力は `Z_global` とする。

推奨初期値:

```text
coarse nodes: 128-512
hidden dim: 128
layers: 4-6
global latent tokens: 32-64
```

### 7.2 Fine Patch Encoder

目的は、穴、自由縁、曲率部、締結点周辺など、CAEメッシュ品質に効く局所形状を読むことである。

パッチ中心は以下から選ぶ。

```text
curvature-aware FPS
boundary samples
joint-near samples
hole-center-near samples
area-uniform samples
```

各パッチは局所座標系に変換する。

```text
origin:
  patch center

z axis:
  local normal

x/y axis:
  boundary tangent, principal direction, or stable PCA direction
```

局所座標化により、同じ穴周り、自由縁、曲げ近傍の作法を部品位置に依存せず共有できる。

実装候補:

```text
mini PointNet
EdgeConv
local self-attention
patch pooling
```

出力は `patch tokens` とする。

推奨初期値:

```text
patch centers: 256-1024
points per patch: 32-64
patch token dim: 128
local encoder layers: 2-4
```

### 7.3 Latent Cross-Attention Fusion

少数のlearnable latent tokensが、粗情報と局所情報を読む。

```text
latent tokens
  attend to coarse tokens
  attend to fine patch tokens
  attend to joint/meta tokens
  -> Z
```

`Z` は最終的な形状潜在である。ここではCLS 1本に潰さない。全体用、境界用、締結点用、局所細分化用の複数latentを持つ。

推奨初期値:

```text
latent tokens: 64-128
latent dim: 128
cross-attention layers: 2-4
heads: 4
```

## 8. Decoder設計

### 8.1 粗CAEメッシュデコーダ

MVPでは、粗メッシュの接続を完全にニューラル生成しない。まず入力中立面から決定論的に粗scaffoldを作り、モデルはその補正と属性を出す。

```text
coarse scaffold:
  boundary-preserving decimated mesh
  joint anchors included
  bbox extremes included
  connected component preserved
```

モデルは以下を予測する。

```text
coarse vertex displacement in local frame
coarse normal correction
coarse feature label
coarse target edge length
coarse quad orientation field
```

この設計により、粗トポロジーを完全生成する不安定さを避ける。完全新規生成へ進む場合は、後で別のcoarse graph generatorを追加する。

### 8.2 Sizing Field Decoder

各頂点・エッジ・faceに目標要素サイズを出す。

```text
small h:
  締結点近傍
  荷重点近傍
  自由縁
  高曲率
  穴/切欠き境界
  応力集中予測領域

large h:
  平坦で低応力な内側領域
```

出力:

```text
target_edge_length
refinement_priority
quad_preferred_direction
feature_class
```

quad-dominant化では、sizing fieldに加えて方向場を持つ。方向場は、自由縁、長手方向、締結点周りの円周方向、曲率主方向を優先して決める。

```text
quad_orientation_field:
  boundary tangent aligned near free edges
  circumferential/radial aligned in washer patches
  principal direction aligned in high-curvature areas
  smooth cross-field in flat interior
```

MVP初期は完全な四角形メッシュ生成を目指さず、以下の段階で進める。

```text
stage 0:
  triangle shell mesh + quality metrics

stage 1:
  tri-pairingでquad候補を作る
  qualityが悪い領域はtriaとして残す

stage 2:
  feature-aligned quad patchをwasher/free-edge/high-curvature領域に導入

stage 3:
  cross-field guided quad remeshingへ拡張
```

したがって、MVP文脈でのquad-dominantは「全領域を完全quad化する」という意味ではなく、CAE上重要な領域からquad化し、低重要度領域ではtria混在を許す方針である。

### 8.3 Adaptive Refinement Decoder

各局所操作にスコアを出す。

```text
edge split score
edge collapse score
edge flip score
face smooth/protect score
```

ただし操作は学習器だけで決めない。以下のhard gateを通す。

```text
split allowed if:
  edge length > split_ratio * target_edge_length
  budget not exceeded
  child edges remain above min_edge_length
  local quality does not regress

collapse allowed if:
  edge length < collapse_ratio * target_edge_length
  boundary/joint protected vertices are not removed
  topology remains manifold

flip allowed if:
  local minimum angle improves
  normal orientation stays consistent
  boundary/feature edges are not destroyed
```

split/collapseにはヒステリシスを持たせ、操作の振動を防ぐ。既存の学習資料 `09_remeshing操作split_collapse_flip.md` の考え方を引き継ぐ。

### 8.4 Local Patch Mesh Decoder

局所パッチごとに、共有重みのデコーダで細部を補正する。

```text
input:
  local patch token
  global latent summary
  local frame
  sizing field
  feature class

output:
  local vertex displacement
  local normal
  local smoothing weight
  local protect flag
```

変位は局所座標系で出す。出力変位は必ず局所目標辺長に対してクランプする。

```text
max displacement:
  displacement <= max_disp_ratio * target_edge_length
```

これにより、細分化で面積が膨張したり、局所的に折れ込むことを防ぐ。

## 9. Quality Projection

ニューラル出力は最終メッシュではない。必ず決定論的な品質投影を通す。

投影器が行うこと:

```text
snap to reference midsurface
boundary preservation
joint anchor preservation
normal orientation repair
component orientation repair
edge length smoothing
Laplacian / tangential smoothing
local flip for min-angle improvement
self-intersection check
non-manifold check
quality budget check
```

重要な原則:

```text
形状方向のprojection:
  中立面から離れないようにする

接線方向のsmoothing:
  要素品質を改善する

法線方向の過大移動:
  禁止または強くクランプする
```

品質投影が失敗した場合は、品質不明のメッシュを出さず、fail-closedで停止する。

### 9.1 Quality metric manifest

CAE品質指標はsolverや社内基準で定義差が大きい。MVPでは、実装前にquality manifestを固定する。

```text
quality_manifest:
  unit_system: mm-tonne-second or mm-kg-second
  shell_element_type: quad4/tria3
  aspect_ratio_definition
  skew_definition
  warpage_definition
  jacobian_proxy_definition
  min_angle_definition
  max_angle_definition
  thresholds_by_analysis_type
```

初期のquality reportは、少なくとも各指標の `definition_id` と `threshold_profile_id` を持つ。数値だけを保存しない。

### 9.2 Solver smoke test

MVP初期の中間正本は `npz + json + vtk` とするが、hard gateとして使う基準solver形式を1つ固定する。初期候補はAbaqus `.inp` とする。理由はorphan mesh、shell section、node/element set、RP/couplingの表現を最小限で検証しやすいためである。

```text
solver_smoke_test:
  export .inp
  import or parse deck
  verify nodes/elements count
  verify shell section assignment
  verify material assignment
  verify node/element sets
  verify no duplicate ids
  optionally run datacheck/no-analysis import
```

Nastran/LS-DYNAへの対応はexport層で追加する。solver固有の制約を学習データ本体へ混ぜない。

## 10. 学習タスク

### 10.1 教師データの現実的問題

現在あるのは中立面STEPであり、正解CAEメッシュが必ずあるわけではない。したがって、MVP初期は以下の2段階に分ける。

```text
Stage A:
  中立面STEPから決定論的CAEメッシュを作り、pseudo targetにする。

Stage B:
  人手または既存CAEメッシュが得られた部品でfine-tuneする。
```

Stage Aは「CAEメッシュ作成者の最終品質」を超えるものではないが、モデル構造・入出力契約・評価器を固めるには十分である。

疑似教師は常にmanifest付きで保存する。

```text
pseudo_target_manifest:
  source_step
  source_joint_json
  tessellation_profile
  remesher_version
  quality_manifest
  target_edge_profile
  washer_patch_profile
  failed_operations
  rollback_count
  solver_smoke_result
  known_limitations
```

学習時には、pseudo targetの品質スコアをサンプル重みとして扱う。品質が低いpseudo targetを正解として強く学習しない。

ML導入前に、pseudo target生成器自体を評価対象にする。つまり、最初の評価レポートはモデルではなくbaseline remesherの評価である。

```text
baseline report:
  input STEP -> deterministic mesh
  washer patch construction
  quality report
  set report
  solver smoke export
  failure reasons
```

### 10.2 自己教師あり事前学習

入力中立面だけから始めるため、エンコーダは自己教師ありで強くする。

```text
multi-tessellation contrast:
  同じSTEPを異なるdeflectionでテッセレーションし、同じ潜在に近づける。

masked patch reconstruction:
  一部パッチを隠し、局所形状/法線/曲率/境界を復元する。

boundary and joint prediction:
  boundary flag、joint近傍、hole中心距離を予測する。

local quality prediction:
  入力メッシュの局所品質、目標edge length、細分化必要度を予測する。
```

これにより、少数アセンブリでもエンコーダが中立面構造を学習しやすくなる。

### 10.3 Supervised / pseudo-supervised training

疑似教師CAEメッシュがある場合、以下を学習する。

```text
coarse shape loss:
  生成メッシュと参照中立面の距離

boundary loss:
  自由縁の位置・形状・法線

normal loss:
  face/vertex normal alignment

sizing loss:
  目標edge length field

operation loss:
  split/collapse/flip policyの教師

quality loss:
  aspect, skew, warpage, min/max angle

constraint loss:
  締結点・穴中心・取り付け点setの一致

cae proxy loss:
  質量、断面二次モーメント、簡易剛性指標
```

数式表記ではなく、実装上の合成損失としては以下にする。

```text
total_loss =
  shape_loss
  + boundary_loss
  + normal_loss
  + sizing_loss
  + operation_loss
  + mesh_quality_loss
  + constraint_loss
  + cae_proxy_loss
```

## 11. 評価基準

MVPでは、見た目の近さよりCAE可用性を上位に置く。

### 11.1 hard gate

```text
solver import success
solver smoke test success for baseline profile
finite coordinates only
non-manifold count = 0
duplicate element count = 0
inverted element count = 0
self-intersection count = 0
thickness/material assignment completeness = 100%
required node/element sets completeness = 100%
washer patch construction success for required joints
```

### 11.2 mesh quality

```text
quad ratio
tria ratio
edge length p50/p95/max
min angle p05/min
max angle p95/max
aspect ratio p95/max
skew p95/max
warpage p95/max
Jacobian proxy p05/min
area ratio
component count
```

Aspect ratioなどは定義揺れがあるため、実装時に必ず定義をmanifestに固定する。

### 11.3 geometry and constraint

```text
bidirectional point-to-surface distance
boundary curve distance
normal error
joint point error
hole center error
mounting plane error
forbidden volume penetration
```

### 11.4 CAE proxy

```text
mass error
section inertia error
linear static stiffness proxy
reference load displacement error
strain energy error if available
hotspot ranking agreement if available
```

## 12. パラメータ効率の設計根拠

巨大Transformerで全点を読む場合、全点間の関係を直接学習するため、点数が増えるほど計算もパラメータ要求も重くなる。

本設計では、以下の分解で効率化する。

```text
全体:
  粗メッシュ/骨格グラフだけを見る。

細部:
  局所パッチだけを見る。

融合:
  少数latent tokenが必要な情報をcross-attentionで読む。

生成:
  粗メッシュを先に作り、必要箇所だけ細分化する。

品質保証:
  ニューラルに暗記させず、決定論的projectorに外出しする。
```

初期MVPのモデル規模目安:

```text
prototype:
  2M-5M parameters

MVP:
  8M-20M parameters

avoid:
  100M+の全点Transformer
```

## 13. 実装モジュール案

最初の実装は、既存の `annotation_app/mesh.py` と `_archive_pre_annotation_tool/biw_poc/src/r0/adaptive_remesh.py` の知見を参照しつつ、新規パッケージとして切る。

```text
cae_mesh_generator/
  pyproject.toml
  src/cae_mesh_generator/
    data/
      scan_fill_volume.py
      manifest.py
      step_tessellate.py
      joints.py
    geometry/
      features.py
      hierarchy.py
      quality.py
      remesh_ops.py
      projection.py
    model/
      encoder.py
      coarse_decoder.py
      refinement_decoder.py
      local_patch_decoder.py
      losses.py
    train/
      pretrain_encoder.py
      train_refiner.py
    export/
      vtk_export.py
      inp_export.py
      bdf_export.py
    eval/
      mesh_quality_report.py
      constraint_report.py
      cae_proxy_report.py
  tests/
```

最初に作るべき実装は、モデル本体ではなく以下である。

```text
1. fill_volume dataset scanner
2. STEP -> attributed tessellation
3. mesh quality metric implementation
4. deterministic baseline remesher
5. pseudo target manifest
6. encoder prototype
```

## 14. リスクと対策

### リスク: 正解CAEメッシュがない

対策:

- 決定論的remesherでpseudo targetを作る。
- 同じSTEPの複数テッセレーションから自己教師あり事前学習を行う。
- 少数でも人手CAEメッシュが得られたらfine-tuneと評価に使う。

### リスク: メッシュ生成が形状一致だけに過適合する

対策:

- hard gateをCAE品質優先にする。
- 形状距離だけでbest checkpointを選ばない。
- solver importとquality metricsを正式評価に含める。

### リスク: 局所細分化で面積膨張やシワが出る

対策:

- 変位をtarget edge length比でクランプする。
- 法線方向移動と接線方向移動を分離する。
- 面積比、warpage、self-intersectionをgateに入れる。

### リスク: 2アセンブリでは汎化評価が弱い

対策:

- 既存の追加AllCATPartからmid/fill抽出を進める。
- 部品family単位でhold-outする。
- 単一部品/単一アセンブリ評価は暗記リスクありと明記する。

## 15. 次の作業順

1. `fill_volume` のmanifest生成器を実装する。
2. quality manifestとsolver smoke test profileを固定する。
3. 各STEPのテッセレーション、bbox、面数、境界数、joint数を保存する。
4. washer patch / node set / element setの中間表現を実装する。
5. mesh quality metricを実装する。
6. 決定論的なCAE pseudo target remesherを作る。
7. baseline時点で `npz + vtk + quality_report.json + solver_smoke.inp` を出力し、regression test化する。
8. MLなしbaselineの失敗理由を分類する。
9. Hierarchical Midsurface Encoderのprototypeを実装する。
10. 自己教師ありpretrainを行う。
11. まず sizing/refinement/protected-region 予測器を追加する。
12. 粗メッシュ + 適応細分化decoderへ進む。

この順に進めると、モデルより前にデータ契約と評価契約が固まる。CAE用途では、この順番を崩さないことが重要である。

## 16. 現在のprototype実装状況

`cae_mesh_generator` には、中立面形状のみを入力するautoencoder probeを実装した。これは最終CAE mesh generatorではなく、dual encoderが板金中立面の大域構造と局所構造を潜在表現へ持てるかを確認するための段階である。

実装済み:

- STEP filled midsurfaceのテッセレーション、面積重み付き点群サンプリング、part-local正規化
- coarse global encoder: FPS代表点 + Transformer encoder
- fine local encoder: FPS center + kNN patch + shared PointNet
- latent cross-attention fusion
- baseline point decoder: latent queryからunordered point cloudを復元
- structured decoder: coarse scaffoldを復元し、各scaffold点から近傍fine tokenへlocal attentionして局所点を細分化
- visual evaluation: target/reconstruction/error PLY、PNG、Plotly HTML、split別aggregate metrics、structured scaffold overlay

q20-40 holdoutでの初期結果:

| decoder | train Chamfer mean | val Chamfer mean | val target within 5mm |
|---|---:|---:|---:|
| fixed point query | 12.110mm | 42.118mm | 4.7% |
| structured scaffold + local refinement | 10.590mm | 30.687mm | 13.3% |

この結果は、固定query点群decoderより、coarse scaffoldを介するdecoderの方が未知部品で構造を壊しにくいことを示す。ただし、まだCAE-readyではない。現出力は点群であり、node/element topology、境界保持、四角/三角要素品質、node/element set、solver importは未実装である。

次の実装は、structured scaffoldを以下のCAE Mesh IRへ拡張する。

- scaffold node position
- scaffold edge/face candidate
- local target edge length
- boundary/protected-region/refinement logits
- deterministic projection to filled midsurface
- element quality gate and failure report
