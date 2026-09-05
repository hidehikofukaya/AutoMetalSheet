# 引継ぎメモ: softmin_r717 実験（R7-17系、Round 12〜15）

作成日: 2026-06-21 / 作成者: Claude（このセッション）/ 宛先: codex（並行作業エージェント）
更新: Round 15差分（SoftminGuidanceModel 再学習・再構成・配向異常の調査）を反映、新規§0として最上段に配置。旧§0（GTフロアの適用層フォーク、未回答）は§1に繰り下げ。
追補: ユーザー提供のスクリーンショット（GT+recon重ね表示）により、欠陥が「メッシュ全体にランダム分散」ではなく「境界輪郭全周を取り囲む櫛状（小腸状）スパイク＋内部のカバレッジ不足」という構造を持つことが判明、§0内の仮説を改訂。

## 0. 【新規・最重要】SoftminGuidanceModel 再構成メッシュの法線配向がほぼランダム — 原因未確定、codexの知見を求む

### 背景
codexが構築した`SoftminGuidanceModel`系（`softmin_guidance.py` / `softmin_guidance_dataset.py` / `train_softmin_guidance.py` / `reconstruct_softmin_guidance.py` / `diagnose_softmin_guidance.py`、いずれもClaude未着手のcodex主導実装）のチェックポイントが本セッション開始時点で紛失しており（再構成済みメッシュ`body002_guidance_codex_midsurface.ply`のみ現存）、ユーザー指示によりclaude側で同一データセットから再学習・チェックポイント保存付きで再実行しました。学習自体は正常終了しましたが、再構成メッシュを診断したところ、**codexの紛失チェックポイントの再構成結果と共通する、未解明の法線配向異常**が見つかりました。これがStage Aの設計の話とは独立した、より緊急性の高い論点と判断し、本メモの最上段としました。

### 再学習の数値サマリー
```
python train_softmin_guidance.py --data .../body002_filled_dataset.h5 \
  --ckpt-dir checkpoints_softmin_r717_ab --epochs 2000 --ckpt-every 100 \
  --log-every 50 --device cuda --seed 0
```
epoch1: total=10.53 / composite=2.19323 → epoch2000: total=0.46920 / composite=0.33771
（near_potential_mae=0.13705mm, near_step_mae=0.20066mm @epoch2000）。
best checkpoint: epoch=1928, near composite=0.33238。21チェックポイント保存確認済み（`best.pt`/`last.pt`/`epoch0100.pt`〜`epoch2000.pt`、計約4.3GB）。
収束カーブはepoch2000時点でもなだらかに下降中で、完全にプラトーには達していません。

### 再構成 → 診断の比較（`diagnose_softmin_guidance.py --recon-ply`）

| 指標 | claude再学習チェックポイント | codex紛失チェックポイントの旧再構成 |
|---|---|---|
| 頂点/面数 | 7,441 / 12,355 | 5,539 / 8,510 |
| connected components | **7**（最大成分95.6%の頂点を保持） | **27**（最大成分は**32%**のみ） |
| face-normal alignment vs 最近傍input法線（mean cos） | 0.015 | 0.004 |
| 同、frac(\|cos\|<0.5、つまり60°超ズレ) | **0.752** | **0.741** |
| Recon→GT 真の距離（mean/median/p90/max） | 8.916 / 8.994 / 17.313 / 25.654mm | 8.405 / 7.675 / 16.132 / 27.841mm |
| corrcoef(Recon→GT距離, GT境界エッジまでの距離) | **-0.0185** | **-0.0337** |

**観察1（収束との関係）**: claude側の再学習はメッシュの**連結性**を大幅に改善しました（主成分が32%→95.6%）が、**法線配向の異常はほぼ同じ水準のまま**でした。つまりこの異常は学習収束不足の単純な副作用ではなく、より構造的な原因と考えられます。

**観察2（既知のTD-16パターンとの不一致）**: 別系統（VecSetAE/DCUDF、CLAUDE.md §14のTD-16）で確立している失敗パターンは「Recon→GTの誤差はGTの開いた境界（cut locus）付近に強く集中する」というものでした（実測corrcoef≈1.0に近い相関、§14参照）。しかし本系統では**corrcoef(Recon→GT距離, 境界距離) ≈ -0.02〜-0.03とほぼ無相関**であり、誤差が境界に集中しているのではなく、**メッシュ全体に分散した配向の異常**であることを示唆しています。TD-16の知見をそのまま当てはめるのは不適切と判断しました。

### 検討した仮説と、データで否定できたもの

**仮説（否定済み）**: `train_softmin_guidance.py`の`compute_guidance_losses()`内、`direction_weight = category_weight * direction_strength`（341〜348行付近）が、方向損失の勾配を広範囲で抑制しているのではないか（`direction_strength`は候補枝の方向ベクトルが打ち消し合う＝GTの`soft_direction`自体が不明瞭な箇所で小さくなる設計のため）。

実データで検証した結果、これは支持されませんでした：
```
query_direction_strength: 全14,336点 mean=0.980 median=1.0
  near(cat0)カテゴリ: mean=0.967 median=0.99995、frac(<0.9)=12.4%
  far/boundaryカテゴリ: ほぼ全点が0.99以上
```
つまり`direction_strength`が低い（方向損失が弱められる）箇所はごく一部（near点の12.4%程度）に限られ、メッシュ全体に及ぶ配向ランダム性の説明にはなりません。**この仮説は棄却**しました。

### 現時点で最も有力と考える仮説（未確認、codexの知見を求めたい点）

`bounded_project()`（`reconstruct_softmin_guidance.py`）はモデルが予測した`direction_toward_surface`を使って点群を移動させるだけです。一方、最終メッシュの**面法線そのもの**は、移動後の点群に対して`pv.PolyData(projected).reconstruct_surface(nbr_sz=nbr_sz)`（VTKのHoppe型陰関数再構成、`nbr_sz=20`のデフォルト）が**点群の局所分布からPCA的に再推定**したものであり、`orient_components_consistently()`（`reconstruct.py`からimport、本系統でも流用）はその後に**連結成分単位で1回だけ**符号を反転させる後処理にすぎません（実際「9 connected components, 4 flipped」のログを確認済み）。

つまり今回の診断（「面法線 vs 最近傍input法線」の比較）は、次の2つの異なる層の問題を区別できていません。
  (a) モデルの`direction_toward_surface`予測自体が間違っている（学習・モデル品質の問題）
  (b) 投影後の点群密度・分布がVTKの局所法線推定にとって不適切で、`reconstruct_surface()`が面ごとに矛盾した法線を合成してしまう（再構成パラメータ・点群品質の問題、予測方向が正しくても起こりうる）

主成分（頂点の95.6%を占める）の中だけでも面の75%が60°超ズレているという事実は、**連結成分単位でしか直せない`orient_components_consistently()`では原理的に説明できない**規模の局所的不整合であり、(b)が支配的である可能性を示唆していると考えています。ただし確証はありません。

### Round 15 追補: ユーザーのスクリーンショット目視確認による知見（`poc_result_viewer.py`、GT + `body002_r717ab_recon_midsurface.ply`の重ね表示）

ユーザーから`poc_result_viewer.py`でGT層（灰色）とclaude再学習チェックポイントの再構成層（水色）を重ねたスクリーンショットが提供され、「小腸のようにうねりながら連なっている。あるべき姿ではない」との指摘がありました。目視で確認できる構造は以下の通りです。

- 矩形パネル状のGT中立面（灰色）の**輪郭全周を取り囲むように**、再構成メッシュ（水色）が**密な櫛状（コームティース状）の細長いスパイク**として連なっている。スパイクは内側のGTパネル境界から外側に向かって放射状に突き出ており、隣接スパイク同士の高さ・向きが不揃いにうねっているため、全体として「小腸」のような有機的でデコボコした見た目になっている。
- 一方、GTパネルの**内側（フラットな広い面）はほぼ再構成メッシュに覆われておらず、灰色のGTがそのまま透けて見えている**。つまり再構成メッシュは「内部の平坦な面」をほとんど再現できておらず、「境界周辺」にのみ密に（しかし不正確に）メッシュを生成している、という偏った分布になっている。

### 仮説の改訂（Round 15当初の「メッシュ全体にランダムに分散した配向異常」から修正）

この視覚的証拠により、当初の仮説（「面法線の不整合がメッシュ全体に一様にランダム分散している」）を**修正**します。実際には欠陥は次の2つの偏りを持つ、より構造的なものです。

1. **境界周辺への集中**: 異常なメッシュ（スパイク）は中立面の輪郭（cut locus境界）にほぼ沿って分布しており、全くのランダム分散ではなく、境界に近い領域に偏って発生しているように見えます。これは§0で当初「corrcoef(Recon→GT距離, 境界距離)≈-0.02〜-0.03とほぼ無相関」としてTD-16パターン（境界に強く相関）と異なると報告した点と一見矛盾します。再解釈すると、**輪郭のほぼ全周にわたって一様にスパイクが発生しているため、「境界からの距離」という単一スカラーでは判別力を持たなくなっている**（境界に近いか遠いかによらず、輪郭沿いのどこでもスパイクが起きている）可能性が高いと考えます。TD-16（別系統）は境界の一部だけが突出する局所的な現象でしたが、本系統はほぼ全周が一様に侵食されている点で、現象としては類縁ですが規模・分布が大きく異なります。
2. **内部のカバレッジ不足**: フラットな内部面がほとんど再構成されていないのは、これまでの診断（along-normal偏差がlateralより支配的、frac=0.792）や、connected component報告で見られた「spike/needle」形状分類と整合します。境界付近に候補点・投影点が偏り、内部の平坦領域に十分な密度の候補点が生成・収束していない可能性があります。

### 改訂後の有力仮説: 境界カテゴリの緩い教師信号 + 反復投影によるドリフトの増幅

`train_softmin_guidance.py`の`compute_guidance_losses()`では、カテゴリ別に`distance_clips_mm`が異なります（near/far/boundaryで別値、boundaryのクリップ幅は他カテゴリよりかなり広い設計、R7-09由来）。つまり**境界カテゴリの点ほど、step_distance・direction共に教師信号の許容誤差が大きく、モデルの予測精度が本質的に粗くなりやすい**設計です。

これに`reconstruct_coarse_surface()`の`projection_iterations=3`（各反復で点ごとに独立して`direction_toward_surface`方向へ`max_step_mm=5.0mm`を上限に移動）が重なると、境界付近の点群について、点ごとにわずかに異なる（精度の粗い）方向予測が3回複利的に効いてしまい、**本来collapseして一枚のフラットな面に収束すべき点群が、点ごとに少しずつ異なる方向へ広がっていく**ことで、観測されたような放射状の細長いスパイク（櫛状の小腸構造）が生まれるのではないか、という仮説です。この仮説は以下と整合します。
- along-normal偏差がlateralより支配的（スパイクは法線方向に「突き出る」ため）
- spike/needle形状のconnected component（個々のスパイクが細長い塊として検出される）
- 面法線のランダム性（細く尖った形状はVTKの局所PCA法線推定が原理的に不安定になりやすい）
- 境界沿いにほぼ一様に分布（境界カテゴリの教師信号が全体的に粗いため、特定の1点ではなく輪郭全周で起きる）

ただしこれは未検証の仮説です。確認には、VTKの`reconstruct_surface()`に**投入する直前**の投影済み候補点群（3回の反復後、メッシュ化前）を直接ダンプして、その時点で既に櫛状に広がっているかどうかを見れば、(a)（投影・方向予測側の問題）と(b)（VTK再構成側の問題）を明確に切り分けられます。まだ実施していません。

### codexへの相談事項
1. `reconstruct_coarse_surface()`のデフォルト（`grid_res=48`, `candidate_factor=3.0`, `projection_iterations=3`, `max_step_mm=5.0`, `nbr_sz=20`）は、Recon→GTがmean8〜9mm程度に留まっている現状の収束度に対して、`reconstruct_surface()`に渡す点群密度・均一性として十分と考えていますか？ 自分（claude）ではまだ「投影後・VTK投入前の点群がGTにどれだけ収束しているか」を直接測れていません。
2. 境界カテゴリの教師信号（`distance_clips_mm`のboundary幅）が他カテゴリより緩いことが、`projection_iterations=3`の反復投影と組み合わさって境界沿いの櫛状スパイクを増幅している、という上記仮説についてどう思われますか。心当たりや、別の原因に心当たりはありますか。
3. もし(b)（VTK側の法線推定の不整合）が疑わしいとしたら、`nbr_sz`を上げる、`projection_iterations`を増やす、あるいは投影前にambiguity-basedな間引き/平滑化を挟む、などの対策に心当たりはありますか。
4. 同じ`reconstruct_surface()` + `orient_components_consistently()`の組み合わせは別系統（DCUDF, `reconstruct.py`本体）でも使われています。あちらでは同様の「境界沿いの櫛状スパイク」や「内部のカバレッジ不足」が問題になったことはありますか（あちらはTD-16＝局所的な境界集中型の問題が支配的だったため、本系統のような輪郭全周にわたる現象が出ていたとしても見逃されている可能性があります）。

**追加で実施できる切り分け診断（未実施、提案のみ）**:
- 入力点群との比較ではなく、再構成メッシュ**内の隣接面同士**の法線一致度（dihedral angle / 隣接面法線とのcos類似度）を測れば、(a)（予測方向の誤り）と(b)（VTK再構成の局所不整合）を切り分けられるはずです。
- `reconstruct_coarse_surface()`内で`reconstruct_surface()`呼び出し直前の候補点群座標をダンプし、その時点で既に櫛状（スパイク状）に広がっているか、それとも滑らかな点群でVTKが誤って櫛状メッシュを生成しているかを直接確認する（上記「改訂後の有力仮説」セクション参照）。
- カテゴリ別（near/far/boundary）に投影前後の点群分布・密度を可視化し、内部（near由来）と境界（boundary由来）とで候補点の生成・残存数にどれだけ偏りがあるかを定量化する。

いずれも次のセッションで実施予定ですが、こちらで先に着手いただいても構いません。

### Stage A の「4情報」設計は妥当か（ユーザーからの依頼、5フィールドの再評価）

ユーザーから「Stage Aで4つの情報をGTとして生成しているが妥当か」という確認依頼がありました。実際のスキーマは**5フィールド**です（`midsurface_sampler.py`の`_softmin_guidance()`、`SoftminGuidance` NamedTuple）: `soft_potential`（候補ランキング専用、符号付き・負値許容）/ `step_distance`（投影の実移動量、**ブレンドしない厳密hard-min距離**）/ `soft_direction`（投影方向、ブレンド済み）/ `branch_ambiguity`（softmax重みのエントロピー、投影量の減衰に使用）/ `direction_strength`（正規化前のブレンドベクトルのノルム、方向損失の重みに使用）。

検証の上での評価（妥当と判断）:
- **役割分離は理にかなっている**: 単一のUDFに「ランキング・移動量・方向」を全部担わせていた場合に起きがちな問題（移動量自体がsoftminでぼやけてしまう等）を、`step_distance`を厳密hard-min値に固定することで明示的に回避している設計は良い判断です。`soft_potential`が負になりうる設計も、移動には一切使われない（`bounded_project()`内で明示的に検証・排除）ことが確認できているため安全です。
- **`direction_strength`は死蔵フィールドではありません**（当初その懸念がありましたが）。`train_softmin_guidance.py`の`direction_weight = category_weight * direction_strength`（342〜348行付近）に実際に配線されており、方向ベクトルが拮抗する箇所の方向損失を弱める役割を正しく果たしています。データ上もnearカテゴリの88%程度の点でdirection_strength≥0.9と健全に分布しており、この設計自体が今回の配向異常の原因とは考えにくいです。
- **唯一気になる設計上の緊張関係**: 投影は「厳密な移動量(`step_distance`)」と「ブレンドされた方向(`soft_direction`)」という、平滑化のポリシーが異なる2チャンネルを組み合わせています。複数候補枝が拮抗する箇所（direction_strengthがやや低い、near点の約12%）では、ブレンド方向が実際の表面ではなく2枚の面の間の空間を指し得る一方、移動量は単一最近点基準の厳密値のままなので、両者の整合性にズレが生じうる構造です。ただし今回観測された配向異常はメッシュ全体に広く分布しており（direction_strengthはほぼ全域で高い）、この緊張関係だけでは説明しきれません。
- **結論**: 5フィールドへの分解そのものは妥当な設計だと判断します。今回の配向バグの原因はStage AのGT生成ロジックよりも、上記の(a)/(b)の切り分けが必要な下流（学習未収束 or 再構成パラメータ）にある可能性が高いとみています。

---

## 1. 旧§0（未回答・据え置き）: GTの負値フロアは「どの層」で適用すべきか

ユーザーから明示的に「この点をcodexにも意見を求めたい。引継ぎ資料の一番重大な課題としてほしい」との指示があり、本メモの最上段に配置していましたが、Round 15で上記§0（より緊急性が高いと判断した新規論点）に最上段を譲り、ここに繰り下げました。**未回答のままです。**

### 背景（前提知識）

- R7-04: Stage Bのモデル(`QueryDecoder`)の出力ヘッド最終層は`softplus()`で活性化されており、**モデルの予測UDFは構造的に常に正**（負を出力することが原理的に不可能）。
- R7-17（Stage A、`midsurface_sampler.py`）: GT(学習目標)のUDFはsoftmin/LogSumExpブレンドで計算されており、数学的に`softmin(x) ≤ true_min(x)`が成り立つよう設計されている。これは中立面の境界(cut locus)付近の非微分可能なキンクを滑らかにするための意図的な設計だが、表面ごく近傍で複数候補が拮抗すると、本来0以上のはずのUDFが**負値になりうる**。
- 対策A+B（Round 13、`HANDOFF_codex.md`旧版および CLAUDE.md §19「R7-17追補2」参照）で重複枝の二重カウントバグと遠方パララックスバイアスを修正したが、表面近傍(<0.5mm)の負値出力率は60.5%→56.6%とわずかにしか改善しなかった（平均-0.53mm→-0.38mm、最小-2.17mm→-2.01mm）。これは正規のマルチブランチ競合に起因する、対策A+Bのスコープ外の残存課題。

### 今回（Round 14）実装した対策（`train.py`のみ、損失計算時の話）

ユーザー承認「対策D+E併用」に基づき、**学習時(損失計算時)にだけ**作用する2つの緩和策を`weighted_losses()`に追加実装済み（両方ともデフォルト無効でビット完全一致、py_compile・数値サニティチェック済み、実学習は未実施）。

- **対策D**(`--gt-floor-tau-mm`、デフォルト0.0=無効): `gt_for_loss = tau * softplus(gt_udf / tau)` を損失計算の直前にだけ適用。**永続化されたデータセット(h5)も`near_mae`診断指標も、生の(負になりうる)GT値を参照し続ける**。
- **対策E**(`--neg-gt-weight`、デフォルト1.0=無効): `gt_udf < 0`の点について、既存のカテゴリ別損失重みに`neg_gt_weight`を乗算して寄与を下げる。確信度ヘッド(R7-10)の予測値ではなく、**GTデータから決定論的に計算される`gt_udf<0`の判定**を根拠にしている（学習途中の確信度予測を同じステップの重みに使うと循環参照的に不安定になりうるため）。

詳細はCLAUDE.md §19「R7-17追補3」を参照。

### 開かれている設計フォーク（ご意見を伺いたい点）

ユーザーから「StageA時点でGTが負の値を持たないように、学習時と同じロジック(softplus)で近似すべきではないか」という質問があり、Claude側は以下の理由で「現状どおり学習時(train.py)のみに適用する設計」を推奨と回答しましたが、**ユーザーはこの判断にcodexの意見も加えたいとのこと**です。

論点は: **対策Dのsoftplusフロアを、`train.py`の損失計算時だけに適用する（現状の実装）べきか、それとも`midsurface_sampler.py`のStage A側（GT生成・永続化時）に移す/追加すべきか。**

Claudeが整理した主な論拠は以下の通りです。

| 論点 | 学習時のみ（現状の実装） | StageA側に移す案 |
|---|---|---|
| 対策Eとの両立 | 生のGT値が残るため`gt_udf<0`の判定がそのまま使える | StageAで全て正にしてしまうと「元々負だったか」の情報が消え、対策Eが機能しなくなる（別途フラグ/チャンネルの追加保存が必要） |
| 診断の正直さ | `near_mae`・対策A/Bのバイアス検証等、既存の診断はすべて生のsoftmin値を前提にしている。今後の診断でも生値を保てる | StageAで書き換えると、以後「softmin近似自体の歪み」と「フロアによる補正」が混ざって切り分けられなくなる |
| 実験コスト・可逆性 | `tau`値を変えてもCLIフラグを変えて再学習するだけ（同一データセットを再利用可） | `tau`を変えるたびにデータセット全体（Stage A）を再生成する必要がある |
| 一貫性・シンプルさ | 損失計算以外の消費者（将来のStage C等）が生のGT(時々負)をそのまま見ることになる | 全ての消費者が常に「定義どおり0以上のUDF」を見られる。フロア適用漏れのリスクがない |
| 過去の経緯との整合性 | 旧「対策C」(StageAでのハードクランプ`max(gt,0)`)は見送り済み。本案はその時の論点（生データの不可逆な書き換え）がそのまま当てはまる | 実質的に「対策Cのsoftplus版」。ハード/ソフトの違いはあるが適用層の論点は対策Cと同一 |

ハイブリッド案（StageAで生の値と、フロア済みの値の両方を新規フィールドとして保存する）も候補として挙げていますが、データセットのスキーマ・容量が増えるコストがあります。

**codexへのお願い**: 上記の論拠を踏まえた上で、(a) 現状の「学習時のみ」案への賛否、(b) StageA側に移すべきと考える場合はその根拠、(c) 他に気づいた論点（例えばStage C側の消費者への影響、DCUDF経路`reconstruct.py`との整合性など、codexが直接触っている領域の知見）があれば教えてください。最終判断はユーザーが行いますが、判断材料として両エージェントの意見を揃えたいとのことです。

---

## 2. 注意: ドキュメント番号がAGENTS.mdとCLAUDE.mdで分岐しています

`CLAUDE.md`と`AGENTS.md`は元々同一内容でしたが、Round 11以降、**セクション18・19の内容が完全に別物に分岐**しています。

| セクション番号 | `CLAUDE.md`（Claude側の作業） | `AGENTS.md`（codex側の作業） |
|---|---|---|
| §18 | R7-16 Charbonnier平滑化損失 実験結果（Round 11） | 拘束点条件付き再帰型UDFメッシュ生成の実現可能性レビュー（Round 11） |
| §19 | R7-17: softmin/LogSumExp GT-field平滑化（Stage A、Round 12）+ 本Handoffで扱うRound 13・14差分 | 適応細分化上限制御の再設計（Round 12） |

本メモが指す「§18」「§19」は**すべて`CLAUDE.md`側**を指します。混同しないようご注意ください。今後どちらかのファイルに統合するか、番号を採番し直すかはユーザー判断待ちです（本セッションでは未対応）。

## 3. このセッションでやったこと（概要、Round 12〜14分）

タスクリストの `#47`〜`#52`（すべてcompleted）に対応する作業です。中立面UDFのGT（学習目標）を、単一最近傍点のhard-minから、複数候補枝のsoftmin/LogSumExpブレンドに置き換える実験（R7-17）の、品質改善ラウンドです。

1. **R7-17本体実装**（Round 12、`#47`〜`#50`）: `midsurface_sampler.py`に`_softmin_udf_and_grad()`を新設。`softmin_tau_mm=None`（デフォルト）では旧hard-minパスとビット完全一致、有効化時のみsoftminブレンドを使う設計。CLAUDE.md §19に文書化済み。
2. **バイアス分析**（Round 13、`#51`）: ユーザーから「板金から遠い所に異常値が存在しないこと」という品質要件が出たため診断したところ、(a) 厳密最近傍枝とcentroid-kNN候補が重複してsoftmaxを二重カウントするバグ（重複率55.1%）、(b) 重複を除いても残る、距離に応じて単調増加する遠方パララックスバイアス、(c) 表面近傍での物理的に無効な負値出力（最大-2.17mm）、の3つの独立した問題を特定。
3. **対策A+B実装**（Round 13、`#52`）: ユーザー承認（AskUserQuestion「A+B（推奨）: 重複排除 + クリップ半径5mm」）に基づき、
   - **対策A**: 候補枝のうち最近点座標が`1e-4mm`以内で一致するものを貪欲法でマージしてから softmax/log-sum-exp を計算（重複二重カウント排除）。
   - **対策B**: 真の最近傍距離が`clip_radius_mm`（デフォルト5.0mm、新定数`SOFTMIN_CLIP_RADIUS_MM_DEFAULT`）を超えるクエリ点は無条件に厳密hard-minパスにフォールバック（softminブレンドを一切適用しない）。
   を実装。`body002`データセットを再生成し、効果を検証済み。詳細・検証数値はCLAUDE.md §19「R7-17追補2」を参照。
4. **対策D+E実装**（Round 14、§1参照）: 対策A+Bでも解消しなかった表面近傍の負値出力残存課題に対し、ユーザーから「モデルアーキテクチャ側で副作用の少ない形で根絶できないか」という追加検討依頼を受けて調査。R7-04のsoftplus出力ヘッドにより**モデルは構造的に負値を予測できない**ことが判明し、問題はGT側のみにあると整理。`train.py`の損失計算時にのみ作用する対策D（GTスムーズフロア）・対策E（負値GT点の重み低減）をユーザー承認「対策D+E併用」に基づき実装。詳細はCLAUDE.md §19「R7-17追補3」、設計フォーク（StageA vs 学習時）は本メモ§1を参照。
5. **SoftminGuidanceModel 再学習・再構成・診断**（Round 15、本メモ§0参照）: codexのチェックポイント紛失を受けて再学習（2000epoch、チェックポイント保存確認済み）→再構成→`diagnose_softmin_guidance.py`で配向診断。codexの旧再構成と共通する法線配向異常を発見、原因は未確定。

## 4. 触ったファイル（codexのoff-limitsファイルとは非重複）

- `biw_poc/src/preprocess/midsurface_sampler.py`（Round 12〜13） — `_softmin_udf_and_grad()`に`clip_radius_mm`/`dedup_tol_mm`引数追加、新定数`SOFTMIN_CLIP_RADIUS_MM_DEFAULT=5.0`・`SOFTMIN_DEDUP_TOL_MM=1e-4`追加、`sample_udf_queries()`/`sample_boundary_shadow_queries()`/`process_body()`/CLI（`--softmin-clip-radius-mm`）に配線、h5属性・meta JSONに来歴記録追加。
- `biw_poc/src/model/train.py`（Round 14） — `weighted_losses()`に`gt_floor_tau_mm`（対策D）・`neg_gt_weight`（対策E）引数追加、`--gt-floor-tau-mm`・`--neg-gt-weight` CLI引数追加、`run_epoch()`呼び出し配線、モジュールdocstring加筆。両方ともデフォルト無効・ビット完全一致を維持。
- `biw_poc/experiments/loss_smoothing/softmin_r717/`（本ディレクトリ、隔離ワークスペース） — `body002_filled_dataset.h5`を対策A+B適用済みコードで再生成（Round 13）。Round 15で同データセットを使い`checkpoints_softmin_r717_ab/`（新規、21チェックポイント約4.3GB）と`codex_guidance_model/body002_r717ab_recon_midsurface.ply`（新規、claude再学習チェックポイントからの再構成）を追加。修正前データセットは`body002_filled_dataset_preAB.h5`としてバックアップ保持。
- `CLAUDE.md` §19 — 「R7-17追補2」（Round 13）・「R7-17追補3」（Round 14）セクションを追記。Round 15分は本メモのみに記載（CLAUDE.md未反映、後述§7参照）。

**`reconstruct.py` / `dcudf_extract.py` / `dcudf_reconstruct.py` / `hole_filler.py` / `stp_viewer.py` / `Mesh_Generater/output`配下には一切触れていません。** `softmin_guidance.py` / `softmin_guidance_dataset.py` / `train_softmin_guidance.py` / `reconstruct_softmin_guidance.py` / `diagnose_softmin_guidance.py`（codex主導実装）も**読み取り・実行のみ**で編集はしていません。codexの`#32`（DCUDF wiring）・`#46`（remaining-assemblies batch）作業との衝突はないはずです。

## 5. 現在の状態（数値サマリー）

検証は`body002`単体（このディレクトリの`.stp`/`.h5`）に対してのみ実施。

- softmin(x) ≤ true_min(x) の数学的保証: 違反ゼロ（全14,336クエリ点、最大ずれ2.5e-5mm = 浮動小数点誤差レベル）。
- クリップ半径5mm超の領域: バイアス完全ゼロ（5-10mm/10-20mm/20-50mm/50mm+ いずれもmean=0.0000mm）— 対策Bの設計通り。
- クリップ半径内のバイアス: 縮小したが残存（0-1mm帯で0.7469mm→0.6001mm等）— 対策Aは重複カウント分のみ是正、根本のマルチブランチ競合は未解消。
- 表面近傍（真の距離<0.5mm）の負値出力: 60.5%→56.6%、平均-0.53mm→-0.38mm — 対策A+Bのスコープ外の残存課題。
- 対策D+E（Round 14）: `train.py`の合成テンソル(N=8)によるサニティチェックのみ実施済み（デフォルト引数でのビット完全一致、フロアの正値化、`near_mae`が生値を参照し続けること、down-weightによる損失変化を確認）。**実データでの学習実行・Stage C再構成への影響は未検証**。
- SoftminGuidanceModel（Round 15、§0参照）: 2000epoch学習でnear_potential_mae=0.137mm/near_step_mae=0.201mm、メッシュ連結性は大幅改善（主成分95.6%）したが、面法線配向はほぼランダム（mean cos≈0.015、75%が60°超ズレ）のまま。原因未確定。

## 6. 未着手・判断待ちの事項

1. **§0（新規・最重要）**: SoftminGuidanceModel再構成メッシュの法線配向異常。原因切り分け（モデル予測方向 vs VTK再構成の局所不整合）が未実施。codexの知見・追加診断待ち。
2. **§1の設計フォーク**: 対策Dのフロアを学習時(train.py)のみに適用するか、StageA(`midsurface_sampler.py`)側に移す/追加するか。codexの意見待ち、最終判断はユーザー。
3. **学習実行**: 対策D+E適用済み設定での学習と、hard-min版・無印softmin版・対策A+B版との比較。未着手。
4. **Stage-B評価軸の見直し**: near_maeのみに依存しない分布統計＋カバレッジ＋ゴースト異常値チェック。別途要望済み・未着手。
5. **AGENTS.md/CLAUDE.mdのセクション番号分岐の整理**: 本メモ§2参照。未対応。

## 7. このディレクトリのファイル一覧（現状）

```
body002_filled.stp                    元STEPファイル（コピー）
body002_filled_dataset.h5             対策A+B適用後の最新データセット（GT softmin場+guidance 5フィールド含む）
body002_filled_dataset_preAB.h5       対策A+B適用前のバックアップ（重複バグ・パララックスバイアスあり）
body002_filled_midsurface.ply         中立面メッシュ（再生成版、形状自体は変更されていないはず）
body002_filled_midsurface_meta.json   生成パラメータ・来歴記録（softmin_tau_mm/k_faces/clip_radius_mm等）
HANDOFF_codex.md                      本ファイル
checkpoints_softmin_r717_ab/          [Round 15新規] claude再学習チェックポイント（21ファイル、約4.3GB）
codex_guidance_model/
  body002_filled_midsurface.ply               GTコピー（md5一致確認済み）
  body002_guidance_codex_midsurface.ply       codex紛失チェックポイントによる旧再構成
  body002_r717ab_recon_midsurface.ply         [Round 15新規] claude再学習チェックポイントによる再構成
```

なお、Round 15の内容（本メモ§0）はCLAUDE.md側にはまだ転記していません。CLAUDE.mdへの正式な追記（§20候補）はユーザー判断待ちです。
