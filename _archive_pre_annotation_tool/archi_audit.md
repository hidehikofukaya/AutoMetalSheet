# archi.md 実現可能性監査レポート（第3版）

初版監査日: 2026-06-20
第2版改訂日: 2026-06-20（DCUDF白紙・工数見積もり不要のユーザー訂正を反映）
第3版改訂日: 2026-06-20（codexによるarchi.md/AGENTS.md改訂、commit `563b751`をコード照合の上で再監査）

監査対象: `archi.md`（commit `563b751`時点、全2702行）、`AGENTS.md` §20（同commitで追加）
監査範囲: 引き続き`archi.md` §0〜§0I（行1〜1653、文書冒頭のYAML frontmatterで`normative_scope: "§0〜§0I"`と明記）を主対象とする。§1〜§16（行1654〜2702、`legacy_draft_scope`として明記）は対象外。
target_git_commit: `563b751e4ed14150fa91fde9d64af0c0f6162ef6`
ステータス: 実装は未着手。本レポートは監査のみを目的とする。

---

## 改訂履歴（第3版で何をしたか）

codex（別エージェント）が、第2版を入力として`archi.md`と`AGENTS.md`を改訂し、commit `563b751`としてコミットした旨の報告があった。**報告内容を鵜呑みにせず、以下の手順で実コード・実コミット・実ファイルに対して直接照合した。**

1. `git log` / `git show --stat 563b751` でコミットの実在性（AGENTS.md +86/-2、archi.md +610/-67行）を確認した。
2. `git show 563b751 -- archi.md` / `-- AGENTS.md` の差分全文（archi.md側883行、AGENTS.md側108行）を読み、テキストレベルの主張を1件ずつ確認した。
3. `reconstruct.py`の実コード（`Grep`＋`Read`）で、`main()`のargparseデフォルト、`process()`のシグネチャデフォルト、`reconstruct_midsurface()`内のゲート処理を直接確認し、archi.md §0.3.1が主張する「CLI/API/YAML三者の設定分裂」表が正確かどうかを検証した。
4. archi.md §0A.3.B1が新規追加した「候補点ゲート通過数不足時はfail-closedとし、ungated fallbackを禁止する」という記述の根拠となった現行コードの欠陥を、該当行（`reconstruct.py` 514-519行）で直接確認した。
5. archi.md §0.3.1・AGENTS.md §20が新規追加した「保存済みlegacy mesh（coarse/refine3）の再計測値」（頂点数、面積、面積比、vertex/face-connected component数、boundary edge数、最小角、angle p1、aspect ratio p95）を、**保存済みPLYファイルに対して独自にPythonスクリプト（trimesh + scipy.sparse）を書いて再計測し、記載値と突き合わせた**。
6. 改訂後の規範部から、旧い矛盾表現（split/collapse比率の直書き`1.4`/`0.6`、`1M vert`というランタイム目標、`step_confidence`の現役フィールドとしての残存）が実際に除去されているかを、`grep`で機械的に再走査した。

### 検証結果サマリ

| codexの主張 | 検証方法 | 結果 |
|---|---|---|
| CLI(`main()`)とPython API(`process()`)とYAMLでデフォルト値が三者三様に分裂している | `reconstruct.py`の`main()`/`process()`/`reconstruct_midsurface()`シグネチャを直接読み、`yaml`/`config`参照の有無をgrep | **正確**。`main()`: refine_rounds=4, conf_threshold=0.0, input_dist=10.0, prune_dist=20.0。`process()`/`reconstruct_midsurface()`: refine_rounds=0, conf_threshold=0.5, input_dist=None, prune_dist=None。`smooth_iters`はCLI引数化されておらず両経路とも既定0。ファイル全体に`yaml`/`config`読み込みコードは一切なし（`configs/model.yaml`は実行時に読まれない）。 |
| 候補点ゲート通過数が不足すると、ゲートなしの全グリッドへフォールバックする「fail-open」挙動が現行コードに存在する | `reconstruct.py` 514-519行を直接確認 | **正確**。`len(gate_idx) >= nbr_sz*2`を満たさない場合、`WARNING`を出力した上で`pool_idx = np.arange(len(udf))`（全グリッド点）へフォールバックしている。これは初版・第2版いずれの監査でも指摘していなかった、独立した安全上の欠陥である。 |
| 保存済みlegacy mesh（coarse/refine3）の再計測値8項目 | PLYファイルを独自にtrimesh+scipy.sparseで再計算 | **8項目中7項目で完全一致、1項目（aspect ratio p95）は3種類の標準公式いずれでも再現できず**（詳細は§1.A）。頂点数4,080/208,881、面数6,300/400,363、面積344,634.27/950,093.73mm²（比2.76）、boundary edge 2,140/17,745、vertex-connected component 29/31（純粋な頂点隣接グラフで再計算して一致を確認）、face-connected component 41/46、最小角0.0358°/0.0001°、angle p1 3.02°/1.47°はすべて記載値と小数点以下まで一致。 |
| 旧い矛盾表現が規範部から除去されている | `grep`での全文走査 | **正確**。`1.4 * h_target`/`0.6 * h_target`の直書き0件、`1M vert`0件。`step_confidence`は「出力しない／削除した」という否定文脈の2件のみ残存（意図通り）。 |

**結論**: codexの改訂は、自己申告内容と実際のコード・コミット・ファイルがほぼ完全に一致しており、内容のねつ造や誇張は確認されなかった。さらに、codexのサブエージェント（幾何・CAE観点）は**第2版監査が見落としていた実在の安全上の欠陥（候補ゲート不足時のfail-openフォールバック）を独自に発見し是正している**。これは本監査（Claude側）の見落としであり、率直に認める。唯一、aspect ratio p95の数値は独自検証で再現できなかった（§1.A）。

---

## 0. 総合評価（第3版）

**実現可能（Feasible）。第2版の評価を維持する。** codexの改訂により、状態機械の矛盾（予算枯渇時に必須違反があっても`PASS_WITH_WARNINGS`に到達できた経路）、候補ゲートのfail-open挙動、split/collapse比率の文書内不一致、`step_confidence`の用途未定義、GT/input/fieldの指標混同など、第2版で「軽微」としていた項目および第2版が見落としていた項目の多くが是正された。**実装着手のブロッカーは引き続き見当たらない。** ただし、今回のコード照合・再計測を通じて、codexの改訂自体にも新たな指摘事項（§1.A〜§2.E）が見つかったため、無条件の追認はしない。

---

## 1. 第3版の検証作業で見つかった指摘事項

### 1.A（軽微）legacy mesh再計測表の「aspect ratio p95」を独自に再現できなかった

AGENTS.md §20・archi.md §0.3.1の再計測表のうち、頂点数・面数・面積・boundary edge数・vertex/face-connected component数・最小角・angle p1の7項目は、独自のtrimeshベースの再計算で記載値と完全一致した（うちvertex-connected componentは、単純な`trimesh.split()`では面随伴ベースの値しか出ないため、頂点間の純粋な隣接グラフを別途構築して再計算し、初めて記載値29/31と一致することを確認した — 些末だが、再計測自体が単純作業ではないことの記録として残す）。

一方、「aspect ratio p95」（coarse: 13.1、refine3: 29.8）は、以下3種類の標準的な三角形アスペクト比公式のいずれを使っても再現できなかった。

- 最長辺/最短辺比: 8.79 / 12.31
- Verdict系（最長辺 / (2√3·内接円半径)）: 7.71 / 17.29
- 外接半径/内接半径比: 18.33 / 93.29

いずれも記載値（13.1 / 29.8）とは一致しない。これは数値そのものの信頼性に重大な疑義を呈するものではない（実装側が上記以外の定義、例えば面積正規化済みの別系統公式や、メッシュ品質ライブラリ固有のスケーリングを使っている可能性が高く、虚偽の数値という証拠はない）が、**監査としては「8項目中7項目を独立再現できた」という限定付きの裏付けに留める**必要がある。

**推奨アクション**: archi.md/AGENTS.mdの再計測表に、aspect ratioの定義式（使用ライブラリ・関数名を含む）を脚注として明記する。これにより将来の監査が同じ詰まりに当たらずに済む。優先度は低い（このメトリクスは表内の参考値であり、Go/No-Go判定の直接入力にはなっていない）。

---

## 2. 第2版指摘に対するcodexの対応の再検証

第2版で挙げた指摘のうち、codexが「採用」とした項目は、AGENTS.md §20・archi.md §0I.6に挙げられた通り、いずれも実際の本文修正と対応していることをdiffで確認した（§2.1〜3.5系列はNeural Subdivision等の引用追加、PipelineFailureReplayCorpus追加、シーム4段階フォールバック追加、NoveltyAndSupportGate追加、R7-14定量値の本文転記、`step_confidence`の削除として、それぞれ本文に反映されている）。「要実験」とされた15部品クロスチェック（第2版3.1）は、R0プロファイリング項目としてブロッカー扱いしない判断であり、優先度から見て妥当である。

特筆すべき点として、codexは第2版の総合評価（「重大ブロッカーなし」）を**単純に追認せず**、独自に発見した安全上の欠陥（fail-openフォールバック）を根拠に「そのまま採用しない」と明記した上で、`INSUFFICIENT_EVIDENCE`状態とfail-closed契約を新設している。これは監査プロセスとして健全な態度であり、第3版監査としてもこの判断を支持する。

---

## 3. 第3版で新たに見つかった設計上の指摘事項

codexの改訂内容そのものを精査した結果、以下の指摘事項が新たに見つかった。いずれも実装着手のブロッカーではなく、Phase R1（学習コンポーネント）着手前に詰めるべき設計の精緻化事項である。

### 2.A（中程度）equivariance regression testが大域SO(3)のみで、パッチ内回転（面内軸選択）のあいまいさを直接検証していない

§0A.3.D（LocalMeshRefiner）は、パッチごとに法線`n`・接線基底`t1, t2`からなる局所直交フレームを構築し、位置・変位をこのフレームで表現する設計を新規追加した。これ自体は健全な設計だが、同セクションは「主方向が不安定な平坦・等方パッチでは、境界/feature方向を第1接線に使い、それもない場合は複数回転augmentationを行う」と明記しており、**`t1`軸の選択自体に本質的なあいまいさ（面内回転＝SO(2)の自由度）が残ることを文書自身が認識している**。

しかし、直後に定義されるequivariance regression testは「学習・評価にランダムSO(3)回転を入れ、回転後に戻した予測差を測る」とのみ記述されている。局所フレーム構成は定義上、大域SO(3)回転の大部分を自動的に吸収する（パッチが世界座標系に対してどう回転していても、ローカルフレームに変換すれば同じ入力になるよう設計されている）ため、**大域SO(3)テストは多くの場合自明に近い形で通過してしまい、文書が懸念している「`t1`軸選択の不安定性」という、より起こりやすい失敗モードを検出できない可能性がある**。

**推奨修正文案**:
```text
equivariance test (二系統に分離する):

1. global SO(3) test:
   パッチ全体をランダムSO(3)回転 → ローカルフレーム再構築 → 予測 → 逆回転
   既存記述通り。局所フレーム実装のバグ検出が主目的。

2. in-plane (about-normal) consistency test:
   法線を固定したまま、t1/t2選択を複数回独立に再サンプリングする
   （boundary/feature方向が取れない等方・平坦パッチに限定）
   同一入力に対する予測のtangential成分のばらつきを測定する。
   許容ばらつき閾値を超えるパッチ比率をPhase R1のGo条件に加える。
```

### 2.B（中程度）`rank(e)`辞書式タプルに同値判定のepsilonが定義されておらず、決定性要求と矛盾しうる

§0A.3.G「split選択」は、旧来の重み付きスカラーutilityを廃し、`rank(e) = (kernel_eligible, boundary_constraint_regression_zero, expected_geometry_gain, expected_cae_gain, -predicted_vertex_cost, -predicted_runtime)`という辞書式タプル比較に置き換えた。これは§0Bの辞書式最適化方針と整合する健全な変更である。

ただし、`expected_geometry_gain`等は学習済みモデルやヒューリスティクスが出す**連続値**であり、厳密な浮動小数点の辞書式比較をそのまま使うと、上位項目のごく僅かな数値ノイズ差（事実上同値）だけで下位項目（CAE gain、頂点コスト）の大小関係が完全に無視される。これは「上位の僅差は同値とみなし、下位の基準で決める」という辞書式最適化の本来の意図（§0Bで明示されている考え方）から外れ、かつ**Phase R0のGo条件が要求する「同一入力・seed・configでのbyte-identical artifact hash」という決定性要求とも緊張関係になる**（連続値の数値誤差レベルの差が選択結果を変えうるため、わずかな計算順序・浮動小数点丸めの違いがメッシュトポロジーそのものを変える経路になりかねない）。

**推奨修正文案**:
```text
rank(e)の各連続値項目には比較前にtier epsilonを適用する。

geometry_tier(e)  = round(expected_geometry_gain / eps_geom)
cae_tier(e)       = round(expected_cae_gain / eps_cae)

rank(e) = (
  kernel_eligible,
  boundary_constraint_regression_zero,
  geometry_tier(e),
  cae_tier(e),
  -predicted_vertex_cost,
  -predicted_runtime,
)

eps_geom, eps_cae はh_target同様schema-validated configに置き、
評価器のevaluator_numeric_toleranceと整合させる。
```
（`h_target`のEMA更新や停止条件の`improvement_epsilon_ratio`など、本文の他箇所では既にこの「epsilon付き比較」のパターンが使われており、`rank(e)`だけが例外になっている。）

### 2.C（中程度）「held-out 5部品・3 part family」という統計的検証の最小サンプル数が、自部品で利用可能な実測サンプル数に対して過小である可能性

§0E.2は、L1〜L3（学習方式）がB0〜B2（決定論的ベースライン）に対してpaired bootstrap 95% CIで有意に優れることをGo条件とし、「R1 technical Goには最低5 held-out parts・3 part familiesを要求」と定めている。

しかし、§0.4は実STPファイル15部品の実測で、頂点数が256〜18,640（約73倍）というように、**部品ごとの規模・複雑さのばらつきが非常に大きい**ことを既に記録している。この分散の大きさを踏まえると、5部品というサンプルサイズでのpaired bootstrap CIは、母集団のばらつきを十分に捉えられず、CIが極端に広くなる（＝有意差を検出できない）か、逆にたまたま選ばれた5部品の偏りにより誤って「有意」と判定されるリスクのどちらかに振れやすい。本プロジェクトには既に15部品分の実測データ資産があることから、検証データ数をこれより著しく絞る積極的な理由が見当たらない。

**推奨修正文案**:
```text
R1 technical Goのサンプル要件:
  既存の実STP 15部品コーパス（§0.4測定対象）を母集団とし、
  held-out part数は母集団の50%以上（8部品以上）を最低ラインとする。
  5部品で運用する場合は、検出力分析（目標効果量・分散の事前推定からの
  必要サンプル数試算）を添付しなければGo判定に使用しない。
```

### 2.D（軽微）`INSUFFICIENT_VALIDATION_DATA`が§0A.5の統一terminal状態列挙に含まれていない

§0E.2で新設された`INSUFFICIENT_VALIDATION_DATA`（Phase間比較のサンプル不足時に次Phaseへ進まない、という趣旨の状態）は、§0A.5が定義する`RefinementRun`のterminal状態列挙（PASS / PASS_WITH_WARNINGS / INFEASIBLE_MESH_BUDGET / INFEASIBLE_GEOMETRY / INSUFFICIENT_EVIDENCE / OUT_OF_DISTRIBUTION / STALLED_SAFE / OSCILLATION_DETECTED / FAILED_NUMERICALLY）に含まれていない。

文脈上は「個々のRefinementRunの終了状態」ではなく「Phase全体のGo/No-Go判定状態」という別の名前空間を指していると解釈できるため、設計として誤りとは言えない。しかし、いずれも"terminal-state-likeな大文字スネークケース識別子"という同じ命名規則を共有しているため、実装者が誤って同一の状態enumに混在させるリスクがある。**推奨**: 「`RefinementRun` terminal状態」と「Phase Go/No-Go判定状態」が別の名前空間であることを一文明記する（例: `PhaseGateStatus`という別の型名を与える）。

### 2.E（軽微）Phase R3/R5に新設された数値Go条件の出典が未記載（第2版指摘3.2と同型のパターンの再発）

第2版指摘3.2（「現行の最大約46mm」という数値の出典不足）はcodexの改訂で是正されたが（§0I.4に設定条件を明記）、今回新たにPhase R3・R5に追加されたGo条件の数値（例: 「vertex count <= 0.5 * S0」「displacement/reaction/eigenvalue difference <= 5%」）には、同様に導出根拠・参照元が付記されていない。これらは妥当な目安値ではあるが（メッシュ収束スタディで5%前後を閾値とするのは一般的な実務慣行ではある）、出典のない数値を文書内に追加するという、第2版で一度指摘・是正したのと同じパターンが別箇所で繰り返されている。**推奨**: 各Phaseの数値Go条件に、根拠（社内実績／文献／暫定値で次Phaseで再校正、のいずれか）を一行ずつ付記する運用ルールをAGENTS.mdに追加する。

---

## 4. 健全と確認できたcodexの改訂判断（コード照合済み）

| 判断 | 確認方法 | 評価 |
|---|---|---|
| R13-01: ReconstructionProfileによる設定統一（LEGACY_R713/COARSE_GATED/ADAPTIVE_BASELINE） | `reconstruct.py`の実デフォルト値と§0.3.1の表を1行ずつ突合 | 正確。設定分裂の実態を過不足なく記述している。 |
| R13-02: fail-closed証拠ゲート | `reconstruct.py` 514-519行の該当WARNING分岐を直接確認 | 正確かつ重要な追加修正。第2版監査の見落としを補っている。 |
| R13-03: 予算枯渇時のPASS_WITH_WARNINGS到達不能化 | `grep`での全文再走査、新フローチャート（§0A.2）の確認 | 是正を確認。旧フローチャートの矛盾（予算枯渇から直接PASS_WITH_WARNINGSへ到達できた経路）も新フローチャートで解消されている。 |
| R13-07: legacy mesh再計測値 | PLYファイルを独立に再計測 | 7/8項目で記載値と一致（aspect ratio p95を除く、§1.A参照）。データのねつ造を示す証拠はない。 |
| R13-14: `step_confidence`削除 | `grep`で全文中の残存箇所を確認 | 「出力しない」という否定文脈の2箇所のみ残存。能動的フィールドとしての記述は除去済み。 |

---

## 5. 総合評価と推奨アクション（第3版、優先順位順）

**結論**: codexの改訂はおおむね正確で、コード・実測値との照合に耐える（aspect ratio p95の定義式不明という軽微な例外を除く）。特に候補ゲートのfail-open欠陥の発見は、本監査（Claude側）の見落としを補う価値ある追加修正である。一方で、今回のコード照合・再計測を通じて、Phase R1（学習コンポーネント）の設計細部になお詰めるべき点（同変性テストの範囲、辞書式ランキングの数値安定性、統計的検証のサンプル数）が見つかった。これらはPhase R0（決定論的baseline実装）の着手を妨げるものではない。

**P0（Phase R0着手時、追加作業ではなく既存タスクの実施順の確認）**
- 変更なし（第2版のP0を維持）: 現行パイプラインの実際の運用設定を`ReconstructionProfile`として確定し、面積比・反転面積率を実測してからPhase R1の基準値とする。

**P1（Phase R1の設計確定時に対応）**
1. §2.B: `rank(e)`の連続値項目にtier epsilonを導入し、決定性要求との整合を取る。
2. §2.C: Held-out part数の統計的検出力を再検討する（5部品 → 8部品以上、または検出力分析の添付を必須化）。
3. §2.A: equivariance regression testをglobal SO(3)とin-plane（about-normal）の2系統に分離する。

**P2（ドキュメント品質向上、低コスト）**
4. §1.A: 再計測表のaspect ratio定義式を脚注で明記する。
5. §2.D: `INSUFFICIENT_VALIDATION_DATA`が`RefinementRun`terminal状態と別名前空間であることを明記する。
6. §2.E: Phase R3/R5の新規数値Go条件に根拠の一行注記を追加する運用ルールを定める。

本レポートは監査のみを目的としており、上記アクションの実装着手はユーザーの判断を待つ。

---

## 付録: 監査メタデータ（AGENTS.md R13-16要件への対応）

```text
target_file: archi.md, AGENTS.md
target_git_commit: 563b751e4ed14150fa91fde9d64af0c0f6162ef6
normative_scope: archi.md §0〜§0I（行1〜1653）
audit_version: 第3版
auditor: Claude（本セッション）
reviewer: 未定（ユーザー確認待ち）
audit_method: |
  1) git showによるコミット実在性・差分全文確認
  2) reconstruct.py 実コードのGrep/Read照合
  3) 保存済みlegacy mesh PLYの独立再計測（trimesh + scipy.sparse）
  4) grepによる旧矛盾表現の機械的再走査
open_findings: 2.A, 2.B, 2.C（中程度）/ 1.A, 2.D, 2.E（軽微）
approval_status: レビュー入力（正式承認証跡ではない、AGENTS.md正本優先順位に従う）
```
