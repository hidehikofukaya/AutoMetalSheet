# 知識ベース — ここから読む

締結点2点のみから板金中立面のワイヤーフレームをE2E生成する研究の、全知見の索引。
**この KB を上から順に読めば、過去の全経緯にキャッチアップできる**ように書いてある。

Date: 2026-08-30
Branch: `feat/softmin-guidance-poc`

---

**開発の哲学と評価の習慣は `CLAUDE.md` にある。**この KB は経緯・数値・記録を置く。

## 読む順序

| # | ファイル | 何が書いてあるか | 誰が読むべきか |
|---|---|---|---|
| 1 | [01_task_and_data.md](01_task_and_data.md) | 課題定義、データ、**情報限界**、入力に何が入っているか | 全員。最初に読む |
| 2 | [02_architecture.md](02_architecture.md) | 現在のアーキテクチャ(mermaid)、各器の責務 | 実装する人 |
| 3 | [03_representations.md](03_representations.md) | **本研究の中心的な物語**。表現をどう変えてきたか | 全員 |
| 4 | [04_experiment_ledger.md](04_experiment_ledger.md) | 試した施策の全記録(目的/結果/原因/採否) | 新しい案を出す前に |
| 5 | [05_measurement_pitfalls.md](05_measurement_pitfalls.md) | **指標が嘘をついた事例集**と、方法論上の誤り | 評価する人。必読 |
| 6 | [06_status_and_numbers.md](06_status_and_numbers.md) | 現在の数値、チェックポイント、再現コマンド | 引き継ぐ人 |
| 7 | [07_plan_8h.md](07_plan_8h.md) | 分岐付き作業計画 | 次に作業する人 |
| 8 | [08_night_202608_30.md](08_night_202608_30.md) | 2026-08-30 夜間の実施記録と判定 | 続きをやる人 |
| 9 | [09_generalisation.md](09_generalisation.md) | 穴・ビードへの一般化と、データ側への依頼 | 抽出を拡張する人 |
| 10 | [10_sidecar_finding.md](10_sidecar_finding.md) | **曲げ表現が根本的に誤っていた。**生成物の90.6%はビードの稜線 | 曲げに関わる人。必読 |
| 11 | [11_generality_audit.md](11_generality_audit.md) | 特定形状への特化の洗い出し。構造的な問題と上限の切り分け | 設計を変える人 |
| 12 | [12_conditioning_rule.md](12_conditioning_rule.md) | **導出できる情報は効かない。**4度の失敗と1つの成功 | 条件を足そうとする人。必読 |
| 13 | [13_mechanics_geometry.md](13_mechanics_geometry.md) | 材料力学×幾何の整理。何を入力に、何を教師信号にできるか | 情報を増やそうとする人 |
| 14 | [14_delta_tokens.md](14_delta_tokens.md) | 変位トークン。**保留中、次は曲げで試す** | 続きをやる人 |
| 15 | [15_micro_structure_design.md](15_micro_structure_design.md) | 微細構造の設計と結果(A✗ B△ C○、関係損失✗) | 経緯を追う人 |
| 16 | [16_engineering_structure.md](16_engineering_structure.md) | 目標の再定義。is_arcは許容差の産物。教師の正準化、展開図表現 | 次に実装する人 |
| 18 | [18_catia_edges.md](18_catia_edges.md) | **設計提案: CATIAのエッジ分割を教師にする。**再当てはめ角の62%はCATIA頂点に無い。接合G1チャンネル | 承認待ち |
| 17 | [17_rationality.md](17_rationality.md) | **合理性 = 形の理由を説明できること。**3層評価(製造可能性/簡潔性/機能)。座面余裕比0.763という不合格 | **次に実装する人。必読** |

---

## 3行でいうと

1. **表現に性質を埋め込むと勝ち、損失に書くと負ける。** 閉じたループ・線らしさ・平面性のすべてでこのパターンが再現した。
2. **数値は3回嘘をつき、絵は5回真実を見せた。**ただし絵も2回嘘をついた(存在しない欠陥を見せた)。両方必要。
3. **現状の誤差はモデルの当てはめ不足ではなく条件の情報不足。**過学習ゼロ、訓練誤差>検証誤差。効く手は条件を増やすことだけ。

---

## 現在地(2026-08-30)

```mermaid
flowchart LR
    A["締結点2点<br/>位置・軸・間隔"] --> B["外形フレーム生成器<br/>32辺スロット x 11ch<br/>runs/frame3"]
    B --> C["法線整合<br/>lam=2"]
    C --> D["9回生成 → 中央選択"]
    D --> E["外形ワイヤーフレーム<br/>閉ループ・直線と円弧<br/>5.47mm"]
    E --> F["曲げフレーム生成器<br/>24スロット x 11ch<br/>runs/bendframe1<br/>浮遊ゼロ・折りが28%短い"]
    F -.未着手.-> G["面"]

    style E fill:#d5f5e3
    style F fill:#d5f5e3
    style G fill:#fadbd8
```

**完了**: 外形のパラメトリック生成(閉ループ保証、CADに直接渡せる形)
**完了**: 曲げフレーム生成(浮遊ゼロ、シルエット内100%。折りが28%短いのが残課題)
**進行中**: 一括生成器の点群を条件に加える往復構成(第1回は不合格、第2回実行中)
**未着手**: 面
**データ待ち**: 穴・ビード・変曲の生成(表現は一般化済み、学習データに1件も無い)
→ `docs/requests/2026-08-30_wireframe_extraction.md`

---

## 過去の資料(統合前)

この KB は以下を統合・要約している。原典が必要なときだけ参照。

| 資料 | 位置づけ |
|---|---|
| `docs/research_log.md` | 施策ごとの生ログ。KB 04 の原典 |
| `docs/staged_generator_design.md` | 段階生成器の設計判断。KB 02 の原典 |
| `docs/stage2_bend_result.md` | 曲げ点群方式の詳細。KB 04 に要約 |
| `docs/loop_architecture_proposal.md` | 2周ループ構成の検討 |
| `docs/session_knowledge_202608.md` | セッション知見 |
| `docs/current_architecture.md` | 旧アーキテクチャ図 |
| `docs/progress_202608.md` | 進捗ログ |
| `docs/architecture_survey_202608.md` | 2025-26サーベイ |
| `docs/frontier_gaps_and_reform.md` | 改革案 |
| `docs/final_wireframe_theory.md` ほか cae_*, fable5_*, plan_* | **歴史的資料**。現行設計には直接つながらない |

`~/.claude/projects/.../memory/MEMORY.md` にも横断的な知見がある(CATIA COM、hole_filler、包絡条件のリーク等)。
