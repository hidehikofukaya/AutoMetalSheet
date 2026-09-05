# docs — ここから読む

2026-09-05 に 72 ファイルを 4 分類・22 ファイルに統合した。統合ファイルは元文書を**そのまま**順に収録しているので、
`KB 21.24` のような旧番号の参照は対応ファイル内を検索すれば見つかる(対応表は末尾)。
開発の哲学と評価の習慣は `../CLAUDE.md`。

| 分類 | 何が入っているか | 誰が読むか |
|---|---|---|
| **decisions/** | 設計書と、その決定の履歴(承認・撤回・採否) | 実装を変える人。まず `roadmap.md` |
| **knowledge/** | 研究で得た知見の全部(課題・表現・施策台帳・測定事故・法則・評価・教師・過去アーキ) | 新しい案を出す前に |
| **learning/** | 仕組みを学ぶための資料(UML 付き解説、サーベイ、スケーリング) | 理解したい人 |
| **tips/** | 運用ノート(計算資源、ワークスペース、ツール) | 手を動かす人 |
| `_archive/` | 退役した文書(AE 時代の報告、往復文)。要約は `knowledge/08` | 経緯を掘る人だけ |

## decisions/
| ファイル | 内容 |
|---|---|
| `roadmap.md` | 現在地・ボトルネック B1〜B6・ロードマップ P0〜P6・方針の更新(旧 KB 22) |
| `face_loops.md` | 面ループ表現の設計と 21.0〜21.24 の全決定履歴(旧 KB 21)。**現行の曲げ/面の設計書** |
| `bend_design.md` | 曲げ表現の設計と棄却測定(旧 KB 20) |
| `language_integration.md` | 言語モデル統合 PoC 計画(旧 KB 23) |
| `staged_generator_design.md` | 段階生成器(外形→曲げ→面)とメッシュ経由設計 |
| `requests_partmaker.md` | データ側との依頼・回答 5 通 |
| `rejected_alternatives.md` | 案 A/B/C、反復ループ、次期案(棄却・保留) |

## knowledge/
| ファイル | 旧番号 |
|---|---|
| `01_task_and_data.md` | KB 01, 09 |
| `02_representations.md` | KB 03, 10, 14, 15 + 理論メモ 3 本 |
| `03_experiment_ledger.md` | KB 04, 06, 07, 08 + research_log / overnight / progress / session_knowledge |
| `04_measurement_pitfalls.md` | KB 05 |
| `05_laws.md` | KB 12(条件付け), 11(汎用性監査), 19(汎用性と針) |
| `06_rationality_eval.md` | KB 16, 17 |
| `07_teacher_data.md` | KB 13, 18 |
| `08_past_architectures.md` | KB 02 + current_architecture / retrieval_floor / frontier_gaps + 退役文書の要約 |

## learning/
`01_generative_model_explained.md`(UML 8 枚) / `02_survey.md` / `03_scaling_and_capacity.md`

## tips/
`compute.md`(vast.ai + Kaggle) / `workspace.md` / `annotation_tool.md`

## 旧 KB 番号 → 新ファイル
00→この README / 01,09→knowledge/01 / 02→knowledge/08 / 03,10,14,15→knowledge/02 / 04,06,07,08→knowledge/03 /
05→knowledge/04 / 11,12,19→knowledge/05 / 13,18→knowledge/07 / 16,17→knowledge/06 / 20→decisions/bend_design /
21→decisions/face_loops / 22→decisions/roadmap / 23→decisions/language_integration
