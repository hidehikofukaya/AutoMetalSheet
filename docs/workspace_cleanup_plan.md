# ワークスペース削除計画(実行前の承認待ち)

Date: 2026-08-27 / 調査: Opus 5
方針: **やや安全目** — 実験の記録(JSON/PNG)と現行データは一切消さず、
再生成可能なバイナリと到達不能な git ゴミだけを対象にする。

実行: `python tools/cleanup_workspace.py`(既定は dry-run)
      `python tools/cleanup_workspace.py --apply g1,t1,t2,t3,t4,g2`

---

## 0. 結論(先に数字)

| 現状 | 削除後(全 tier 適用時) |
|---|---|
| **63 GB** | **約 10 GB** |

内訳: git のゴミ **約 50 GB** + runs/アーカイブの再生成可能バイナリ **5.3 GB**。

---

## 1. 最大の発見: `.git` が 55.6 GB(リポジトリの 88%)

```
git count-objects -vH  →  count: 17,480 / size: 52.64 GiB / size-pack: 0 bytes
到達可能オブジェクト     →   5,958 個 / 実質 3.3 GB(非圧縮)
```

- **オブジェクトの 66%(11,522個)が到達不能なゴミ**。2026-08-10 の `git gc` が
  中断した痕跡(`.git/objects/pack/tmp_pack_v5MKtQ` = 1.8 GB が残存)があり、
  パックが一つも作られていない(`size-pack: 0`)ため、全履歴が
  未圧縮の loose object のまま積み上がっている
- コミットは12個、ブランチは main と feat/softmin-guidance-poc の2本のみ。
  **本来の履歴は 3.3 GB 相当**しかない

### 安全性の確認(済)

- `git gc` は**到達可能な履歴を一切消さない**。未 push のコミット
  (main が9件、feat が1件先行)もブランチ ref から到達可能なので安全
- stash は 0 件、退避中の作業なし
- 両ブランチとも origin に存在するため、万一の際の二重の保険もある

---

## 2. 削除 tier(安全な順)

| tier | 内容 | 削減 | 残すもの |
|---|---|---:|---|
| **g1** | 中断 gc の残骸 `tmp_pack_v5MKtQ` | 1,803 MB | — (git が必要時に再生成) |
| **t1** | smoke ラン7個(2-3エポックの使い捨て) | 288 MB | — |
| **t2** | 打ち切り系列の `*.pt` 22個<br>(wireflow=点群flow、wtok_ar=頂点段AR) | 2,433 MB | **history.json / metrics.json / renders は全て保持** |
| **t3** | 保持モデルの best 以外の `*.pt` 3個 | 379 MB | 各系列の `best.pt` |
| **t4** | biw_poc アーカイブの `.ply/.pt/.h5` 70個 | 389 MB | ソース・設定(git 履歴にも存在) |
| **g2** | `git gc --prune=now` | **約 48 GB** | 12コミット・2ブランチの全履歴 |

### t2 の根拠(なぜ消してよいか)

- **wireflow(点群 flow)**: 純AEでも床の12.7倍という表現の天井を実測し、路線終了が
  確定済み。結論は `project_wireflow_cga_status` と docs に記録済み
- **wtok_ar(頂点段 AR)**: 「点の霧」でカーブ主導へ移行済み。同上
- チェックポイントを再度読む予定はなく、**科学的価値のある記録(損失曲線・
  評価指標・レンダリング画像)は JSON/PNG 側に全て残る**

---

## 3. 保持するもの(削除対象外)

| 対象 | 理由 |
|---|---|
| `runs/wtok_synth/` (272 MB) | **現行の学習データ**(2,300部品 + 中間ワイヤーフレーム) |
| `runs/wtok_dataset/` (19 MB) | 実部品140件の変換済みデータ。再生成に CATIA/OCC が必要 |
| `runs/cga_dataset_mesh/` (10 MB) | 測地距離データ。小さく、再計算が重い |
| `runs/twopoint_*` (30 MB) | パラメトリック路線の結果(feasibility 検証器として今後も参照) |
| 各系列の `best.pt` | v1 との比較基準として現役 |
| 全 `history.json` / `metrics.json` / `renders/` | 実験記録そのもの |
| `_archive_pre_annotation_tool/` のソース | git 履歴にもあるが二重に保持 |

---

## 4. 推奨実行順

```powershell
cd C:\Users\hide2\IdeaBox\AutoMetalSheet
python tools/cleanup_workspace.py                      # まず確認
python tools/cleanup_workspace.py --apply g1           # 1.8GB (最も安全)
python tools/cleanup_workspace.py --apply t1,t2,t3,t4  # 3.5GB
python tools/cleanup_workspace.py --apply g2           # 約48GB (数分かかる)
```

g2 は `git gc --prune=now` を実行し、完了後に `git count-objects -vH` を表示します。
時間がかかる場合は `--prune=now` を外した通常の `git gc` でも、対象オブジェクトは
2週間以上前(8/10以前)なので同じだけ回収できます。

---

## 5. 今後の再発防止(任意)

`.gitignore` に `runs/` が入っていないため、`git add .` を実行すると数 GB の
チェックポイントがステージされる危険があります。以下の追記を推奨:

```gitignore
runs/
kaggle_bundle/
_archive_pre_annotation_tool/
```
