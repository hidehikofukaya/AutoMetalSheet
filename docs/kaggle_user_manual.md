# Kaggle 学習 作業マニュアル(ユーザー用)

Date: 2026-08-27 / 実装: Opus 5
対象: wtok カーブ主導 AR v2(`curve2.py`)の学習と評価を Kaggle 無料枠で回す手順

---

## 0. 全体像(1分)

```
[ローカル PC]                          [Kaggle]
STEP → wtok 変換 (pythonocc 必須)
  ↓ tools/make_kaggle_bundle.py
kaggle_bundle/                    →   Dataset "wtok-code" (コード)
  wtok_code.zip                   →   Dataset "wtok-synth" (データ 4.2MB)
  wtok_synth_data.zip
  kaggle_notebook.py              →   Notebook にペースト → GPU 学習+評価
                                  ←   Output (best.pt / history.json /
ローカルへ保管・必要時 CATIA 検証        metrics.json / renders/*.png)
```

**重要**: 学習と全ての合否判定は Kaggle だけで完結します。ローカルが要るのは
STEP を触る作業(新チャンクの変換)と CATIA 検証だけです。

---

## 1. 初回セットアップ

### 1-1. バンドル作成(ローカル、1分)

```powershell
cd C:\Users\hide2\IdeaBox\AutoMetalSheet
python tools/make_kaggle_bundle.py
```

`kaggle_bundle/` に3つ出来ます:

| ファイル | 中身 | 再アップのタイミング |
|---|---|---|
| `wtok_code.zip` | 学習コード14ファイル(数十KB) | **コードを直した時** |
| `wtok_synth_data.zip` | 部品2,300件 + val リスト(4.2MB) | **新チャンクを変換した時** |
| `kaggle_notebook.py` | Notebook に貼る内容 | 同上 |

### 1-2. Kaggle に Dataset を2つ作る

1. https://www.kaggle.com/datasets → **New Dataset**
2. `wtok_code.zip` をドロップ → タイトル `wtok-code` → Create
3. 同様に `wtok_synth_data.zip` → タイトル `wtok-synth` → Create

> Kaggle は zip を自動展開します。展開後は `/kaggle/input/wtok-code/cae_mesh_generator/...`、
> `/kaggle/input/wtok-synth/parts/...` になります。

### 1-3. Notebook を作る

1. https://www.kaggle.com/code → **New Notebook**
2. 右パネル **Input → Add Input** で `wtok-code` と `wtok-synth` を追加
3. 右パネル **Settings → Accelerator = GPU T4 x2**(または P100)
4. `kaggle_bundle/kaggle_notebook.py` の中身をセルに貼り付け
5. **Run All**

> **Dataset の展開先について**: Kaggle は zip を `/kaggle/input/<slug>/` に平坦展開する場合と、
> zip 名のサブフォルダを挟む場合があります。notebook は `/kaggle/input` 以下を自動探索して
> `cae_mesh_generator/wtok/curve2.py` と `parts/*.json` を見つけるので、**どちらでも動きます**
> (見つからない時は `/kaggle/input` の中身を一覧表示します)。

---

## 2. 実行と監視

- 進捗はセル出力に `epoch N: nll ... val gen ... half ...` として出ます
- 学習が終わると評価が自動で走り、集計 JSON とレンダリング画像が作られます
- **必ず最後に `Save Version` → `Save & Run All (Commit)`** を押してください。
  これをしないと `/kaggle/working` の成果物(Output)が保存されません

### 見るべき数字

| 指標 | 意味 | 良い方向 |
|---|---|---|
| `val_nll_gen` | 教師強制の困難度 | 下がる |
| `gen_chamfer_mm_median` | 自由生成の形状誤差 | 下がる |
| `edge_ratio_median` | 生成ワイヤー数 ÷ 正解数 | **1.0 に近づく**(v1 は 4.7 = 出しすぎ) |
| `falsepos_mean_mm` | 余計なワイヤーの散らばり | 下がる(v1 の主因) |
| `cov_outline_mean_mm` | 外形のカバー精度 | 下がる |
| `acc_STOP` | 停止判断の正解率 | 上がる(v1 は 0.52) |

---

## 3. 12時間制限をまたいで続ける(セッション分割)

Kaggle は1セッション最大12時間で強制終了します。`MAX_HOURS = 8.5` を入れてあるので、
その前に**自動で安全停止**して Output を確実に残します。続きは:

1. 前回の Notebook で **Output → 右上の … → Create Dataset**
   (または Notebook の Output タブから直接 Dataset 化)。名前例 `wtok-run-prev`
2. 新しい Notebook(または同じものをコピー)で **Input に `wtok-run-prev` を追加**
3. セル冒頭の `RESUME_DIR` を書き換えて Run All:

```python
RESUME_DIR = "/kaggle/input/wtok-run-prev"
```

`history.json` は自動で引き継がれ、エポック番号が連続します(検証済み)。

---

## 4. 無料枠(週30h)が尽きたらローカルで継続

チェックポイント形式は Kaggle とローカルで同一です。**往復動作は検証済み**。

```powershell
# Kaggle の Output(last.pt / history.json)を runs\<任意名>\ に置いてから
cd C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator
$env:PYTHONPATH="src"; $env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m cae_mesh_generator.wtok.curve2 `
  --dataset ..\runs\wtok_synth `
  --val-list ..\runs\wtok_curve_synth_v1\val_names_100.json `
  --output-dir ..\runs\<任意名> --epochs 150 --stage2-after 30 `
  --val-every 5 --sample-every 25 --device cuda `
  --resume ..\runs\<任意名>\last.pt
```

逆(ローカル → Kaggle)も同じで、`runs\<名>\last.pt` を Dataset にして
`RESUME_DIR` に指定するだけです。

> 注意: Kaggle(16GB VRAM)で `--dim` などを変えた場合、ローカル(6GB)では
> OOM することがあります。その時は同じ引数を明示的に指定して下げてください。

---

## 5. 新しい合成データを追加する時(ローカル作業が必須)

```powershell
# 1) PartMaker 側で chunk_05 等が完成したら、変換リストに追加
#    cae_mesh_generator/src/cae_mesh_generator/wtok/build_synthetic.py の GROUPS に
#    ("prod02/chunk_05", "p2c05") を追記(※生成途中のチャンクは絶対に入れない)

# 2) 変換(既存分はスキップされる)
cd C:\Users\hide2\IdeaBox\AutoMetalSheet\cae_mesh_generator
$env:PYTHONPATH="src"; $env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m cae_mesh_generator.wtok.build_synthetic --output-dir ..\runs\wtok_synth

# 3) バンドル再作成 → Kaggle の wtok-synth Dataset に「New Version」でアップ
cd ..
python tools/make_kaggle_bundle.py
```

**val リストは固定です**(`val_names_100.json`)。データを足しても検証部品は変わらないので、
過去の実験と数字を直接比較できます。新部品は自動的に train に入ります。

---

## 6. 週30h クォータの目安と節約

- 確認場所: Kaggle 右上のアバター → **Settings** の GPU 使用量、または Notebook 実行中の右下
- 目安: 2,300部品 × 150エポックで **1〜2時間程度**(T4)。枠は十分です
- 節約のコツ: **アブレーション(条件比較)は `--epochs 50` の短縮版で先に選別**し、
  勝った設定だけ本番エポック数で回す

---

## 7. 困った時

| 症状 | 対処 |
|---|---|
| `No module named cae_mesh_generator` | **最新の notebook はパスを自動探索するので通常起きません**。出た場合は Input に `wtok-code` を追加し忘れています。セル冒頭が `/kaggle/input` の中身を一覧表示するので、実際の展開先を確認できます |
| `could not locate the code/data dataset` | 該当 Dataset が Input に未追加。一覧表示された実パスを見て追加する |
| `No such file .../parts` | `wtok-synth` を追加したか確認 |
| GPU が使われていない(遅い) | Settings → Accelerator が None になっていないか |
| Output が残らない | **Save Version → Save & Run All (Commit)** を押し忘れ |
| 途中で12h切れした | `MAX_HOURS` を下げる(例 7.5)。§3 の手順で再開 |
