# 計算資源の運用(vast.ai / Kaggle)

> 統合日 2026-09-05。以下は元文書を順に**そのまま**収録(見出し番号は元のまま。`KB 21.24` のような参照はこのファイル内を検索)。
> 収録元: `vast_ai_setup.md`, `kaggle_user_manual.md`, `kaggle_training_handoff.md`


---

<!-- 元文書: vast_ai_setup.md -->

# vast.ai 運用手順(2026-09-04)

役割分担(合意済み): ① Kaggle = PoC(探索)、② ローカル GPU = 1 ジョブ上限で評価・診断と短い学習、
③ vast.ai = 本番(**多部品 × 複数シードを 1 インスタンスで並列**)。評価器は常にローカル。

---

## 1. ユーザーがやること(vast.ai 側)

### 初回のみ
1. https://vast.ai でアカウント作成、クレジット入金(まず $10〜20)
2. **SSH 公開鍵を登録**: Account → SSH Keys。ローカルに鍵が無ければ PowerShell で
   ```
   ssh-keygen -t ed25519 -C vast
   type $env:USERPROFILE\.ssh\id_ed25519.pub
   ```
   の出力を貼る
3. **API キー**を取得(Account → API Key)し、`vastai` CLI を入れる:
   ```
   pip install vastai
   vastai set api-key <キー>
   ```
   (CLI は任意。Web UI だけでも運用できる)

### インスタンスを借りるとき
4. Web UI の **Search** で条件を入れる(価格より効く順):
   - GPU: **RTX 4070 Ti / 4070 / 4060 Ti 16GB**(本番)、**RTX 3060 12GB**(探索・単発)
   - **vCPU ≥ 8(12 推奨)、RAM ≥ 24GB**、Disk ≥ 60GB(NVMe)
   - Reliability ≥ 0.98、Verified、DL 帯域 ≥ 200 Mbps
   - 種別: 本番は **On-demand**、短いスイープは **Interruptible**(`--resume` で復帰できる)
   - Image/Template: **PyTorch(最新 CUDA 12 系)**、Launch mode: **SSH**
   CLI なら:
   ```
   vastai search offers 'gpu_name in [RTX_4070_Ti,RTX_4070,RTX_3060] cpu_cores>=8 cpu_ram>=24 reliability>0.98 verified=true inet_down>200 disk_space>=60' -o dph
   vastai create instance <offer_id> --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime --disk 60 --ssh
   ```
5. 起動後、Instances で **SSH 接続コマンド**(`ssh -p <port> root@<host>`)を控える
6. 接続コマンド(ホスト・ポート)を私に渡す → 以降の転送・起動・回収は私が行う
   (私が実行できない場合は §3 のコマンドをそのまま打つ)
7. 終わったら **Destroy**(Stop だけだとディスク課金が続く)

### 毎回の確認
- 起動直後に `nvidia-smi` と `nproc` で GPU と vCPU が広告どおりか見る(違う機体はすぐ Destroy)
- 課金は時間単位。本番 1 日 ≈ $2〜4(4070 Ti)、$1.5(3060)

---

## 2. こちら(AutoMetalSheet 側)が用意済みのもの

| 物 | 場所 | 内容 |
|---|---|---|
| `--workers` フラグ | `wtok/staged.py` | DataLoader 並列(Linux で 8〜12)。ローカルは 0 のまま |
| 複数アーム起動 | `tools/vast/run_arms.py` | sweep JSON の各アームを 1 GPU 上で同時実行(`--parallel 4`)。`last.pt` があれば自動 `--resume` |
| 初期設定 | `tools/vast/setup.sh` | zip 展開・依存確認・機体確認 |
| スイープ例 | `tools/vast/sweep_example.json` | 2b sib×2 シード + 2b 基準 + 2a seed1(すべて 2677 部品) |
| スペック表 | `tools/export_spec_vectors.py` | PartMaker が見えない機体向け(`spec_vectors.json`)。新チャンク取り込み後に再実行 |
| バンドル | `tools/make_kaggle_bundle.py --data-dir runs/wtok_synth_g1 --val-list runs/wtok_synth_g1/val_names_100.json` | `kaggle_bundle/wtok_code.zip`(コード)+ `wtok_synth_data.zip`(parts / face_targets / spec / val リスト 3 種)。Kaggle と共通 |

---

## 3. 1 サイクルの手順(私が実行。手動なら同じコマンド)

```bash
# 0. ローカルで最新化
python tools/export_spec_vectors.py --wtok runs/wtok_synth_g1
python tools/make_kaggle_bundle.py --data-dir runs/wtok_synth_g1 --val-list runs/wtok_synth_g1/val_names_100.json

# 1. 転送(PowerShell / Git Bash)
scp -P <port> kaggle_bundle/wtok_code.zip kaggle_bundle/wtok_synth_data.zip tools/vast/setup.sh tools/vast/run_arms.py tools/vast/sweep_example.json root@<host>:/workspace/upload/

# 2. 機体で初期設定と起動
ssh -p <port> root@<host>
bash /workspace/upload/setup.sh
mkdir -p /workspace/code/tools/vast && cp /workspace/upload/run_arms.py /workspace/code/tools/vast/
cd /workspace/code && nohup python tools/vast/run_arms.py /workspace/upload/sweep_example.json \
    --data /workspace/data --out /workspace/runs --workers 8 --parallel 4 > /workspace/runs/arms.log 2>&1 &
tail -f /workspace/runs/arms.log        # 各アームは /workspace/runs/<tag>.log

# 3. 回収(ローカルから。ckpt と履歴だけ。評価はローカルで)
scp -P <port> -r root@<host>:/workspace/runs/<tag> runs/vast_<tag>
python -m cae_mesh_generator.wtok.face_eval --ckpt2a runs/faceset_2677/best.pt --ckpt2b runs/vast_<tag>/best.pt --ring-k 3 --ring-despike --val-list val_flange_30.json
```

---

## 4. 昇格ゲート(Kaggle/ローカルの PoC → vast 本番)

1. オラクル分解(D1/D2)で「効く余地」が見えている
2. PoC で相対改善がノイズ幅(近傍 ±0.2mm、余剰 ±0.2×、未整合 ±1pt)を超えた
3. 本番は seed ≥ 2 で回し、平均で判定する(習慣 B1/B2)

## 5. 初回運用で分かったこと(2026-09-04、インスタンス 49803200、RTX 3060 12GB、$0.058/h)

- **CLI は 2FA セッションが要る**: `vastai tfa send-sms` → `vastai tfa login --method-type sms --secret <S> -c <CODE>`。
  API キーだけでは `show instances` も 401
- **SSH は直接ポートで**: `ssh3.vast.ai:<port>` のプロキシは `kex_exchange_identification: Connection closed` で入れなかった
  (Jupyter モードで作った機体)。`vastai show instances --raw` の `ports['22/tcp'][0]['HostPort']` と `public_ipaddr` へ直接
  `ssh -p <HostPort> root@<ip>` で入れる
- **base-image には torch が無い**: `source /venv/main/bin/activate && uv pip install torch numpy scipy matplotlib`(2〜3 分)。
  非対話 ssh では `python` が無いので venv を activate してから使う。機体のガイドは `/etc/vast-agents-guide.md`
- **`cpu_cores_effective` を見る**(この機体は 56 コア表示で実効 9.3)。`--workers 2 × --parallel 4` で十分
- **VRAM**: 2b@2677(batch 64、兄弟文脈あり)は **4.1GB/アーム**、2a は 1.1GB。12GB で 2b×3 + 2a×1 が上限(11.4GB)。
  4 アーム同時だと GPU が 100% になり、2a は 61s/epoch(ローカル単独 17s、3060 単独見込み ~8s)。
  **演算が飽和するので「同時アーム数」は 12GB/3060 では 3 が実効的**。4070 Ti 以上なら 4〜5
- `/workspace` は volume ではない(recycle/destroy で消える)。結果は必ず回収する

## 6. 注意

- Interruptible で落ちたら同じ sweep を再投入するだけ(`last.pt` から再開)
- 1 アーム ≈ 2〜2.5GB VRAM。12GB なら `--parallel 4`、8GB なら 3、16GB 以上なら 6
- `OMP_NUM_THREADS=2` を各アームに設定済み(numpy が全コアを奪い合わないため)
- `mesh_synth` の npz は学習に不要(曲率オラクル `--surf-curv` を使うときだけ)

## 7. GPU 選定の規則(2026-09-05、ユーザー方針): **ジョブ総額で選ぶ**

時間単価ではなく **総額 = 単価 × ジョブ時間** で選ぶ。ジョブ時間はこのモデル(11.8M、系列 ≤176)では GPU ではなく
**メインプロセスの CPU クロック**(バッチ整形・ステップのオーバーヘッド)で決まる。実測(5 族外形 3650 部品、1 epoch):

| 機体 | CPU | 単価 | 1 epoch | 1200 ep の総額 |
|---|---|---|---|---|
| RTX 4070 Ti(ID 49432279) | 4.7GHz、実効 16 | $0.108 | **6.9s** | **$0.25** |
| RTX 4080 | Threadripper 3.5GHz、実効 16 | $0.196 | 9.1s | $0.60 |
| RTX 4070 Ti(ID 47037349) | Xeon E5 2.3GHz(実効 1.2GHz)、実効 18 | $0.114 | 20.5s | $0.76 |
| RTX 3060(4 アーム共有時) | 実効 9.3 | $0.058 | (2b 566s) | 割高(上限内に終わらない) |

手順: `vastai search offers ... -o dph --raw` で候補を取り、`cpu_ghz` と `cpu_cores_effective` を見て
**予想 epoch 時間 ≈ 6.9s × (4.7 / cpu_ghz)** で総額を出し、最小を選ぶ。GPU の世代差(4070 Ti / 4080 / 4090)は
この負荷では 1.3 倍以内で、CPU クロックの差(2〜4 倍)の方が大きい。アーム数を増やすときは VRAM(2〜4GB/アーム)と
実効コア数(3 プロセス/アーム)で頭打ちを見る。


---

<!-- 元文書: kaggle_user_manual.md -->

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


---

<!-- 元文書: kaggle_training_handoff.md -->

# Kaggle 学習移行 — Opus 作業ハンドオフ

Date: 2026-08-27
Author: fable5(ユーザーとの決定事項の記録+Opus への作業指示)
Status: **実装完了(Opus 5、2026-08-27)**。実装結果は §8 に追記。
本書が唯一の正。運用手順は docs/kaggle_user_manual.md。

## 0. 背景(1分で)

E2E 板金ワイヤーフレーム生成(wtok カーブ主導 AR)を合成2点部品で学習中。
v1(`runs/wtok_curve_synth_v1`、ep~205 で停止)の診断で「教師強制は良いが自由走行が
複利誤差で破綻(exposure bias)」と確定し、3処方を実装した **v2 = `cae_mesh_generator/src/cae_mesh_generator/wtok/curve2.py`**
(相対座標化+ソフトターゲットCE+エッジ数自己条件付け。スモーク済: オフセット往復20/20)。
ローカル GPU(GTX 1660 SUPER)では 90s/epoch と遅いため、**v2 以降の学習は Kaggle 無料枠で行う**。

## 1. 決定事項(変更には ユーザー承認が必要)

| 項目 | 決定 |
|---|---|
| val セット | **100部品に縮小**。`runs/wtok_curve_synth_v1/val_names_100.json`(旧130名を sorted して先頭100 — 決定的規則。残り30は train へ) |
| 学習対象 | curve2.py(CurveAR2)。プロトコルは v1 と同一(dropout 0.1 / ミラー / ジッタ±1bin / stage2-after 30 / accum 8) |
| データ | `runs/wtok_synth/parts`(現在2,300 JSON、**27MB** — Kaggle Dataset として容易にアップ可能)。チャンク追加時は zip を再アップロードし Dataset の新バージョンを作る |
| エポック | 150 を初期目標(v1 は ep165 で過学習開始した知見)。Kaggle 速度次第で200まで |
| 比較基準 | v1 best(ep165)と同一評価プロトコル(val 100 で再計算) |
| GPU方針 | Kaggle 無料枠(週30h、1セッション最大12h)。ローカルは фallback(§4) |

## 2. タスク1: Kaggle 学習コード+環境+ユーザーマニュアル

### 2.1 最重要の技術的罠: OCC 依存の分離

curve2.py の import 連鎖に **pythonocc(Kaggle に無い)** が混入している:
`curve2 → dataset_curve/codec → convert.py が OCC をモジュールレベル import`。
学習・サンプリング・Chamfer 評価自体は OCC 不要(純 numpy/torch)。

**対処(必須)**: `convert.py` から定数(`BITS`, `HI_BITS`, `E_ARITY` 等)を OCC 非依存の
小モジュール(例 `wtok/constants.py`)へ抽出し、`codec.py`/`dataset_curve.py`/`curve2.py` の
import を差し替える。または convert.py の OCC import を関数内へ遅延させる。
**ローカルの既存パイプライン(build_synthetic 等)を壊さないこと** — 変更後にローカルで
`build_curve_item2` のスモークと `python -m ...wtok.build_synthetic --limit 2` を必ず再実行。

他のローカル依存: `wireflow.dataset.stable_seed`(zlib のみ)、`wireflow.train.safe_save`
(純torch)— これらは同梱可能。**Windows 絶対パスのハードコード**(dataset_curve の
DEFAULT 系)は引数化されているか確認。

### 2.2 成果物

1. **自己完結の学習スクリプト or ノートブック**(推奨: リポジトリの wtok パッケージ最小サブセットを
   zip して Kaggle Dataset 化し、ノートブックは `!pip 不要 / sys.path.append / 学習コマンド` の薄いラッパ)
2. **チェックポイント運用**: 12h セッション制限があるため
   - `--val-every 5` 毎に last.pt/best.pt/history.json を `/kaggle/working` へ(safe_save は流用可)
   - セッション終了前に Output として保存 → 次セッションは Output を入力 Dataset にして `--resume`
   - **9h でセーフ終了するタイマー**(epochs を打ち切って正常 exit)を入れると Output 保存が確実
3. **ユーザー作業マニュアル**(md)。手順:
   - Kaggle Dataset 作成(wtok_synth_parts.zip + val_names_100.json + コードzip)
   - Notebook 新規作成 → Dataset を Attach → Settings で GPU(T4×2 or P100)選択
   - 実行・進捗確認(history.json の見方)・Output のダウンロード
   - チャンク追加時の Dataset バージョン更新手順
   - 週30h クォータの確認場所(Profile → Settings)
4. **torch バージョン差の確認**: ローカル 2.6.0+cu124 / Kaggle は異なる可能性。
   checkpoint は state_dict のみ(既にそうなっている)なので互換のはずだが、
   ロード時 `weights_only=False` の挙動差に注意。最初の resume 往復(K→L, L→K)を必ずテスト

## 3. タスク2: 評価の Kaggle / ローカル分別

### Kaggle で完結するもの(GT は変換済み JSON に自足)

- val NLL(z染み分けなし、教師強制)
- **トークン種別精度**(TAU/NEW/PTR/COORD/STOP — v1 診断スクリプト同等品を移植)
- 自由走行サンプリング → 実現曲線 Chamfer(realize_edge/realize_points は純 numpy)
- **誤差分解**(GT外周カバー / GT曲げ線カバー / 偽陽性 — 本セッションで確立した3成分)
- エッジ数比、バケット予測精度、レンダリング PNG(matplotlib)

→ **モデルの合否判定は全て Kaggle 上で完結する**。上記を1コマンドで回す
`evaluate_curve2.py`(v1 の evaluate_curve.py + 誤差分解の統合)を作ること。

### ローカルでしかできないもの

- STEP/B-Rep を触る全て: 新チャンクの変換(wireframe_app 抽出 = pythonocc)、
  mid STP との忠実度再計測(テッセレーション)
- CATIA 検証(将来): 生成ワイヤーフレーム → CATIA サーフェス再構築、実現器(PartMaker
  プランナ)による feasibility 判定 ※プランナ自体は純Pythonなので、PartMaker の
  synthetic_generator ソースを Dataset に含めれば Kaggle でも可 — 優先度低の任意項目

### 運用フロー

```
ローカル: チャンク変換(OCC) → parts JSON を zip → Kaggle Dataset 更新
Kaggle:  学習 + 全モデル評価 + レンダリング → Output(ckpt/history/metrics/png)
ローカル: Output ダウンロード → 保管(runs/)・必要時のみ STEP/CATIA 検証
```

## 4. タスク3: 無料枠枯渇時のローカル継続

- checkpoint 形式は共通(model/opt/args/epoch)なので、**Kaggle Output の last.pt を
  `runs/` に置き、ローカルの curve2.py に `--resume` で渡すだけ**で継続できる設計を維持する
- Opus が用意するもの: ①K→L / L→K の resume 往復テスト手順(1エポックずつ回して
  history が連続することを確認)、②双方の起動コマンドを並記したチートシート、
  ③GPU OOM 差異への注意(ローカル6GB vs Kaggle 16GB — Kaggle 専用に batch/accum を
  変える場合は args に記録されるので、ローカル継続時は明示的に上書きすること)
- 週30h の配分目安: 150エポック × 推定20-30s/epoch(T4) ≈ 1〜1.5h/run — 枠は十分。
  枯渇は主に試行錯誤の反復時に起きるので、**アブレーションは epochs 50 の短縮版で先に選別**する運用を推奨

## 5. ユーザーからの技術質問への回答(fable5 の分析)

### Q1: パラメータ不足の兆候はあるか? → **現時点では「ない」。逆の兆候(過学習)がある**

根拠: v1(10.5M params)は ep165 で val NLL が底(2.634)を打ち、以降 train 2.47 vs
val 2.65 と乖離拡大 = **2,050部品に対して容量は過剰側**。教師強制の各種精度
(文法83-100%、粗桁中央値誤差1-2bin)も容量律速を示さない。
監視すべき反証シグナル: データ増で val NLL の底が下がり続け、かつ教師強制精度が
頭打ちのまま → その時初めて width/depth 拡大(dim 384 等)を検討。Kaggle なら安価に試せるが、
**優先度は v2 の目的関数変更の検証より下**。

### Q2: 深さ起因の表現力不足はあるか? → **現時点では「ない」**

根拠: 失敗モードは「教師強制で解けるのに自由走行で複利する」であり、これは深さを
足しても直らない種類の問題(exposure bias)。系列~930トークンは全域 attention で
被覆済み(打ち切りなし)、ポインタ機構も機能(86%)。
実在するアーキテクチャ上の弱点は表現力ではなく: ①KVキャッシュ無しの O(n²) サンプリング
(評価が遅い — Kaggle 移行と併せて **KVキャッシュ実装は費用対効果の高い任意タスク**)、
②条件エンコーダが薄い(FIX+包絡のみ — 将来の周辺部品/NG条件の追加時に要拡張。今は不要)。

## 6. 既知の罠リスト(Opus は必読)

1. **part_id はチャンク間で重複**する — 一意キーはグループタグ付き(`p2c01__SYN_...`)。既対応だが新規コードでも同じ規約を守ること
2. `KMP_DUPLICATE_LIB_OK=TRUE` がローカル必須(anaconda の OpenMP 衝突)。Kaggle では不要のはず
3. checkpoint 保存は `safe_save`(tmp+replace+リトライ)を使う — ローカルで AV による保存失敗で 100ep 落とした実績あり
4. 学習は batch=1+勾配蓄積8(可変長系列のため)。バッチ化するなら SDPA の is_causal 前提(パディングマスク無し)を壊さないこと
5. サンプル評価(sample_eval2)は自由走行なので遅い — Kaggle でも `--sample-every 25` 程度に
6. prod02/chunk_05 以降が届いたらローカルで変換 → Dataset 更新(§3 フロー)

## 7. 完了条件

- [x] Kaggle 環境を完全模擬して curve2 が動く(OCC 不在・バンドルのみ・予算停止)
- [x] 同一 ckpt がローカルで `--resume` でき、history が連続する(往復テスト済: ep1-18連続)
- [x] evaluate_curve2(統合評価)が OCC 不在環境で動作しレンダリングまで出力
- [x] ユーザーマニュアル md(docs/kaggle_user_manual.md)
- [ ] 実機 Kaggle での 150ep 完走(ユーザー作業)

## 8. 実装結果(Opus 5)

### 8.1 OCC 依存の分離(§2.1 の必須項目)

`wtok/constants.py` を新設し `BITS/HI_BITS/E_ARITY/VTYPES/ETYPES/SparseW` と
`stable_seed`/`safe_save` を集約。9モジュールの import を差し替え、
**wtok の学習経路は wireflow パッケージにも依存しなくなった**(Kaggle バンドルが
wtok だけで自己完結)。`convert.py` は constants を re-export するのでローカル
パイプラインは非破壊(`build_synthetic --limit 2` で回帰確認済み)。

検証: 偽の `OCC` パッケージで実 pythonocc を遮蔽した状態で、学習経路10モジュール
全ての import・学習・サンプリング・評価が成功。

### 8.2 追加した機能

| 対象 | 内容 |
|---|---|
| `curve2.py` | `--max-hours`(予算超過で安全停止+保存)、`--resume-dir`(読み取り専用の前回 Output から自動再開、history 引き継ぎ) |
| `evaluate_curve2.py`(新規) | 教師強制(NLL・トークン種別精度・座標bin誤差)、自由走行(Chamfer・エッジ比)、**誤差3分解**(外形カバー/曲げカバー/偽陽性)、50%補完、レンダリング、`--baseline` で前回比較表。`--arch v1/v2` 両対応 |
| `tools/make_kaggle_bundle.py`(新規) | コード zip(14ファイル)とデータ zip(4.2MB)を生成 |
| `tools/cleanup_workspace.py`(新規) | ワークスペース削減(別紙 docs/workspace_cleanup_plan.md) |

### 8.3 Kaggle 完全模擬テストの結果

1. バンドル展開のみ+OCC 遮蔽で curve2 起動 → OK
2. `--max-hours` で ep8 で安全停止、`last.pt/best.pt/history.json` 保存 → OK
3. 別ディレクトリから `--resume-dir` で ep9〜16 継続、**history が 1..16 連続** → OK
4. `evaluate_curve2` が集計 JSON + renders/*.png を出力 → OK
5. **K→L 往復**: 模擬 Kaggle の ckpt をローカル実体コードで resume、ep17-18 継続、
   history 1..18 連続・NLL 平滑 → OK

### 8.4 データ規模(実測)

コード zip 数十KB / データ zip **4.2 MB**(2,300部品 JSON、非圧縮27MB)。
Kaggle Dataset の容量制限に対して余裕。チャンク追加時も再アップは数秒。

## 2FA の注意(2026-09-05 追記)
`!` プレフィックスでユーザーが実行した `vastai tfa login` は、Claude 側のシェルのセッションキーには反映されなかった
(別環境で実行されるため)。手順: Claude が `vastai tfa send-sms` を実行して Secret を提示 → ユーザーは**コードの数字だけ**
を返信 → Claude が自分のシェルで `tfa login` と `destroy` を続けて実行する。コードは 2〜3 分で失効する。
