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
