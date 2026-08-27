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
