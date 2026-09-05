# AutoMetalSheet - 板金設計自動化システム

## プロジェクト概要
板金設計の自動化システム。数学的・材料力学的に正確な板金CADデータの自動生成を目標とする。

## 技術調査状況

### 調査日: 2026-03-20

---

## 1. CATIA V5/V6 CATPart自動生成

### アクセス方法の階層

| レイヤー | 技術 | 難易度 | 板金アクセス |
|----------|------|--------|-------------|
| Automation API (COM) | VBA / Python (pycatia, pywin32) | 低 | 限定的 |
| EKL/Knowledge Expert | CATIA内蔵スクリプト言語 | 中 | 中程度 |
| CAA (C++ API) | ネイティブC++ | 高 | 完全 |

### pycatia (Python → CATIA COM)
- PyPI: `pip install pycatia` (最新: 0.9.5)
- GitHub: https://github.com/evereux/pycatia
- CATIA V5 (Windows必須) とのCOM連携
- Sheet Metal専用モジュールは**未確認**（ドキュメントに明記なし）
- 実績: パラメータ抽出、形状作成、スクリーンショット、STEP変換

### CATIA V5 Automation API の Sheet Metal 対応状況
- **重大な制限**: CATIA V5 Automation API（VBA/COM層）ではSheet Metalの
  Wall、Flange、Bend等のフィーチャーをプログラムから直接生成する
  公開APIは**事実上存在しない**（CAA C++層のみでアクセス可能）
- VBAで可能な操作: パラメータ変更、既存フィーチャーの修正、STEPインポート後の認識

### CAA (C++ Component Application Architecture)
- Dassault Systèmesの最上位API
- Sheet Metal Design モジュールのC++ APIが存在（V6 Aerospace Sheet Metal版あり）
- 開発には専用ライセンス、MSVC環境、膨大な学習コストが必要
- 公式ドキュメント: https://www.3ds.com/support/documentation/resource-library/installation-catia-v5-and-caa-api

### EKL (Enterprise Knowledge Language)
- CATIA V5/3DEXPERIENCEの組み込みスクリプト言語
- パラメータ駆動設計・ルール定義に使用
- Sheet Metal フィーチャーのパラメータ変更は可能
- 複雑なフィーチャー生成は困難

### CATDrawing生成との連携
- VBA/CATScript: DrawingViews コレクションから2Dビュー生成が可能
- DrawingViewGenerativeBehavior オブジェクトで3Dモデルから2Dビュー作成
- pycatia経由でも操作可能

---

## 2. CATProduct (アセンブリ) の自動生成

- VBA/Python で Products コレクションを操作可能
- 部品の追加・配置のプログラム的制御は可能
- pycatia ライブラリで `ProductStructure Object Library` を参照
- Tools → Generate CATPart from Product でCATProductをCATPart化可能

---

## 3. 板金CADツール比較

### Siemens NX (NXOpen API)
**最も充実した板金プログラミングAPI**
- 言語: Python, C#, C++, Java, VB
- 専用名前空間: `NXOpen.Features.SheetMetal`
- 利用可能クラス:
  - `ConvertToSheetmetalBuilder`
  - `TabBuilder`
  - `AeroJoggleBuilder`
  - その他多数の板金フィーチャービルダー
- ドキュメント: docs.plm.automation.siemens.com
- GUIマクロ記録 → コード生成が使える

### SolidWorks (SOLIDWORKS API)
**COM APIで板金フィーチャー生成が可能**
- 言語: VBA, VB.NET, C#
- 主要インターフェース:
  - `IBaseFlangeFeatureData` - ベースフランジ制御
  - `IBendsFeatureData` - ベンドデータ
  - `ISketchedBendFeatureData` - スケッチベンド
- メソッド: `InsertSheetMetal`, `InsertSheetMetalBaseFlange`
- K-Factor、Bend Allowance、Bend Deductionの設定可能
- 2024年版ドキュメントにサンプルコードあり
- GitHub: https://github.com/codestackdev/solidworks-api-examples

### Autodesk Inventor (API)
- `iLogic` (VB.NET) でSheet Metal自動化が最も一般的
- `ISheetMetalFeatures` オブジェクトモデル
- K-Factor設定: `oStyle.UnfoldMethod.kFactor`
- 曲げ半径設定: `oStyle.BendRadius`
- C# サポートあり (COM経由)
- 専用関数: WriteSheetMetalDXF, FlangeWidthsCreation, BendExtentEdges

### FreeCAD SheetMetal Workbench (OSS)
**完全無料・Pythonで直接制御可能**
- GitHub: https://github.com/shaise/FreeCAD_SheetMetal
- 最新バージョン: 0.7.58 (2025年10月)
- 利用可能コマンド (Python):
  ```python
  SheetMetal_AddBase      # ベース壁生成
  SheetMetal_AddWall      # 壁追加 (フランジ)
  SheetMetal_AddFoldWall  # 折り曲げ
  SheetMetal_AddBend      # ベンド追加
  SheetMetal_Unfold       # 展開図生成
  SheetMetal_AddCornerRelief  # コーナーリリーフ
  SheetMetal_Forming      # フォーミング
  ```
- Pythonからのプログラム的生成例:
  ```python
  import FreeCAD
  obj = FreeCAD.ActiveDocument.addObject("Part::FeaturePython", "BaseBend")
  SMBaseBend(obj)
  obj.radius = 2.5
  obj.thickness = 1.5
  ```
- `smBase(thk, length, radius, Side, ...)` 関数で直接生成
- K-factor: Spreadsheetオブジェクトで管理、材料テーブルから自動参照
- 展開計算: `SheetMetalUnfolder.py` (K-factor考慮)
- 展開計算ツール: `tools/calc-unfold.py`

---

## 4. OSS/商用ツール (CATIA不要)

### OpenCASCADE Technology (OCCT)
- C++ ライブラリ (Python: pythonocc-core 7.9.3 - 2026年2月)
- STEP/IGES読み書き
- B-Rep板金モデリング可能 (低レベルAPI)
- 板金特化機能は**商用コンポーネント**
  - Sheet Metal Operations Component (有償SDK)
  - 認識: フランジ、ベンド、ジョグ、穴、カットアウト
  - 展開: K-factor使用のフラットパターン生成
  - 要: OCCT 7.6.1以上
  - ライセンス: 年間/永続ライセンス

### Analysis Situs (OSS + 商用拡張)
- サイト: https://analysissitus.org/
- OpenCASCADE基盤のCAD調査・プロトタイピングSDK
- **板金認識・展開はSMRU商用拡張**
  - 板金認識 (フィーチャー自動検出)
  - フラットパターン生成
  - 曲げシーケンスシミュレーション
- コアはOSS (STEP/IGES/BREPのI/O、形状修復)
- AAG (Abstract Adjacency Graph) アルゴリズム実装

### build123d (Python)
- PyPI: `pip install build123d`
- OpenCASCADE基盤のPythonフレームワーク
- 板金操作は**ロードマップ段階** (bend/hem/flange/tab/relief)
- フラットパターン生成は計画中

### pythonocc-core (Python)
- `pip install pythonocc-core` (conda-forge推奨)
- 汎用B-Repモデリング (板金専用機能なし)
- STEP/IGES/STL/OBJエクスポート可能

---

## 5. 板金工学計算 - 数学的基礎

### ベンドアローワンス (BA)
```
BA = (π/180) × Angle × (R + K × T)
```
- Angle: 曲げ角度 (度)
- R: 内側曲げ半径
- K: K-factor (通常 0.3〜0.5)
- T: 板厚

### K-factor
- 中立軸の位置 = 内面から中立軸までの距離 / 板厚
- **重要: 「角度依存」ではなく「R/t比依存」で実装すること**
  - R/t < 1.0  : K ≈ 0.33
  - R/t = 2.0  : K ≈ 0.38
  - R/t = 3.0  : K ≈ 0.41
  - R/t = 5.0  : K ≈ 0.44
  - R/t ≥ 8.0  : K → 0.50（薄肉近似）
- SPFH590/DP780は同R/tでもK値が軟鋼より小さい（内側に寄る）
- 材料ごとのK-factor Lookup Tableで実装する（一定値では不正確）

### スプリングバック計算
```
Rf / Ri ≈ 1 + (3σy·Ri)/(E·t) − 4(σy·Ri/(E·t))³
```
または近似式:
```
Δθ = K_sb × (θ_initial - 90°)
```
- σy: 降伏強度
- E: ヤング率
- 高強度鋼ほどスプリングバック大
- R/t > 8 で急激にスプリングバック増大
- **精度限界（重要）**:
  - SPCC/SPFH590 まで: 誤差 ±1-2度（実用可）
  - DP780以上: 等方性硬化モデルの誤差 ±3-6度（Bauschinger効果を非考慮）
  - DP780以上は計算結果に「±3度の不確実性帯」を付加して設計者に提示すること
  - Yoshida-Uemori混合硬化モデルは未実装 → Phase 3以降の課題

### 展開計算アルゴリズム
1. B-Repモデルを面隣接グラフ(AAG/FAG)で解析
2. 平面面 (フランジ) と円筒面 (ベンド) を分類
3. K-factorから中立軸半径を計算
4. 円筒ベンドを「変形なし」で展開 (OpenCASCADEの近似補間使用)
5. フランジ面を参照平面上に回転・配置

---

## 6. STEP/IGES経由でのCATPart生成

### ワークフロー
```
B-Rep生成 (OCC/FreeCAD等) → STEP出力 → CATIA Recognize Tool
```
- CATIA: 「Recognize」ツールでSTEP/IGESの板金ソリッドを板金フィーチャーに変換
- ただし完全な変換は保証されない (形状の複雑さに依存)
- STEP AP242: PMI (製品・製造情報) も含めて転送可能
- 自動化: CATIA COMで `FileOpen` + `StartCommand("RecognizeSheetMetal")` 相当

### CATIA → 他ツール
- STEPエクスポート: ST1ライセンス必要
- AP203/AP214/AP242 対応
- DXF/DWG: 展開図エクスポートに使用

---

## 推奨アーキテクチャ案

### 案A: FreeCAD + Python (フルOSS)
- 完全無料・ライセンス問題なし
- Python APIで板金フィーチャー生成
- STEP/DXF出力でCATIAへ連携
- 課題: CATIA特有フィーチャーの再現が困難

### 案B: SolidWorks API (COM + C#/Python)
- 成熟したAPIで板金フィーチャー生成実績あり
- K-Factor/BendAllowance設定可能
- ライセンス: SolidWorks本体が必要

### 案C: Siemens NX Open (Python/C#)
- 最も完全な板金APIを保有
- NXライセンスが必要
- マクロ記録機能でコード生成補助

### 案D: CATIA CAA C++ (ネイティブ)
- 最高のCATIA互換性
- 開発コスト最大
- 商用案件向け

### 案E: OpenCASCADE + pythonocc (B-Repレベル)
- 中立的なカーネルで形状生成
- 板金専用APIは自作または商用SDK購入
- CATIA/NX/SolidWorksいずれにも連携可能

---

---

## 7. 決定論的エンジン 設計判断メモ（2026-03-20 分析）

### K-factor実装方針
- `一定値` ではなく `R/t比依存ルックアップテーブル` で実装する（材料ごと）
- データソース: JIS G 3141（SPCC）、JIS G 3134（SPFH）、JFEスチール技術資料

### DFMルール管理
- ルール自体をデータ（YAML）として管理。コードにハードコードしない
- 優先順位: OEM固有要件 > 社内標準 > JIS/公的規格 > 材料メーカー推奨 > 業界標準
- ルール変更はGit PR + 設計責任者レビューで管理。CHANGELOG.md必須
- ルールIDを付与（例: DFM-BEND-001）して追跡可能にする

### 包絡面計算
- 静的干渉: pythonocc `BRepAlgoAPI_Section` + Shapely `unary_union`
- 動的スイープ包絡面: 高コスト（各姿勢のB-Rep Boolean Union）→ 事前計算キャッシュで対応
- `steps=18`（5度刻み）でBooleanUnion x18回 → 数十秒〜数分（複雑形状）

### スプリングバック
- SPCC/SPFH590: Wagonerモデルで実用精度（誤差±1-2度）
- DP780以上: 誤差±3-6度。計算結果に `uncertainty_deg` と `warning` を付加して設計者提示

---

## 8. 慣例形状学習システム 設計判断メモ（2026-03-20 分析）

### CATPartからの慣例抽出
- pycatia でフィーチャーツリーからパラメータ統計を自動抽出
- 抽出すべき統計: フランジ高さ/板厚 比率の分布（mean/stdev/P10/P90）
- フィーチャー出現頻度（RoundRelief vs SquareRelief 等）
- 頻出フィーチャーシーケンス（PrefixSpanアルゴリズム、min_support=0.3）

### ベクトルDB
- 推奨: **pgvector（PostgreSQL拡張）**
  - 理由: 数値パラメータのフィルタリング + ベクトル検索のハイブリッドが得意
  - 社内サーバー管理に向く（Qdrant/Pineconeより低コスト）
- 埋め込み: テキスト部分（テキスト埋め込みモデル）+ 数値特徴量（正規化して重みづけ連結）

### ファインチューニングの判断基準
- RAGで対応可能: 慣例の参照・統計値の提供・類似設計の検索
- FTが必要な場合:
  - 自然言語 → CadQueryコード生成スタイルの学習
  - 社内設計用語（"メクラ穴", "ヌスミ加工"）の解釈
  - CATPartフィーチャーツリー構造の推論
- 最低データ量: 200件（限定的）/ 実用: 500件 / 理想: 2000件以上
- 訓練データ生成: CATPartパラメータ抽出 → CadQueryコード自動生成 → 要件の逆生成（LLM）

### ハード制約とソフト制約の衝突解決
- 衝突タイプ:
  1. 慣例 < DFM最小値 → DFMハード制約が必ず勝つ
  2. 慣例 ≠ OEM要件 → OEM要件（契約要件）が勝つ（DFM満足を確認してから）
  3. 部門間の慣例衝突 → ユーザーへ確認（自動解決しない）
- 衝突時は説明文を必ず生成: 「社内慣例では●●だが、DFM制約により○○に変更」
- 慣例DBはコンテキスト付き管理（部門 × 製品ライン × OEM）

### 慣例DBの更新ガイドライン
- Tier 1（自動）: 新規CATPartが追加されるたびに統計自動更新
- Tier 2（主任レビュー）: 標準値変更 → Git PR + 主任設計者承認
- Tier 3（製造部門承認必須）: DFMルール変更・仕入れ先制約変更

---

## 9. 実装フェーズ優先順位（Round 3 Final 確定版）

### Round 2で確定した12項目のアーキテクチャ修正
1. MVP: CP1-Lite + CP2 の閉ループ（半構造化フォーム + 矛盾チェック含む）
2. 精度指標: 「完成品一致率」→「CP込み工数削減率30%+」「DFM違反率<5%」
3. LLM: 3層ハイブリッド（Qwen-2.5-72B INT8/AWS Bedrock/Public Cloud）
4. EnvelopeChecker: マージン0.2mm→材料依存テーブル（SPCC:0.5mm/SPFH590:1.0mm/DP780:2.0mm）
5. INFEASIBLE早期検出の追加
6. Structured Generation（Outlines）の採用
7. テンプレート6種→10種（hat_asymmetric/omega/multi_bend/Z_with_lip追加）
8. K-factor: n値/r値補正係数追加 + 量産/試作モード分離
9. スプリングバック境界: 強度値→組織タイプ分類（フェライト/マルテンサイト/DP）
10. OEM YAML: 階層差分構造 + 信頼度スコア + バージョン管理
11. Phase 1でDP780のDFMチェックのみ先行実装
12. データ戦略: CATPartスケッチ逆変換廃止→合成+パラメータ抽出

### フェーズ優先実装順

| Phase | コンポーネント | 難易度 | ROI | 週単位目安 |
|-------|--------------|--------|-----|----------|
| **Phase 1** | 材料DB（SPCC/SPFH590 + K-factor n値/r値補正）| 低 | 高 | W1 |
| **Phase 1** | DFMルールエンジン（YAML管理 + DP780 DFMのみ先行）| 中 | 最高 | W2 |
| **Phase 1** | INFEASIBLE早期検出 | 低 | 高 | W3 |
| **Phase 1** | 包絡面チェック（材料依存マージンテーブル）| 中 | 高 | W4 |
| **Phase 1** | 3層LLMハイブリッド基盤 + Outlines | 中 | 高 | W5〜W6 |
| **Phase 1** | テンプレート10種 + OEM YAML階層差分 | 中 | 高 | W7〜W8 |
| **Phase 1** | FreeCADヘッドレスパイプライン | 中 | 高 | W9〜W10 |
| **Phase 1** | 半構造化フォームUI + CP1-Lite + CP2 | 中 | 最高 | W11〜W12 |
| **Phase 1** | 統合・ベンチマーク・パイロット | — | — | W13〜W24 |
| **Phase 2** | Step 2a 幾何学サポートUI（ヒートマップ）| 高 | 高 | M7〜M9 |
| **Phase 2** | pgvector + 慣例抽出パイプライン（合成データ）| 中 | 高 | M10〜M12 |
| **Phase 2** | LoRA FT（500件トリガー）| 高 | 中 | M13〜M15 |
| **Phase 2** | OEM別DFMルール展開（1社ずつ）| 中 | 高 | M16〜M18 |
| **Phase 3** | CAA C++連携 または SWx APIピボット | 非常に高 | 中 | — |
| **Phase 3** | 包絡面計算（動的スイープ）| 非常に高 | 中 | — |
| **Phase 3** | DP780以上 Yoshida-Uemori精密SB | 非常に高 | 低 | — |

### Phase 1 Go/No-Go 判定基準（6ヶ月時点）

| 指標 | Go | Conditional | No-Go |
|------|-----|-------------|-------|
| CP込み工数削減率 | ≥30% | 20〜30% | <20% |
| DFM違反率（出力物）| <5% | 5〜10% | ≥10% |
| 断面生成収束率（N=5）| ≥70% | 55〜70% | <55% |
| BA/BD精度 | ±0.05mm | ±0.1mm | 超過 |
| INFEASIBLE誤検出率 | <3% | 3〜8% | ≥8% |
| 社内ユーザー継続利用意向 | ≥70% | 50〜70% | <50% |

### ROI サマリー

| 項目 | 値 |
|------|-----|
| Phase 1 開発費 | ¥37,600,000 |
| Phase 1+2 合計 | ¥91,400,000 |
| 顧客1社の年間価値 | ¥11,500,000 |
| 外販推奨価格 | ¥3,000,000/社/年 |
| 損益分岐点 | Phase 2 Month 10頃（基本シナリオ: 2社導入）|

### 意図的技術的負債（Phase 1残存・返済計画付き）

| 負債ID | 内容 | 返済Phase | 事業リスク化閾値 |
|--------|------|---------|----------------|
| TD-01 | CATIAフィーチャーツリー喪失 | Phase 3 | Recognize不可案件3件以上 |
| TD-03 | Step 2a完全手動 | Phase 3 | 断面指定工数が全体50%超 |
| TD-04 | DP780 等方性モデル限定 | 将来 | DP780ミス3件以上（即時停止） |
| TD-06 | SQLite（pgvectorなし）| Phase 2前半 | 検索3秒超 or 件数200件超 |
| TD-07 | 動的スイープ包絡面未実装 | Phase 3 | 動的干渉リコール発生（即時停止）|
| TD-08 | CATIA V5専用（V6非対応）| Phase 2〜3 | V6顧客案件2件以上発生（即時対応検討）|
| TD-09 | 設変連鎖変更未対応（PARAM_CHANGE単独のみ）| Phase 2 | 連鎖変更エラーが月3件以上（前倒し対応）|

---

## 10. 設変・流用・探索的設計対応 確定設計判断（Round 4）

### 記録日: 2026-03-23

---

### 確定設計判断テーブル

| 判断ID | カテゴリ | 判断内容 | 根拠 | Phase | 影響コンポーネント | 取り消し条件 |
|--------|---------|---------|------|-------|-----------------|------------|
| R4-01 | アーキテクチャ | CESPアーキテクチャを採択し、モードセレクター方式を棄却する。設変・流用・探索的設計は別モードではなく、単一パイプライン上のコンテキスト差異として扱う。 | モードごとにパイプラインを分岐させると保守コストが指数的に増加し、DFMエンジンの共通化が阻害されるため。 | Phase 1 | RequestRouter, PipelineOrchestrator | 3種以上のコンテキストで同一パイプライン上の処理分岐が20箇所超になった場合 |
| R4-02 | アーキテクチャ | 設変・流用・探索的の各コンテキストは「モード」ではなく「コンテキストフィールド」（context_type / source_part_id / delta_spec 等）として入力構造体に付与する。パイプライン内部の制御フローはStrategyパターンで分岐する。 | モード切り替えUIはユーザーの思考を固定化し、実務でのコンテキスト混在（設変+流用同時）に対応できないため。 | Phase 1 | InputSpec, StrategySelector | 「コンテキストフィールドの組み合わせ数」が実装可能なStrategyの上限（目安:12）を超えた場合 |
| R4-03 | アーキテクチャ | Phase 1ではコンテキストフィールドをUI上に非表示とし、内部処理のみで使用する。Go/No-Go指標の純度を確保するため、評価期間中は新規フィーチャーの影響をベースラインに混入させない。 | Phase 1のGo/No-Go判定（工数削減率30%等）はCP1-Lite+CP2の閉ループ単体性能を測定するものであり、コンテキスト対応の影響を混在させると判定基準が曖昧になるため。 | Phase 1 → Phase 2でUI公開 | FormUI, Go/No-Go評価基盤 | Go/No-Go評価が完了し、Phase 2移行が承認された時点で取り消し（UI公開） |
| R4-04 | アーキテクチャ | DFMエンジン特化を唯一の防衛可能な競合優位と位置づけ、開発リソースを優先集中する。CESPの他コンポーネントはDFMエンジンを補完する役割に限定する。 | CADDi Drawer等の既存ツールが検索・類似照合で優位を持つ中、DFM違反ゼロ保証という品質軸での差別化のみが参入障壁を形成できるため。 | 全Phase共通方針 | DFMRuleEngine, YAML管理体制 | 競合製品がDFM違反ゼロ保証を同等水準で実装し、価格競争に移行した場合 |
| R4-05 | PartIngestionPipeline | 設変・流用・探索的の3コンテキストに共通する「既存部品読み込みパイプライン（PartIngestionPipeline）」をPhase 1で先行実装する。これがなければ設変・流用コンテキストは動作不能であるため、必須先行実装とする。 | 3コンテキストすべてが既存部品の読み込みを前提とするため、共通基盤の先行実装なしには後続開発が並列化できないため。 | Phase 1 (W3〜W4) | PartIngestionPipeline, CATPartReader, STEPReader | 設変・流用コンテキストをPhase 2以降に全面延期する経営判断が下された場合 |
| R4-06 | PartIngestionPipeline | CATPartの読み込みにはpycatiaを使用し、設計者が明示的に入力したパラメータのみを抽出する（確信度HIGH）。フィーチャーツリーから暗黙的に逆算されるパラメータは使用しない。 | pycatia COM層で取得できる値はユーザー入力パラメータに限られ、ソリッド形状の幾何逆算は不確実性が高くDFM判定のインプットとして不適切なため。 | Phase 1 | CATPartReader, ParameterWithConfidence | CATIA CAA C++連携（Phase 3）が実装された時点で上限を拡大する |
| R4-07 | PartIngestionPipeline | STEPファイルの読み込みにはpythonocc BRep解析を使用し、パラメータを幾何形状から逆算する（確信度MEDIUM、tolerance ±0.1mm）。CATPartより確信度が低いことをシステムが明示する。 | STEPはフィーチャーツリー情報を持たないB-Rep形式であり、パラメータの幾何逆算は原理的に誤差を伴うため、確信度を下げた扱いが誠実かつ安全なため。 | Phase 1 | STEPReader, ParameterWithConfidence | pythonocc以外の高精度逆算ライブラリ（例: Analysis Situs SMRU）が評価により採用された場合 |
| R4-08 | PartIngestionPipeline | Phase 1のPartIngestionPipelineが抽出するパラメータは「フランジ高さ・曲げR・板厚」の最小限3パラメータに絞る。穴位置・コーナーリリーフ形状等の追加パラメータはPhase 2以降に拡張する。 | MVP段階で抽出対象を広げすぎると、確信度管理の複雑度が急増し、Phase 1のスケジュールリスクが高まるため。 | Phase 1 (最小限) → Phase 2 (拡張) | CATPartReader, STEPReader | パイロット顧客から最低3パラメータ以外の要求が明示的に挙がり、かつGo/No-Go判定に影響することが確認された場合 |
| R4-09 | ParameterWithConfidence型 | システム内を流通する全パラメータに `ParameterWithConfidence` 型を適用し、value / confidence / source / tolerance / requires_cp_confirmation の5フィールドを必須とする。naked floatでのパラメータ受け渡しを禁止する。 | 信頼度の異なるパラメータが同一型で流通すると、DFMエンジンが誤ったハード判定を下すリスクがあり、型レベルで信頼度を強制する設計が安全性の根拠となるため。 | Phase 1 | 全コンポーネント（型定義として横断的に影響） | 型の複雑度がLLMの構造化生成（Outlines）と相性が悪く、生成精度の低下が実測された場合 |
| R4-10 | ParameterWithConfidence型 | DFMエンジンは confidence < 0.75 のパラメータに対してエラーを発生させず、WARNING + tolerance分の保守値シフトを行う。保守値シフトの方向は「DFM違反リスクが減少する方向」とする。 | 確信度が低いパラメータでエラー停止するとパイプラインが使用不能になる頻度が高くなり、実務運用の支障になるため。WARNINGで継続しつつ安全側に倒す設計が実用と安全のバランスをとるため。 | Phase 1 | DFMRuleEngine | 保守値シフトにより設計者の意図から乖離した出力が頻発し（月3件以上の苦情）、シフト無効化の要望が出た場合 |
| R4-11 | ParameterWithConfidence型 | sourceフィールドごとにデフォルトtoleranceを設定する。初期値: `catpart_explicit: 0.0mm`（設計者明示）、`step_extracted: 0.15mm`（幾何逆算）、`llm_inferred: 0.30mm`（LLM推定）。数値はパイロット後に実測値で補正する。 | sourceが異なれば誤差の性質が根本的に異なるため、一律toleranceは過保守または過楽観になる。source別デフォルト設定が最小限の複雑度で最大限の精度をもたらすため。 | Phase 1 (初期値) → パイロット後補正 | ParameterWithConfidence, PartIngestionPipeline | パイロットデータから実測toleranceが初期値と2倍以上乖離した場合（即時補正） |
| R4-12 | 設変コンテキスト | Phase 1の設変MVPはパラメータ変更型（PARAM_CHANGE）のみに限定する。フィーチャー追加・削除・トポロジー変更を伴う設変はPhase 2以降とする。 | フィーチャートポロジー変更はCAA C++なしにはCATPartへの書き戻しが困難であり（TD-01）、Phase 1の技術スタック（pycatia/FreeCAD）で対応できる範囲をPARAM_CHANGEに限定することでリスクを制御するため。 | Phase 1 (PARAM_CHANGEのみ) → Phase 3 (トポロジー変更) | ECNDeltaSpec, PartIngestionPipeline | PARAM_CHANGE限定ではパイロット顧客の主要ユースケースをカバーできないことが判明した場合 |
| R4-13 | 設変コンテキスト | フィーチャーをClass A（安全関連）/ Class B（取付関連）/ Class C（外観・その他）に分類し、クラスごとに確信度閾値・CP介入閾値を差別化する。Class A: confidence必須 ≥ 0.90 / Class B: ≥ 0.75 / Class C: ≥ 0.60。 | 安全関連フィーチャーの誤処理は製品リコールに直結するため、クラス別に閾値を設けることが品質管理の原則に合致するため。 | Phase 1 | FeatureClassifier, DFMRuleEngine, CPManager | Class A閾値0.90でのFalse Negative（見逃し）が実測で0件超になった場合（即時引き上げ） |
| R4-14 | 設変コンテキスト | Class Aフィーチャーへの誤処理件数ゼロを絶対条件（Non-Negotiable Constraint）とする。この条件を満たせない場合はシステムを停止する。Go/No-Go指標とは独立した強制停止条件として管理する。 | 安全関連部品の誤生成は製品リコール・人身事故のリスクを持ち、事業継続の根幹を損なうため、他の指標とのトレードオフを一切認めない。 | 全Phase | FeatureClassifier, AlertManager, Go/No-Go評価基盤 | 取り消し不可（絶対制約） |
| R4-15 | 設変コンテキスト | フィーチャー同定はCP（Confirmation Point）機構での人間確認を標準フローとし、例外ではなく設計として位置づける。自動同定結果を設計者が確認・修正してから後続処理が進む。 | Class A/Bフィーチャーの誤同定リスクを、システムの自動化率よりも人間確認の保険コストで吸収する設計判断。Phase 1の技術成熟度では自動同定の信頼性が十分でないため。 | Phase 1 | CPManager, FeatureClassifier | 自動同定の精度が6ヶ月間の実測でClass A ≥ 99.5%、Class B ≥ 98%を達成した場合（CP省略検討可） |
| R4-16 | 設変コンテキスト | 新規技術的負債として TD-08（CATIA V5専用、V6非対応）と TD-09（設変連鎖変更未対応）を登録する。TD-08はV6顧客案件が2件以上発生した時点で即時対応を検討し、TD-09はPARAM_CHANGE複数箇所の相互依存エラーが月3件以上発生した時点で対応を前倒しする。 | 既知の限界を明示的に技術的負債として登録することで、見えないリスクを管理可能なリスクに変換するため。 | Phase 1登録 → 閾値到達時に対応 | TD管理台帳, CATPartReader | — |
| R4-17 | 流用設計コンテキスト | Phase 2 MVPのベクトル検索は形状128次元 + テキスト768次元の2ベクトルに限定する。機能インターフェースベクトル（64次元）の追加は手動タグ付きデータが100件蓄積した後とする。 | データが不足した状態で高次元ベクトルを追加すると、検索精度が向上せずインデックス維持コストのみが増加するため。100件の閾値は埋め込み空間の最低限の密度を確保するための経験則。 | Phase 2 (2ベクトル) → 100件後に3ベクトル化 | pgvector, PartEmbeddingPipeline | 手動タグ付けコストが想定を超え、100件達成に6ヶ月以上かかることが確定した場合（追加延期） |
| R4-18 | 流用設計コンテキスト | 流用部品のQuality Tier 1〜2は自動判定（DFM違反率・設計パラメータ完全性スコアによる）、Tier 3以上は設計者による手動フラグ付けを必須とする。 | Tier 3以上は「OEM承認済み実績品」等の定性的条件を含み、自動判定の誤分類リスクが高い。手動フラグを必須にすることで誤流用のリスクを抑制するため。 | Phase 2 | QualityTierClassifier, pgvector | 手動フラグ付け工数がパイロット評価でユーザー継続利用意向を下げている（<70%）と判明した場合 |
| R4-19 | 流用設計コンテキスト | inheritance_matrixはDFMルールYAMLに統合し、二重管理を禁止する。流用可否判定のルールは常にDFMルールYAMLが単一ソース・オブ・トゥルースとなる。 | 継承マトリクスを独立ファイルで管理すると、DFMルール変更時の同期漏れによる流用可否の矛盾が発生し、設計事故の原因になるため。 | Phase 2 | DFMRuleEngine, OEM YAML | DFMルールYAMLの肥大化（行数5000超）によりパース速度が問題化した場合（分割を検討） |
| R4-20 | 流用設計コンテキスト | source_type='competitor'（競合他社部品）の流用を禁止する（RR-08対応）。競合部品がDBに混入した場合のアラートと自動除外ロジックを実装する。 | 競合他社のCADデータを自社製品に流用することは知的財産権侵害のリスクを持ち、事業継続の法的リスクとなるため。 | Phase 2 | PartIngestionPipeline, pgvector, QualityTierClassifier | 取り消し不可（法的リスク管理） |
| R4-21 | 流用設計コンテキスト | CADDi Drawerとは競合ではなく補完関係として定義し、将来的な連携APIの設計上の余地を残す。現時点では競合分析ではなく差別化軸の明確化に注力する。 | CADDi Drawerは検索・類似照合に特化し、DFMチェック・自動生成機能を持たない。機能的な棲み分けが明確であり、将来の顧客データ連携の可能性を閉じないため。 | 方針（全Phase） | プロダクト戦略, API設計方針 | CADDi DrawerがDFM自動チェック機能を本格展開した場合（競合再評価） |
| R4-22 | 探索的設計コンテキスト | EDM（Exploratory Design Module）のMVPを「Socratic対話エンジン」ではなく「未確定耐性パイプライン（Uncertainty-Tolerant Pipeline）」として再定義する。不完全な仕様でもDFMチェックを実行できることを最優先機能とする。 | Socratic対話エンジンはLLMの対話品質に依存し、Phase 1の技術成熟度では品質保証が困難。未確定耐性パイプラインは確定パイプラインの延長実装であり、開発コストが低くMVP価値が高いため。 | Phase 1（未確定耐性）→ Phase 2（Socratic対話） | EDM, CP1-Lite, INFEASIBLEChecker | 未確定耐性パイプラインのみでは探索的設計ユースケースの主要ニーズをカバーできないことがパイロットで判明した場合 |
| R4-23 | 探索的設計コンテキスト | CP1-LiteのすべてのInputフィールドをNullable化し、未入力フィールドに対応するERCコード（ERC-001〜006）を定義する。ERCコードはDFMチェック時のスキップ条件・保守値適用条件を制御する。 | 探索的設計では仕様が確定していないフィールドが存在することが前提であり、必須フィールドエラーでパイプラインが停止することは探索的設計ユースケースを根本的に破壊するため。 | Phase 1 | CP1-Lite InputSpec, DFMRuleEngine, ERCCodeManager | ERCコードの分岐数が増えすぎ（12超）、DFMルール条件との組み合わせ管理が破綻した場合 |
| R4-24 | 探索的設計コンテキスト | INFEASIBLE検出を `FEASIBLE_IN_RANGE`（一部パラメータ条件下で実現可能）/ `INFEASIBLE_IN_ALL_CASES`（全条件下で不可能）の2種に拡張する。従来の単一INFEASIBLE出力を廃止する。 | 単一のINFEASIBLE出力は探索的設計において「どうすれば実現可能か」の情報を設計者に提供できない。2種への拡張により、設計者に修正方向性を示す可能性が生まれるため。 | Phase 1 | INFEASIBLEChecker, DFMRuleEngine | FEASIBLE_IN_RANGEの計算コストが1リクエストあたり2秒超となり、レスポンス性能に問題が出た場合 |
| R4-25 | 探索的設計コンテキスト | Design Space Exploration（DSE）はPhase 2以降に延期し、Phase 1ではFreeCADフル生成なし・パラメータ推定のみの軽量実装にとどめる。 | Phase 1でFreeCADフル生成を探索的設計ループに組み込むと、1探索ステップあたりの計算コストが過大になりUXが破綻するため。パラメータ空間での推定のみで探索価値を提供できることを先に検証する。 | Phase 1 (推定のみ) → Phase 2 (DSE) | EDM, FreeCADPipeline | パラメータ推定のみではGo/No-Go指標の断面生成収束率が55%未満に留まることが判明した場合（即時対応） |
| R4-26 | 探索的設計コンテキスト | Socratic対話エンジンはPhase 2以降に実装し、Phase 1ではLLMは質問テンプレートの言語化のみを担当する。対話フローの制御ロジックはルールベースで実装する。 | LLM主導の対話フロー制御はプロンプト変化による挙動の不安定性が高く、Phase 1のGo/No-Go評価期間中に変数として加えることがリスクになるため。 | Phase 1 (テンプレート言語化のみ) → Phase 2 (Socratic) | EDM, LLMHybridLayer, CPManager | — |
| R4-27 | CPマネジメント | CPの挿入を確信度スコアに基づく動的判定とする。全パラメータの confidence ≥ 0.75 → CP省略 / 1〜3件が confidence < 0.75 → 軽量CP（フォーム確認のみ）/ 4件以上が confidence < 0.75 → 通常CP（説明付き確認）。閾値はパイロット後に実測値で補正する。 | CP数を固定すると、高品質インプット時の設計者負担増（不要な確認）と低品質インプット時のリスク見逃しが同時に発生するため。確信度スコアベースの動的挿入が両問題を解決するため。 | Phase 1 | CPManager, ParameterWithConfidence, PipelineOrchestrator | 動的CP省略によるDFM違反率が5%超になった場合（閾値引き上げ・動的省略無効化） |
| R4-28 | CPマネジメント | CPの数を設計で固定しない。「設変コンテキストにCP1.5を追加する」等の固定増設案を棄却し、コンテキスト種別によるCP数の固定化を禁止する。CP数はR4-27の動的ルールのみで決定する。 | コンテキスト別にCP数を固定すると、コンテキスト判定ロジックとCP挿入ロジックが結合し、将来のコンテキスト追加時に変更箇所が多重化するため。 | Phase 1 | CPManager | — |
| R4-29 | 共通技術 | ECNDelta（設変差分）とReuseDiff（流用差分）を統一した `DeltaSpec` 型として設計する。コンテキスト種別によらず差分情報を単一の型で表現し、DFMエンジンへの入力インターフェースを統一する。 | ECNDeltaとReuseDiffを別型で管理すると、DFMエンジンが両型を受け入れる分岐ロジックを持つ必要があり、DFMエンジンの内部複雑度が不必要に高まるため。 | Phase 1 | DeltaSpec, DFMRuleEngine, ECNProcessor, ReuseProcessor | DeltaSpecの統一によりECN固有情報（承認番号・ECN日付等）の格納が困難になった場合（サブタイプ分割を検討） |
| R4-30 | 共通技術 | source_type による処理分岐はStrategyパターンで実装し、if/elif チェーンを禁止する。各source_typeに対応するStrategyクラスを独立させ、新規source_type追加時はStrategyの追加のみで対応できる設計とする。 | if/elif チェーンはsource_type追加時に既存コードの変更が必要になりOCPに違反する。StrategyパターンはOCP（開放閉鎖原則）を満たし、テスト容易性も高いため。 | Phase 1 | StrategySelector, PartIngestionPipeline, PipelineOrchestrator | — |

---

### 判断間の依存関係グラフ

```mermaid
graph TD
    R4_01["R4-01: CESPアーキテクチャ採択"] --> R4_02["R4-02: コンテキストフィールド設計"]
    R4_01 --> R4_04["R4-04: DFMエンジン集中投資"]
    R4_02 --> R4_03["R4-03: Phase 1でUI非表示"]
    R4_02 --> R4_30["R4-30: Strategyパターン強制"]
    R4_02 --> R4_29["R4-29: DeltaSpec統一型"]

    R4_05["R4-05: PartIngestionPipeline先行"] --> R4_06["R4-06: CATPartはpycatia (HIGH)"]
    R4_05 --> R4_07["R4-07: STEPはpythonocc (MEDIUM)"]
    R4_05 --> R4_08["R4-08: Phase 1は最小3パラメータ"]

    R4_06 --> R4_09["R4-09: ParameterWithConfidence型必須"]
    R4_07 --> R4_09
    R4_08 --> R4_09

    R4_09 --> R4_10["R4-10: DFMはWARNING+保守シフト"]
    R4_09 --> R4_11["R4-11: source別デフォルトtolerance"]
    R4_09 --> R4_27["R4-27: 動的CP挿入（確信度ベース）"]

    R4_27 --> R4_28["R4-28: CP数固定禁止"]
    R4_27 --> R4_15["R4-15: フィーチャー同定はCP標準"]

    R4_10 --> R4_13["R4-13: フィーチャークラスA/B/C差別化"]
    R4_13 --> R4_14["R4-14: Class A誤処理ゼロ絶対条件"]
    R4_13 --> R4_15

    R4_12["R4-12: 設変MVPはPARAM_CHANGEのみ"] --> R4_16["R4-16: TD-08/TD-09登録"]
    R4_12 --> R4_29

    R4_04 --> R4_19["R4-19: inheritance_matrixをDFMYAMLに統合"]
    R4_19 --> R4_20["R4-20: competitor部品流用禁止"]

    R4_17["R4-17: Phase 2は2ベクトル限定"] --> R4_18["R4-18: Quality Tier自動/手動判定"]
    R4_18 --> R4_20
    R4_19 --> R4_18

    R4_22["R4-22: EDM=未確定耐性パイプライン"] --> R4_23["R4-23: CP1-Lite Nullable化+ERCコード"]
    R4_22 --> R4_24["R4-24: INFEASIBLE 2種拡張"]
    R4_22 --> R4_25["R4-25: DSEはPhase 2延期"]
    R4_22 --> R4_26["R4-26: Socratic対話はPhase 2"]

    R4_23 --> R4_24
    R4_25 --> R4_26

    R4_29 --> R4_30
    R4_04 --> R4_21["R4-21: CADDi Drawer補完関係定義"]
```

---

### Round 4議論で未解決のまま残っている論点（Round 5以降の検討事項）

| 論点ID | 論点内容 | 優先度 | 推奨検討タイミング |
|--------|---------|--------|-----------------|
| OI-01 | `llm_inferred` ソースのデフォルトtolerance（R4-11でTBDのまま）。LLM出力の誤差特性を実測するためにはパイロットデータが必要。 | 高 | Phase 1パイロット後（Month 4〜6） |
| OI-02 | FEASIBLE_IN_RANGEの計算アルゴリズム具体化（R4-24）。「どのパラメータをどの範囲で変えれば実現可能か」を効率よく計算する手法が未定。 | 高 | Phase 1 W3〜W4（INFEASIBLE早期検出実装時） |
| OI-03 | 設変連鎖変更（TD-09）の対応方針詳細。複数フィーチャー間の依存グラフ構造とPARAM_CHANGE伝播ルールが未設計。 | 中 | Phase 2開始前 |
| OI-04 | DeltaSpec統一型の具体的スキーマ設計（R4-29）。ECN固有フィールド（承認番号・有効日等）とReuseReasonフィールドの共存方法が未確定。 | 高 | Phase 1 W3〜W4（PartIngestionPipeline実装時） |
| OI-05 | FeatureClassifier（Class A/B/C分類）の初期ルールセット。何をClass Aとするかの定義が製品ドメインに依存しており、パイロット顧客との合意が必要。 | 最高 | Phase 1 W2（DFMルールエンジン実装時）、パイロット顧客確定後 |
| OI-06 | 機能インターフェースベクトル（64次元、R4-17）の具体的な特徴量定義。手動タグ付けの設計者負担コストの推定も未実施。 | 低 | Phase 2中盤（M10〜M12） |
| OI-07 | Socratic対話エンジン（R4-26, Phase 2）の対話フロー設計。質問テンプレートの種類・数・優先順位が未定。 | 低 | Phase 2開始前 |
| OI-08 | ERC-001〜006の具体的定義（R4-23）。各ERCコードが対応するフィールド・スキップ条件・保守値の割り当てが未設計。 | 高 | Phase 1 W11〜W12（CP1-Lite実装時） |
| OI-09 | ToolRegistryの具体的スキーマ設計（R5-02）。category / domain_tags / phase_availability / output_type_locked の全フィールドの型定義と必須化の実装方法が未確定。 | 高 | Stage 1実装時（Phase 1完了後）|
| OI-10 | WorkingContextのシリアライズ形式（JSON vs Protocol Buffers）。JSONはPydantic V2で自然だが、数百パーツ規模でのシリアライズ速度がStage 3以降のボトルネックになる可能性がある。 | 中 | Stage 2実装時（SessionDB設計時）|
| OI-11 | CAEソルバー選定（Nastran vs Abaqus vs LS-DYNA vs OpenFOAM vs CalculiX）。板金シェル要素解析と非線形材料モデル（DP780）への対応可否、OSS vs 商用コストの評価が必要。 | 高 | Stage 3実装前（Phase 2後半）|
| OI-12 | BVH実装ライブラリ選定（pythonocc組み込み vs scipy.spatial.cKDTree vs 独自実装）。pythonocc組み込みは依存追加なしだがAPIが低水準。200部品規模での実測が必要。 | 高 | Stage 3実装時（BVH干渉検出ツール実装時）|

---

### CLAUDE.mdへの追記が必要なセクション一覧

| セクション | 追記内容 | 優先度 |
|-----------|---------|--------|
| 本セクション（10番）| 本Round 4確定設計判断テーブル（今回追記済み） | 完了 |
| セクション9（意図的技術的負債テーブル）| TD-08（CATIA V5専用・V6非対応）、TD-09（設変連鎖変更未対応）の追加 | 高 |
| 新規セクション11 | PartIngestionPipeline 設計仕様（R4-05〜R4-08の実装詳細）| 中 |
| 新規セクション12 | ParameterWithConfidence型 スキーマ定義（R4-09〜R4-11）| 高 |
| 新規セクション13 | DeltaSpec統一型 スキーマ定義（R4-29）| 高 |
| 新規セクション14 | ERCコード定義テーブル（OI-08解決後に追記）| 高（Phase 1 W11〜W12後） |
| 新規セクション15 | FeatureClassifier Class A/B/C 定義（OI-05解決後に追記）| 最高（パイロット顧客確定後） |

---

## 11. Round 3 CESP最終設計 確定判断（2026-03-23）

> 詳細設計: 板金設計自動化_Round3_CESP最終設計.md
> 前提: C-01〜C-05（Round 2核心的結論）全採択済み

### 確定設計判断テーブル（R3-EC-01〜R3-EC-15）

| 判断ID | 内容 | Phase |
|--------|------|-------|
| **R3-EC-01** | `CESPInput.mode_hint` は `Literal` 型5値（auto/new_design/ec/reuse/exploratory）。UI非表示はW11〜W24。 | Phase 1 |
| **R3-EC-02** | `mode_hint="auto"` の推定ロジックは優先順位付き決定木。LLMに委ねない。 | Phase 1 |
| **R3-EC-03** | `ParameterWithConfidence.source` に `"reference_inherited"` を追加。流用設計で参照元の値を継承する際に記録する。 | Phase 1 |
| **R3-EC-04** | `CESPInput` の全フィールドは `Optional` かつ `None` 許容。バリデーションはPydantic V2 `@model_validator(mode='after')` で実施。 | Phase 1 |
| **R3-EC-05** | `PartIngestionPipeline.ingest()` の出力型を `ParameterWithConfidence` ベースの `PartParameters` に統一。Round 2の `ExtractedParameter` 型は廃止。 | Phase 1 |
| **R3-EC-06** | `DeltaSpec` 生成責務を `DeltaGenerator` クラスに分離。`CESPOrchestrator` は `delta_description` を `DeltaGenerator` に渡すだけ。 | Phase 1 |
| **R3-EC-07** | 動的CP挿入の判断閾値は `config.yaml` で管理（ハードコード禁止）。デフォルト: skip_threshold=0.75, light_cp_max=3, full_cp_min=4。 | Phase 1 |
| **R3-EC-08** | `CESPOrchestrator._enrich_with_context()` はイミュータブルな `EnrichedInput` 型を返す。元の `CESPInput` を上書きしない。 | Phase 1 |
| **R3-EC-09** | `PartIngestionPipeline` Phase 1実装範囲: 板厚・曲げR・フランジ高さの最小3パラメータ。その他は `confidence=0.0` のプレースホルダー出力（欠損ではなく「未実装」扱い）。 | Phase 1 |
| **R3-EC-10** | `ParameterWithConfidence.erc_code` は探索的設計専用。設変・流用では常に `None`。オーケストレーターがバリデーション。 | Phase 1 |
| **R3-EC-11** | `override_history` の各 `OverrideRecord` は設計根拠ドキュメントジェネレーターが自動参照。フォーマット: `[タイムスタンプ] [actor] [変更前値→変更後値] [変更理由]`。 | Phase 1 |
| **R3-EC-12** | `StepBRepExtractor` は Phase 1でAAG不使用。BBox + 円筒面検出のみ。AAGはPhase 2延期（TD-08として登録）。 | Phase 2延期 |
| **R3-EC-13** | `CatpartFeatureTreeExtractor` は Phase 1で実装。確信度0.92以上でCP発火頻度削減に直結。 | Phase 1 |
| **R3-EC-14** | `requirement_confidence` は `dict[str, Literal["confirmed","candidates","unknown"]]` 型に確定（3値のみ）。数値マッピングはオーケストレーター内部で実施。 | Phase 1 |
| **R3-EC-15** | `target_oem_id != oem_id` の場合、DFMエンジンは両OEMルールの積集合（より厳しい方）で検証。差分レポートを設計者に提示。 | Phase 1 |

### 新規技術的負債（TD-08〜TD-11）

| 負債ID | 内容 | 返済Phase | 事業リスク化閾値 |
|--------|------|---------|----------------|
| TD-08 | `StepBRepExtractor` Phase 2（AAG解析によるフランジ高さ・曲げ角度高精度抽出）| Phase 2前半（M7〜M9）| フランジ高さ誤差>1mmによるDFM誤判断5件以上 |
| TD-09 | pgvector移行 + feature_vector埋め込み生成（流用検索品質向上）| Phase 2前半（M10〜M12）| 検索3秒超 or DB件数200件超 |
| TD-10 | IGESサポート（現状エラー返却・STEPへの変換を設計者に求める）| Phase 2後半 | IGES入力案件3件以上 |
| TD-11 | `CESPInput` UIの公開（mode_hintのUI表示）| Phase 1 Go/No-Go評価後に判断 | Go/No-Go: 工数削減率≥30%達成後 |
| TD-12 | DesignAgentへの移行（Stage 1: ToolWrapping）| Phase 1登録 → Phase 1完了後返済 | Go/No-Go全充足後3ヶ月以内にStage 1未着手の場合 |
| TD-13 | マルチエージェント協調（VariantAgent / AssemblyAgent）| Phase 2登録 → Phase 2後半返済（Stage 3）| UC1要望が顧客3社以上から明示的に挙がった場合 |
| TD-14 | CAE統合（CAESubmitTool / CAEResultFetchTool 非同期設計）| Phase 2登録 → Phase 3返済（Stage 4）| FEA連携なし起因の失注2件以上 |
| TD-15 | マルチドメイン（樹脂・鋳造）対応 | Phase 3登録 → Phase 3返済（Stage 4）| 混在アセンブリ案件3件以上 |

### source別デフォルト確信度・tolerance テーブル（R3-EC-05確定値）

| source | デフォルト確信度 | デフォルトtolerance | requires_cp_confirmation |
|--------|--------------|-------------------|------------------------|
| `designer` | 0.98 | 0.00mm | False |
| `catpart_explicit` | 0.92 | 0.05mm | False |
| `step_extracted` | 0.70 | 0.15mm | False |
| `reference_inherited` | 0.80 | 0.10mm | False |
| `dfm_default` | 0.85 | 0.00mm | False |
| `llm_inferred` | 0.55 | 0.50mm | **True** |

### 確信度別DFMエンジン挙動（R3-EC-09補足）

| 確信度範囲 | DFM処理モード | 出力 |
|-----------|-------------|------|
| ≥ 0.75 | 通常チェック | 標準DFM結果 |
| 0.60〜0.74 | 保守値シフト | WARNING + シフト適用 |
| 0.20〜0.59 | 最保守値チェック | WARNING + CP確認要求 |
| < 0.20 | DFMチェックスキップ | WARNING: 設計者入力を求める |

### Phase 1追加工数サマリー（Round 2見積もりへの追加分）

| 実装項目 | 追加工数 | 組み込みWeek |
|---------|---------|-------------|
| ParameterWithConfidence + OverrideRecord型定義 | 0.5人日 | W1-W2 |
| DeltaSpec統一型 + DeltaGenerator | 2.0人日 | W5-W6 |
| CESPInputスキーマ + mode_hint推定ロジック | 1.5人日 | W11-W12 |
| CESPOrchestrator骨格（Phase A〜E） | 1.0人日 | W11-W12 |
| CPManager動的CP挿入ロジック | 0.5人日 | W11-W12 |
| DesignRationaleGenerator | 0.5人日 | W11-W12 |
| **合計** | **6.0人日** | W11〜W12に収まる |

---

---

## 12. エージェントアーキテクチャ設計判断（Round 5 改訂版）

### 記録日: 2026-03-23（改訂: CESP廃止・WorkingContext再設計反映）
### 前提: R1〜R4討議の確定設計決定 + 再検討_A（CESP廃止）+ 再検討_B（WorkingContext再設計）

---

### Round 5 再検討による確定変更事項（R5改訂）

| 変更ID | 変更内容 | 変更理由 | 影響コンポーネント |
|--------|---------|---------|-----------------|
| **R5-rev-01** | CESPOrchestratorを廃止する。フェーズ管理の責務はDesignAgentのReActループが担うため、固定シーケンスのOrchestratorは実装しない。最初からToolRegistryで設計する。 | OrchestratorのPhase A〜EはDesignAgentのReActループが自然に辿る処理経路と重複する。固定シーケンスは探索的設計・INFEASIBLE早期検出ケースで逆に足かせになる。旧R5-01の「CESPToolをフォールバックとして保持」も廃止する。 | DesignAgent, ToolRegistry（CESPToolの登録取り消し）|
| **R5-rev-02** | PropagationCSPを廃止する。制約間の連鎖伝播はDesignAgentがLoopDFMToolを反復呼び出しすることで代替する。 | CESPOrchestrator廃止に伴い存在意義を失う。LoopDFMToolの方向性付きフィードバックで代替可能。 | LoopDFMTool（反復呼び出しで代替）|
| **R5-rev-03** | ERC codes（ERC-001〜006）を独立概念として廃止する。未確定パラメータは `ParameterWithConfidence.confidence=0.0 / source="undefined"` で表現する。DFMRuleEngineは `confidence < 0.20` の場合にWARNINGを返す既存挙動で対応する。 | ERC codesはCESPOrchestratorのNullableフィールド処理として設計されており、Orchestrator廃止に伴い設計の文脈が消える。 | DFMRuleEngine（既存挙動で代替）|
| **R5-rev-04** | WorkingContext 4層設計（Project/Assembly/Part/Session）を廃止し、「DesignBootstrapContext（静的）＋ WorkingDesignState（動的）＋ ExternalMemory（ツール経由）」の2概念＋外部記憶構造に再設計する。 | Claude Codeの「必要時のみツールで読む」原則への正しい準拠。4層のLayer 2（AssemblyContext）はPhase 1スコープ（単一部品）では不要。Layer 3＋4の統合で命名を実態に合わせる。 | WorkingDesignState（新設）, DesignBootstrapContext（新設）|
| **R5-rev-05** | AssemblyContextの「常時20Kトークン保持」設計を廃止する。Phase 1ではAssemblyContext自体を実装しない（`assembly_id: Optional[str] = None`）。Phase 2でPlannerAgentとAssemblyTraversalToolを実装する際に追加する。 | Phase 1スコープは単一部品設計であり不要。早期実装はまだ存在しないユースケースのために複雑性を導入するリスクがある。 | WorkingDesignState（assembly_idをOptionalで予約）|
| **R5-rev-06** | 移行ロードマップを刷新する。旧「Stage 0: CESP構築 → Stage 1: ToolWrapping（8人日）」を廃止し、新「Stage 0: 決定論的ツール群先行実装（W1〜W4）→ Stage 1: DesignAgent最小実装（W5〜W12）」に変更する。ToolWrapping工程（8人日）は不要になり節約される。 | CESPを先に作ってからToolWrappingするパスは二重実装コスト（9〜11人日）が発生し、技術的負債を固定化する。最初からToolRegistryで設計することでこのコストを根本排除する。 | Phase 1実装計画, 移行ロードマップ |
| **R5-rev-07** | mode_hint推定ロジックを `ContextInferTool`（カテゴリC）として再設計する。CPManagerを `CPEvaluateTool`（カテゴリC）として再設計する。いずれもDesignAgentがReActループ内でツールとして呼び出す形に変える。 | Orchestrator廃止に伴い、中央集権型の推定・判断をToolとして分散化する。制御の向きの統一（LLMが主体でツールを呼ぶ）に準拠する。 | ContextInferTool（新設）, CPEvaluateTool（新設）|

---

### Claude Code原則からの意図的逸脱（明示的ドキュメント化）

| 逸脱番号 | 逸脱内容 | 逸脱の正当化 |
|---------|---------|------------|
| **逸脱1** | `iteration_count` と `infeasible_flags` をWorkingDesignStateに明示保持。LLMのコンテキストウィンドウに委ねない。 | DFM修正ループのN=5上限管理はLLMに数えさせると誤カウントが起きる。安全関連の状態変数は確定論的システムで管理する必要がある。 |
| **逸脱2** | DesignBootstrapContextを実行時に動的生成してSystem Promptに組み込む。CLAUDE.mdとの違い: 板金設計では oem_id（Toyota向け/Honda向け）が実行時パラメータ。 | 同じシステムを複数OEM向けに使い分けるため、起動時にdesign_context.yamlを読んでSystem Promptを動的生成する必要がある。 |
| **逸脱3** | GuardDFMTool（FreeCAD直前ガードレール）はLLMの判断をバイパスして物理的にFreeCADTool呼び出しを遮断する。 | DFM違反のある形状生成は製造後工程で取り返しのつかない問題になる。R4-14: Class A誤処理件数ゼロ絶対条件の技術的担保として必須。 |

---

### 確定設計判断テーブル（R5-01〜R5-15）

> 注記: R5-01, R5-04, R5-05, R5-10, R5-11 は上記R5改訂により内容を更新済み。

| 判断ID | 内容 | 根拠 | Phase | 影響コンポーネント | 取り消し条件 |
|--------|------|------|-------|-----------------|------------|
| **R5-01** | DesignAgentはReActループを板金設計版として実装する。LLMが主体でToolRegistryのツールを呼び出す「制御の向きの逆転」アーキテクチャを採用する。CESPOrchestratorは廃止し、最初からToolRegistryで設計する（R5改訂）。 | フェーズ管理の責務はDesignAgentのReActループが担う。固定シーケンスのOrchestratorは二重管理構造になるため廃止。 | Phase 1（Stage 0〜）| DesignAgent, ToolRegistry | — |
| **R5-02** | ToolRegistryにカテゴリA/B/C/Dを必須フィールドとして持たせる。`category` / `domain_tags` / `phase_availability` / `output_type_locked` の4フィールドを必須化する。 | カテゴリなしではエージェントがツール選択時に副作用リスクを判断できないため。 | Phase 1 | ToolRegistry, ToolDefinition | — |
| **R5-03** | DFMエンジンをLoopDFM（修正ループ内）とGuardDFM（FreeCAD直前）に二重配置する。GuardDFMはFreeCADToolの呼び出しを物理的に遮断する唯一のゲートウェイとする。意図的逸脱3として明示。 | LoopDFMのみでは反復中のパラメータ変更がDFM違反を再導入するリスクを排除できない。GuardDFMは「最後の防衛線」として機能する。 | Phase 1 | GuardDFMTool, LoopDFMTool, FreeCADTool | 取り消し不可（R4-14 Non-Negotiable に連動）|
| **R5-04** | WorkingContextは「DesignBootstrapContext（静的・System Prompt統合）＋ WorkingDesignState（動的）＋ ExternalMemory（ツール経由オンデマンド）」の2概念＋外部記憶構造とする。旧4層設計（Project/Assembly/Part/Session）は廃止する（R5改訂）。 | Claude Codeの「必要時のみツールで読む」原則への正しい準拠。AssemblyContextはPhase 1スコープ外。 | Phase 1 | WorkingDesignState, DesignBootstrapContext | — |
| **R5-05** | AssemblyContextはPhase 1では実装しない。`WorkingDesignState.assembly_id: Optional[str] = None` として型定義の拡張ポイントのみ予約する。Phase 2でPlannerAgentとAssemblyTraversalToolを実装する際に追加する（R5改訂）。 | Phase 1スコープは単一部品設計であり、早期実装はまだ存在しないユースケースのために複雑性を導入する。 | Phase 1不実装 → Phase 2実装 | WorkingDesignState, AssemblyTraversalTool | — |
| **R5-06** | コンテキスト圧縮は「情報の削除」ではなく「表現の変換 + DBへの退避」として設計する。CPの承認・却下記録とClass Aフィーチャーへの判断記録は圧縮を絶対禁止とし、永久保持する。 | 削除による圧縮は設計根拠・監査証跡の消失リスクを持つ。CP承認記録の消失は訴訟対応を不可能にするため絶対禁止。 | Stage 2〜全Phase | WorkingDesignState, ToolCallRecord, CPRecord | 圧縮後のRAG取り出し精度が50%未満になる場合（アーキテクチャを見直す）|
| **R5-07** | CAEジョブの投入と結果回収を非同期設計で分離する（CAESubmitTool / CAEResultFetchTool）。CAESubmitToolはjob_idを即時返却してブロッキングしない。 | 板金FEAの実行時間は数分〜数十分に及ぶ。同期待機ではDesignAgentが完全停止し、マルチパーツ設計の並列化効果がゼロになる。 | Stage 3〜4（CAE統合）| CAESubmitTool, CAEResultFetchTool | — |
| **R5-08** | Stage間移行は数値基準の充足のみを条件とし、ビジネス都合・スケジュール圧力による強制移行を禁止する。 | アーキテクチャ移行中のDFM品質劣化は顧客信頼の喪失に直結する。数値基準を満たさない移行は許容しない。 | 全Stage | 移行管理プロセス, Go/No-Go評価基盤 | 取り消し不可（品質保証の根幹）|
| **R5-09** | UC1（形状修正案の複数同時提案）に対応するVariantAgentの探索戦略を事前定義テンプレート（VT-01〜VT-06）で実装する。LLMが探索戦略を即興で生成することを禁止する。 | 探索戦略を事前定義することでVariantAgentの動作を予測可能にし、GuardDFMが評価できる形状変化の範囲をコントロールする。 | Stage 3 | VariantAgent, TemplateLibrary | テンプレートでカバーできない修正案が月3件以上の場合（追加を検討）|
| **R5-10** | 移行ステージを Stage 0（決定論的ツール群先行実装W1〜W4）→ Stage 1（DesignAgent最小実装W5〜W12）→ Stage 2（WorkingDesignState完全実装）→ Stage 3（マルチエージェント＋AssemblyContext）→ Stage 4（大規模並列＋CAE＋マルチドメイン）と定義する。旧「Stage 0: CESP → Stage 1: ToolWrapping」は廃止する（R5改訂）。 | CESPを先に作ってからToolWrappingするStage移行は二重実装コスト（9〜11人日）が発生し技術的負債を固定化する。最初からToolRegistryで設計することで移行コストを根本排除する。 | 全Phase | 全コンポーネント | — |
| **R5-11** | planモード（UC3: アセンブリ設計計画の自動立案）はStage 3で実装する。Phase 1ではAssemblyContextを実装しないため、planモードの本体もPhase 2（Stage 3）完了後に実装する（R5改訂）。 | planモードはAssemblyContextが前提条件。Phase 1スコープ外。 | Stage 3（完全実装）| PlannerAgent, AssemblyTraversalTool | Stage 2の単一エージェントだけで顧客要望の90%を満たせる場合（繰り上げを検討）|
| **R5-12** | CAE統合（UC2）はStage 4で実装する。Stage 3のplanモード先行が必須前提条件。 | アセンブリ構造の理解なしにCAE境界条件を自動設定できない。 | Stage 4（完全実装）| CAESubmitTool, CAEResultFetchTool | planモードなしで顧客の主要CAEユースケースを満足できる場合（実装順序を見直す）|
| **R5-13** | 数百パーツ並列設計（UC4）はBVH干渉検出とメッセージキュー（Redis Streams + Celery）が必須前提条件。両者のPoC完了をStage 3→4移行条件に追加する。 | BVHなしの干渉検出はO(n²)でリアルタイム不可。メッセージキューなしの並列設計はエージェント間の状態同期が保証できない。 | Stage 4 | GlobalOrchestrator, BVH干渉検出ツール, Redis Streams | 代替技術で同等性能を実現できることが判明した場合 |
| **R5-14** | カテゴリDツール（LLM推論ツール）はPhase 1（Stage 0〜1）での使用を禁止する。phase_availability=['stage2']として登録し、Stage 1では物理的に呼び出せない設定にする。 | Go/No-Go評価期間中の変数を最小化することが判定基準の信頼性を保証する。 | Phase 1（Stage 0〜1）は禁止 / Stage 2から有効化 | ToolRegistry, Go/No-Go評価基盤 | Phase 1 Go/No-Go達成後、Stage 2移行時に自動的に取り消し（有効化）|
| **R5-15** | カテゴリA（決定論的計算ツール）の出力はLLMが変更できない型安全設計で強制する。`output_type_locked: true` フラグを持つツールの出力はFrozen Pydantic Modelとして扱い、LLMが直接修正するコードパスを型レベルで封鎖する。 | BA計算・K-factor計算・DFM判定等の決定論的ツールの出力をLLMが「修正」すると数学的正確性が失われる。 | Stage 1（型定義）/ Stage 2（完全適用）| ToolInterface型定義, WorkingDesignState | 取り消し不可（R4-14 Non-Negotiable に連動）|

---

### 判断間の主要依存関係

```mermaid
graph TD
    R5_01["R5-01: DesignAgentを制御主体\nCESPOrchestrator廃止"] --> R5_08["R5-08: Stage移行は\n数値基準のみで許可"]
    R5_01 --> R5_02["R5-02: ToolRegistryに\nカテゴリA〜D必須フィールド"]
    R5_02 --> R5_14["R5-14: カテゴリDは\nPhase 1使用禁止"]
    R5_02 --> R5_15["R5-15: カテゴリA出力は\n型安全でLLM変更封鎖"]
    R5_01 --> R5_03["R5-03: DFMエンジン二重配置\nLoopDFM + GuardDFM"]
    R5_03 --> R5_15
    R5_01 --> R5_04["R5-04: WorkingContext\n2概念+外部記憶に再設計"]
    R5_04 --> R5_05["R5-05: AssemblyContext\nPhase 1で実装しない"]
    R5_04 --> R5_06["R5-06: コンテキスト圧縮\n= DBへの退避（削除禁止）"]
    R5_05 --> R5_11["R5-11: planモードは\nStage 3で実装"]
    R5_11 --> R5_12["R5-12: CAE統合は\nStage 4（planモード先行必須）"]
    R5_01 --> R5_07["R5-07: CAESubmitTool/\nCAEResultFetchTool分離（非同期）"]
    R5_12 --> R5_07
    R5_01 --> R5_09["R5-09: 探索戦略テンプレート\nVT-01〜VT-06事前定義（UC1）"]
    R5_09 --> R5_13["R5-13: 数百パーツ並列設計は\nBVH + メッセージキューが前提"]
    R5_08 --> R5_10["R5-10: マルチドメインは\nプラグイン型（変更禁止ゾーン分離）"]
```

---

### Round 5で未解決のまま残っている論点（OI-09〜OI-12は§10 OIテーブルに記載）

| 論点ID | 論点内容 | 優先度 | 推奨検討タイミング |
|--------|---------|--------|-----------------|
| OI-13 | LLMのReActループにおける「終了条件」の設計。DesignAgentがいつ「設計完了」と判断するかの定量的基準（DFM PASS + CP承認完了 + 必須パラメータ全確定 の3条件？）が未確定。 | 高 | Stage 1実装時（DesignAgent最小実装時）|
| OI-14 | GlobalOrchestrator（Stage 4）でのデッドロック防止設計。複数WorkerAgentが互いの部品の設計完了を待つ循環依存が生じた場合の検出・解消ロジックが未設計。 | 中 | Stage 3→4移行時 |
| OI-16 | DesignBootstrapContextのSystem Prompt展開テンプレート設計。OEM情報をどのようにSystem Promptに埋め込むかの形式が未定。DFMルールIDのリストをテキスト化すると何トークン消費するか。 | 高 | Phase 1 W1（材料DB実装と同時）|
| OI-17 | CPEvaluateToolの「active_cp_list」保持場所。WorkingDesignState（DB保存対象）に含めるのが正しいか、それともCPEvaluateToolが独立して管理するべきか。CPがセッションをまたぐ場合の設計が未定。 | 高 | Phase 1 W10〜W11（CPEvaluateTool実装時）|

---

### 段階的移行ロードマップ（Stage 0〜4 概要）

> 変更（Round 5改訂）: 旧「Stage 0: CESP構築（Phase 1 W1〜W24）」を「Stage 0: 決定論的ツール群先行実装（W1〜W4）」に刷新。旧「Stage 1: ToolWrapping（8人日）」を「Stage 1: DesignAgent最小実装（W5〜W12）」に変更。ToolWrapping工程は廃止（節約: 8人日）。

| Stage | 時期 | 主な変化 | 追加開発工数 | Go/No-Go移行条件の核心 |
|-------|------|---------|-----------|----------------------|
| **Stage 0** | Phase 1 W1〜W4 | 決定論的ツール群先行実装（カテゴリA/C基盤）| Phase 1見積もり内 | LoopDFMTool + GuardDFMTool + BendAllowanceTool 単体テスト全Pass |
| **Stage 1** | Phase 1 W5〜W24 | DesignAgent最小実装（1ターン動作）→ UI統合MVP → Go/No-Go評価 | Phase 1見積もり内 | 6指標全Go/Conditional + Class A = 0 |
| **Stage 2** | Phase 2前半 M7〜M15 | WorkingDesignState完全実装。DesignAgentが完全なReActループで動作 | 20人日 | フォールバック率 <10% + DFM <5%（3ヶ月）|
| **Stage 3** | Phase 2後半〜3 M16〜M24 | マルチエージェント（Planner/Worker/Reviewer/Variant）+ AssemblyContext実装 | 56人日 | 並列生成率 ≥85% + CAE実行率 ≥80% |
| **Stage 4** | Phase 3以降 M25〜 | 数百パーツ並列 + 樹脂・鋳造ドメイン追加 | 87人日 | 100部品並列完了率 ≥80% |
| **合計追加** | | ToolWrapping廃止により旧比8人日削減 | **163人日（8.15人月）** | |

---

## 13. PoC本題: 中立面点群→AI形状復元 アーキテクチャ確定（Round 6）

### 記録日: 2026-06-17
### 議事の経緯
ユーザー提案: 「板金STPから中立面点群を作成し、それをCADの幾何形状に戻すAIを作成する」。
方針の妥当性確認を要求されたため検討した結果、**方向性は正しいが`stage1_poc_plan.md`の設計に1点未解決の矛盾があった**ことが判明し、これを解消する形で確定した。

### 発見した矛盾と解消
`stage1_poc_plan.md` はGTを **UDF**（SDFではない）と定義していたが、理由は「板金は開いた面だから」とされていた。しかし `extract_solid_step.py` が抽出するのは体積を持つ**閉じたソリッド**であり、閉じたソリッドならSDFが定義可能なはずで、矛盾していた。
今回ユーザーが提案した「中立面」を基準形状にすると、中立面は厚みゼロの真の開多様体になりinside/outsideが定義不可能になるため、UDFを使う必然性が生まれる。**→ GTの基準形状を「ソリッド」から「中立面メッシュ」に変更することで確定。**

### 確定設計判断テーブル（R6-01〜R6-06）

| 判断ID | 内容 | 根拠 | 状態 |
|--------|------|------|------|
| **R6-01** | GTの基準形状を中立面メッシュとする（ソリッド外皮ではない）。中立面メッシュは元メッシュの三角形コネクティビティをそのまま流用し、頂点座標のみ中立面投影点 `P_mid` に置換して生成する（Poisson再構成等は使わない）。 | UDFの設計選択（SDFでなくUDF）と整合させるため。閉じたソリッドにUDFを使う矛盾を解消する。コネクティビティ流用により穴境界・外周境界の形状が保証され、再構成アーチファクトを回避できる。 | **確定** |
| **R6-02** | 中立面投影時、ray-castが thickness の3倍を超えてもヒットしない頂点は「壁面（切断端面）」として除外する。壁面に接する三角形は中立面メッシュの生成対象から自動的に除外される。 | 切断端面（穴境界・外周）はレイキャストでは意味のある中立面像を持たないため。除外しないと中立面メッシュが歪む。 | **確定** |
| **R6-03** | 中立面点群生成は `hole_filler.py` の出力（`body###_filled.stp`）に対してのみ実行する。穴埋め前のソリッドには適用しない。 | 未処理の小穴（ジグ穴等）が壁面除外ロジックを乱し、不要な境界ループを中立面メッシュに混入させるため。 | **確定** |
| **R6-04** | 中立面投影の中核ロジックは `annotator.py` の `project_to_midsurface()` を再利用する。新規実装するのは「全頂点へのバッチ適用」と「壁面除外」のみ。 | 既存の検証済みロジック（thickness=3.068mm等で動作確認済み）の重複実装を避けるため。 | **確定** |
| **R6-05** | Stage1の出力スコープは「メッシュ/陰関数による形状復元」のみとする。曲げ角度・フランジ長等を編集可能パラメータとして出力するパラメトリックCAD復元は含めない。 | ユーザー確認済み（2026-06-17, AskUserQuestion回答: 「メッシュ復元のみ(推奨)」）。スコープを広げると5週間のPoC期間に収まらない。パラメトリック化は後続フェーズで別モデル（特徴フィッティング、CLAUDE.md §9のテンプレートライブラリ構想と連携）として実装する。 | **確定** |
| **R6-06** | デコーダの出力ヘッドはUDF + normalのみとし、thickness（局所板厚）ヘッドは追加しない。板厚は `2×vol/total_face_area` による部品ごと一定値推定を継続使用する。 | ユーザー確認済み（2026-06-17, AskUserQuestion回答: 「追加しない」）。スタンピングの減肉情報は失われるが、Stage1のモデルをシンプルに保つことを優先。 | **確定** |

### 環境上の修正（要注意）
`stage1_poc_plan.md` はGT UDF計算に `open3d.RaycastingScene.compute_distance()` を使う前提だったが、**この conda 環境に open3d はインストールされていない**（本セッションで確認済み）。代替として、既にインストール済みの **`trimesh.proximity.closest_point()`** を使い、最近傍点までの距離を符号なし距離として扱う。中立面メッシュは非ウォータータイト（開多様体）だが `trimesh.proximity` は三角形への最近傍距離計算であり、ウォータータイト性を要求しないため問題なく動作する。

### 確定アーキテクチャ（3ステージ）

```
Stage A: 決定論的データ生成（AI不使用、部品ごとにオフライン）
  body###_filled.stp
    → [A-1] テッセレーション（annotator.step_to_pyvista 再利用）
    → [A-2] 壁面除外つき中立面投影（annotator.project_to_midsurface を全頂点へバッチ適用、R6-02）
    → [A-3] 中立面メッシュ構築（元コネクティビティ流用、R6-01）→ mid_surface.ply
    → [A-4] ノイズ入り学習用点群（8192pt サブサンプル + ノイズ + マスキング）
    → [A-5] GT UDFクエリサンプリング（trimesh.proximity使用、環境上の修正を反映）
    → dataset/<part>.h5 { points, normals, query_xyz, query_udf }

Stage B: VecSet Autoencoder（stage1_poc_plan.md踏襲、R6-06によりthicknessヘッドなし）
  points[8192,3] → FPS(256)+kNN(32) → mini-PointNet → tokens[256,384]
    → Self-Attn Encoder×6L → Cross-Attn(M=128 learnable queries) → Z[128×384]
  query_xyz[Nq,3] → Cross-Attn Decoder(query attends to Z) → MLP → { UDF(query), normal(query) }

Stage C: 幾何復元（決定論的、AI後処理、R6-05によりメッシュ出力のみ）
  学習済みモデル + 密グリッド評価 → UDF専用メッシュ抽出（勾配投影+ボールピボット等）
    → 中立面メッシュ → 部品一定thicknessで±t/2オフセット → reconstructed.stl
  ※ パラメトリックB-Rep/STEPフィーチャーツリーではない（R6-05）
```

### 既存ツールとの接続
| 既存ツール | 役割 | 接続点 |
|---|---|---|
| `hole_filler.py` | 小穴除去 | Stage A の必須前処理（R6-03） |
| `annotator.py` の `project_to_midsurface()` | 単一点の中立面投影 | Stage A-2 のコアロジックとして再利用（R6-04） |
| `annotator.py` の締結点アノテーション | 溶接・ボルト点記録 | 将来、拘束トークン（K×(x,y,z,type_id)）としてStage Bのエンコーダに注入する設計（Round 2確定）と統合可能 |

### Stage A 実装完了
- `midsurface_sampler.py` 実装済み（Stage A-1〜A-5、`hole_filler.py`/`annotator.py`と同じCLI+bat構成）

---

## 14. Stage B/C 実装・デバッグ確定事項（Round 7）

### 記録日: 2026-06-17
### 前提
PyTorch環境確認済み（pytorch3d/torch-cluster は未インストール、FPS/kNNは純PyTorch自前実装で対応）。単一部品（`body002_filled_dataset.h5`）でのoverfitサニティチェックを実装検証の基準とした（「モデルが1部品のフィールドすら記憶できないなら多部品学習は無意味」というH2仮説のテスト）。

### Stage B（VecSet Autoencoder）確定設計判断（R7-01〜R7-05）

| 判断ID | 内容 | 根拠/経緯 |
|--------|------|----------|
| **R7-01** | データセットに `query_grad`（UDFの勾配方向、trimeshの最近傍点ベクトルから導出。古典的な面法線ではない）を追加。 | Stage Cの勾配投影ステップ（zero level setへのNewton的射影）に必須。 |
| **R7-02** | `dataset.py` で部品ごとの座標正規化（中心化 + 等方性スケーリングで概ね[-1,1]へ）を無条件に適用。`query_udf`(mm)と`query_grad`(単位ベクトル)は正規化しない。学習ループはモデル出力（正規化空間のUDF）を保存済みの`scale`係数で再スケールしてmm単位の損失を計算。 | 生のSTEP/ワールド座標はBIWアセンブリ原点から数百mm離れており、小さい重み初期化のMLPを飽和させない範囲に値が収まらない。正規化なしのoverfit実行は全く収束しなかった（実証済み）。 |
| **R7-03** | Truncated/clamped distance loss（DeepSDF/NDF方式）。`pred_udf`と`gt_udf`の両方を学習前に`clip_dist_mm`（デフォルト5.0mm、`train.clip_dist_mm`）でクランプしてからL1損失を計算。 | 生の遠方ターゲットは最大~200mmに達し、クランプなしでは損失が遠方項に支配され、Stage Cが実際に必要とする近傍領域への勾配信号が枯渇する（実証: 1500epoch無clamp実行でudf_l1=378mm→0.74mmと収束する一方、near_maeは~0.42-0.44mmで全く改善せず）。 |
| **R7-04** | `QueryDecoder.head`の最終`nn.Linear(dim,4)`の出力バイアスを初期化時に`self.head[-1].bias[0]=-5.0`に設定（`softplus(-5)≈0.0067`、ほぼ0）。 | デフォルト初期化では`out[...,0]≈0`→`softplus(0)≈0.693`→`scale`(~300-450mm)倍後、初期状態で全クエリが200-300mm予測になる。R7-03のクランプ勾配は`clip_dist_mm`超過域で厳密にゼロのため、この誤初期化と組み合わさると学習が完全に停止する致命的バグだった（実証: 600epoch実行でnear_maeが0.0056mmではなく390mmまで悪化）。 |
| **R7-05** | `QueryDecoder`の`query_xyz`に対しNeRF式Fourier位置エンコーディング（`FourierFeatures`: `[x, sin(2^k·π·x), cos(2^k·π·x)]`, `k=0..n_freqs-1`, デフォルト`n_freqs=8`）を`PositionalMLP`の前段に追加。**ユーザー承認済み**（2026-06-17、AskUserQuestionで「Fourier位置エンコーディングを追加」を選択）。 | 座標MLPのspectral bias（Rahaman 2019、Tancik 2020）対策。3つの独立した学習設定すべてでnear_mae が~0.42-0.44mmで頭打ちになっており、これは「近傍帯の中央値を予測する定数モデル」の理論MAE(~0.5mm)と一致していた。Fourier特徴追加後、near_mae=0.0056mm（閾値0.1mm、PASS）まで収束（2500epoch）。 |

### Stage C（幾何復元）確定設計判断（R7-06〜R7-07）

| 判断ID | 内容 | 根拠/経緯 |
|--------|------|----------|
| **R7-06** | Stage Cの近傍候補点選択を「絶対距離バンド」（`voxel_size * band_factor`）ではなく「ランク（予測UDFの小さい順 top-K）」方式に変更。`target_k = band_factor * grid_res^2`（2次元多様体の中立面はgrid_res^3体積中O(grid_res^2)セルとしか交差しないという幾何学的事実に基づく）。 | R7-03のtruncated-distance lossはネットワークに「clip_dist_mm を超えること」しか要求せず、それ以上大きい値を予測するインセンティブを与えない。そのため学習済みモデルの予測UDFは[-1,1]^3グリッド全体で小さい値域に頭打ちになりうる（実証: grid_res=48で全グリッドの99.6%=110147/110592点が絶対バンド閾値内と誤判定され、`pyvista.reconstruct_surface()`が110k点で253秒かかった。本番解像度grid_res=96では約7倍の点数になり数時間規模になる見込み）。ランクベース選択は校正の絶対値に依存しないため頑健。 |
| **R7-07** | Stage Cの評価グリッド（`eval_grid()`）を、R7-02正規化後の等方的`[-1,1]^3`立方体全体ではなく、実際の入力点群のbounding box（各軸15%パディング付き、`bbox_pad_frac=0.15`）に制限。**ユーザー承認済み**（2026-06-17、AskUserQuestionで「評価グリッドを入力点群のbboxに制限」を選択、再学習による損失見直しの代替案より優先）。 | R7-02の正規化は部品の最長軸の半径で全軸に等方的スケールを適用するため、非等方形状（薄く細長い板金部品）では正規化立方体の大部分が実体から遠い「未学習の空っぽな体積」になる。この領域でネットワークが校正されておらず、実距離と無関係な「幽霊」低UDF極小値を点在させてしまうことが判明（実証: R7-06のランク選択candidate の86%が最近傍の実学習点から正規化距離0.05超=約20mm以上離れており、中央値は0.58=約220mm。この結果、再構成メッシュのGT中立面に対する平均距離が328mmに達した）。bbox制限後、GT→Recon平均距離 15.3mm→7.2mm、Recon→GT平均距離 328mm→75mm（4.4倍改善）。境界（開いた切断端面）付近の残存オーバーシュート（Recon→GT最大274mm）は既知の残課題（TD-16として登録）。 |
| **R7-08** | Stage Cの密度向上（R7-07時点でRecon→GT=75mmだった残課題のうち「内部領域の精度」側）を、点群を太らせて`reconstruct_surface()`を再実行する方式（試作・破棄）ではなく、**メッシュの1→4三角形分割＋再投影**方式で実装。`subdivide_and_project()`が既存メッシュの三角形コネクティビティ（`reconstruct_surface()`を1回だけ実行して取得）を再利用し、各辺の中点のみを`gradient_project()`でゼロレベル集合上に再投影する。`--refine-rounds`（デフォルト3、`configs/model.yaml`の`reconstruct.refine_rounds`）で反復回数を指定。 | 根本原因の事実確認（本セグメントで実施）：Recon→GT=75mmの内訳は**2つの独立した原因の合成**と判明。①境界外挿（後述、未解決）と②グリッド密度の天井——`gradient_project()`はUDF勾配方向（面に垂直）にしか点を動かさず、面に沿った（接線方向の）点間隔は`eval_grid()`の粗いグリッド間隔のまま（grid_res=96・部品の最大辺~900mmで間隔~10.2mm）に固定される。これがboundary外の outliers を除いた内部領域でも~6-7mmのGT→Recon床として観測されていた。先行研究なし（これは本実装のメッシュ密度設計の問題であり、UDF特有の限界ではない）。**当初の試作**（接線方向ジッター＋点群を太らせて`reconstruct_surface()`を再実行）はVTKのHoppeアルゴリズムが超線形スケールすることが判明し撤回（実証: 点数5倍→実行時間57倍、138,240点で401.8秒、580,608点は8分以上経過後もキルするまで終了せず）。分割＋再投影方式は`reconstruct_surface()`を粗いメッシュに対して1回だけ実行し、以降は線形スケールする`gradient_project()`（~70k点/秒）のみで密度を上げるため、この問題を回避する。 |

### 実装ファイルとデータフロー
- `src/model/vecset_ae.py`: `FourierFeatures`/`PositionalMLP`/`QueryDecoder`/`VecSetAE`（R7-04/R7-05実装）
- `src/model/dataset.py`: `MidSurfaceUDFDataset`（R7-01/R7-02実装）
- `src/model/train.py`: `weighted_losses()`（R7-03実装）、`--n-freqs`/`--clip-dist-mm` CLI引数
- `src/model/reconstruct.py`: `eval_grid()`（R7-07実装）、`reconstruct_midsurface()`（R7-06実装）、`subdivide_and_project()`（R7-08実装）
- `src/model/validate_refine.py`: R7-08の`--refine-rounds`スイープ検証スクリプト（新規追加）
- `configs/model.yaml`: `vecset_ae.n_freqs`, `train.clip_dist_mm`, `reconstruct.band_factor`（意味がR7-06でvoxel倍率→grid_res^2倍率に変更）、`reconstruct.refine_rounds`（R7-08追加）

### 検証結果（body002、単一部品overfitサニティ）
| 指標 | 値 | 判定 |
|------|-----|------|
| near_mae（学習後最良） | 0.0056mm | PASS（閾値0.1mm）|
| `reconstruct_surface()`実行時間（grid_res=96） | 7.1秒（R7-06適用後。R7-06なしでは候補点が桁違いに多く、grid_res=48でも253秒） | 大幅改善 |
| GT→Recon中立面メッシュ距離（平均、refine_rounds=3） | 1.34mm（refine_rounds=0時点: 7.2mm） | PASS（目標~1-2mm） |
| Recon→GT中立面メッシュ距離（平均、refine_rounds=0〜3） | 75mm→55mm（refine_rounds非依存、境界外挿が支配的） | 未解決（Fix 1、保留中） |

### R7-08 `--refine-rounds` スイープ検証結果（body002、validate_refine.py）
| refine_rounds | 総実行時間 | 頂点数/三角形数 | GT→Recon 平均/中央値 | Recon→GT 平均/中央値 |
|---|---|---|---|---|
| 0 | 9.9秒 | 16,551 / 32,036 | 7.20mm / 7.22mm | 74.74mm / 63.83mm |
| 1 | 10.7秒 | 65,426 / 128,144 | 3.64mm / 3.37mm | 70.25mm / 63.73mm |
| 2 | 13.8秒 | 259,284 / 512,576 | 2.08mm / 1.75mm | 60.53mm / 52.07mm |
| 3 | 25.6秒 | 1,031,432 / 2,050,304 | **1.34mm / 1.03mm** | 55.08mm / 45.14mm |

**結論**: GT→Recon（内部領域の形状再現精度＝Fix 2の対象）はrefine_rounds=3でユーザー承認済みの現実的目標（内部~1-2mm）を達成し、実行時間も25.6秒と実用範囲内。`configs/model.yaml`のデフォルトを`refine_rounds=3`に設定済み。一方Recon→GT（境界外挿によるゴーストメッシュ＝Fix 1の対象）はrefine_roundsを上げても55〜75mmの範囲で高止まりしており、密度向上では原理的に改善しない（後述の根本原因②参照）。これによりFix 2とFix 1が完全に独立した問題であることが実験的に再確認された。

### Recon→GT=75mmの根本原因分析（事実確認・先行研究調査、本セグメント実施）

ユーザーからの依頼「Recon→GTの75mmは大きい、1mm以内に抑えたい。事実確認と先行研究調査の上で原因分析と解決策を提案してほしい」に対する調査結果。**2つの独立した原因の合成**と判明：

**原因①: 境界（切断端面）でのUDF外挿（未解決、Fix 1として保留中）**
- 事実確認: Recon→GTの距離が大きい点（>20mm）とGT中立面の開いた境界線（`trimesh`の`outline()`で抽出）までの距離には相関係数1.0という極めて強い相関があることを確認。「悪い点」はGT境界エッジから平均88.41mm、「良い点」は平均2.98mmと、ほぼ完全に境界由来であることが実証された。
- メカニズム: 中立面の境界（部品の切断端）では、真のUDF（unsigned distance field）に数学的な特異点（非微分可能なキンク、"cut locus"）が生じる。なめらかな関数しか表現できないニューラルネット（Fourier特徴を使っていても）はこのキンクを正確に再現できず、境界を超えた領域でも予測UDFが偽って小さい値（「表面の近く」）を返してしまい、本来存在しない余分なメッシュ（10〜270mm規模、場所により大きさが不均一）が生成される。一様な「縁取り」ではなく、ネットワークの境界認識が崩れた箇所ごとに不規則に発生する「ボコボコした余剰メッシュ」である（ユーザーの「パンの耳」という例えに対し、実際は不均一な突起である旨を回答済み）。
- 先行研究での裏付け: UDFベースのニューラル暗黙的サーフェス再構成における既知の未解決問題として文献で広く報告されている。"neural UDFs tend to have larger errors near the zero-level set and its cut locus"（arxiv.org/pdf/2309.08878）、"UDFs struggle to precisely achieve a zero value... theoretically non-differentiable at the zero level set... making it difficult to generate open boundaries"（arxiv.org/pdf/2210.02757）。関連する対策手法としてMeshUDF（github.com/cvlab-epfl/MeshUDF）、CAP-UDF（Consistency-Aware Probabilistic UDF、勾配ベースの符号割り当て+Marching Cubesを使用、junshengzhou.github.io/CAP-UDF）、Neural Dual Contouring、NeuralUDFを確認。
- 状態: ユーザーへ発生メカニズムを詳細説明済みだが、対策の選択（幾何学的クリップ／UDF専用メッシング手法の導入／両方／対応しない）は**ユーザー判断待ちで保留中**。

**原因②: 接線方向の密度天井（解決済み、R7-08）**
- `gradient_project()`は点をUDF勾配方向（面に垂直）にのみ動かすため、面に沿った方向の点間隔は`eval_grid()`の粗いグリッド間隔（grid_res=96、本体ほどの大きさの部品で~10.2mm）に固定されたままだった。これが境界由来の外れ値を除いた内部領域でも~6-7mmのGT→Recon床として現れていた。
- R7-08（メッシュ分割＋再投影、上表参照）で解決。

### 新規技術的負債（TD-16）
| 負債ID | 内容 | 返済Phase | 事業リスク化閾値 |
|--------|------|---------|----------------|
| TD-16 | Stage C再構成メッシュの境界（切断端面）付近に残存するオーバーシュート（GT外への外挿、Recon→GT最大距離274mm規模、root cause確定: UDFのcut locus非微分可能性、文献的にも既知の未解決問題）。R7-08の密度向上では改善しない（refine_rounds 0→3でRecon→GT平均75mm→55mmとわずかに改善するのみで、密度を上げても境界外挿そのものは解消されない）ことを実験的に確認済み。**→ §15（Round 8）でR7-11+R7-12の決定論的ゲート採用により緩和済み（mitigated）。mean=8.57mm/max=47.27mmまで改善。max誤差の完全解消は引き続きユーザー判断待ち。** | §15で緩和対応実施、max誤差の根治は保留 | mean誤差は実用範囲内に収束。max誤差（47.27mm）の根治要否はユーザー判断次第 |

---

## 15. TD-16解決: 確信度/距離ゲート + Deep Ensembles比較 確定事項（Round 8）

### 記録日: 2026-06-17
### 前提
Round 7（§14）で特定したTD-16（境界cut locusでのUDF外挿によるゴーストメッシュ、Recon→GT最大274mm）に対する根本対応の検討・実装・検証ログ。候補(a)「入力点群のbboxによる幾何学的クリップ」を、より一般的な決定論的ゲート方式（学習済みネットワークの確信度・距離信号に基づく候補点フィルタ）として具体化した。

### 確定設計判断テーブル（R7-09〜R7-12）

| 判断ID | 内容 | 根拠/経緯 |
|--------|------|----------|
| **R7-09** | データセットのUDFクエリサンプリングに「境界シャドウ」カテゴリ（`query_category=2`）を追加。GT中立面の開いた境界線（cut edge）近傍を重点的にオーバーサンプリングし、学習損失にEikonal正則化（`lambda_eikonal=0.05`、`\|\|∇UDF\|\|≈1`への正則化）を追加。 | cut locus（境界での非微分可能なUDFキンク）をネットワークがより正確に学習できるようにするための直接対策。境界シャドウクエリがないと、ネットワークは境界近傍のUDF形状を訓練データから一度も明示的に見ないまま外挿することになる。 |
| **R7-10** | `QueryDecoder`に4本目の出力ヘッド（confidence、sigmoid出力）を追加。学習目標はquery点から最近傍GT表面までの距離が`conf_horizon_mm`（デフォルト15mm）以内なら1、それ以遠なら0に近づくよう、`lambda_conf=0.3`で学習。Stage Cは`conf_threshold`を下回るグリッド点を候補から除外する。 | ネットワーク自身に「ここは自分が自信を持って予測できる領域か」を明示的に出力させることで、境界外挿領域を学習済みモデル自身の信号で検出させる試み。 |
| **R7-11** | Stage Cの候補点選択に、決定論的kNN距離ゲートを追加。各グリッド点について実際の入力点群（h5の`points`）最近傍点までの距離を計算し、`input_dist_threshold_mm`（検証値10mm）を超える点は確信度・UDFランクによらず候補から除外。 | R7-10の学習済み確信度ヘッドは外挿領域でも高い確信度を誤って出力するケースがあり（学習データの分布外で校正が崩れるため）、決定論的な幾何学的ゲートで補完する必要があった。Recon→GT mean 53.83mm→31.57mm（R7-10単独比）に改善。 |
| **R7-12** | Stage Cの再構成メッシュ頂点（粗メッシュ＋各`refine_rounds`の分割後頂点）について、`reconstruct_surface()`に実際に入力した候補点群（`proj_xyz`）への最近傍距離が`prune_dist_threshold_mm`（検証値12mm）を超える頂点を削除する`prune_far_vertices()`を追加。粗メッシュ生成直後だけでなく、**毎回の`refine_rounds`の後にも再適用**（ユーザー承認: 「フル実装（推奨）」）。 | VTKの`reconstruct_surface()`（Hoppeアルゴリズム）自体が、疎な領域で自身の入力候補点群から大きく外挿した粗メッシュ頂点を合成することを実証（外挿距離とGTまでの距離に相関係数0.9917）。これはR7-10/R7-11とは独立した第3の原因。`refine_rounds`を増やすほど誤差が悪化する非単調現象を確認（prune未適用時、prune=8mmでrefine_rounds=2のmax=44mmがrefine_rounds=3でmax=75mmに悪化）。`subdivide_and_project()`の各ラウンドが粗メッシュの外挿頂点を起点に新たな小規模外挿を再導入するため、毎ラウンド後のprune再適用が必須と判明。 |

### Deep Ensembles比較（ユーザー承認済み「両方実装して比較する」を完遂、結果: 不採用）

5シード（seed0-4、`checkpoints_r710` + `checkpoints_r710_seed1〜4`、ハイパーパラメータ完全同一・`--seed`のみ変更）を独立学習し、アンサンブル予測UDFの標準偏差（disagreement）をOOD/外れ値ゲートとして使えるか検証した（Lakshminarayanan et al. 2017のDeep Ensembles epistemic uncertainty推定に準拠）。

| 検証項目 | 結果 |
|---|---|
| R7-10単独ベースラインの既知ワースト30頂点（平均d_gt=245mm）でのensemble_std | mean=2.136mm（全グリッド平均ensemble_std=2.156mmと統計的に区別不能） |
| corrcoef(ensemble_std, d_gt)（ワースト30点） | **0.1671**（R7-12のdistance-to-candidate-cloud信号が達成した0.9917と対照的にほぼ無相関） |
| ensemble_std単独をゲートとして使用（std_threshold=1.0mm） | Recon→GT mean=45.28mm, max=257.36mm（R7-10単独ベースラインのmax=244.77mmよりむしろ悪化） |
| ensemble_std単独をゲートとして使用（std_threshold=0.5mm / 0.2mm） | mean=63.91mm/52.29mm, max=270.68mm/253.45mm（いずれもR7-11単独のmax=182.83mmより悪い） |

**結論（不採用、理由分析込み）**: Deep Ensembles方式は本タスクでは機能しなかった。原因は、5メンバー全てが同一アーキテクチャ・同一訓練データ・同一損失関数（特にR7-03のtruncated-distance loss）を共有しており、このlossが設計上もたらす「予測UDFの値域天井」効果（R7-06で既出）が全メンバーに**系統的に共通**して生じるため。Deep Ensemblesが捉えようとするepistemic uncertaintyは独立初期化由来の予測のばらつきに依拠するが、本ケースの誤差はランダムな初期化のばらつきではなく損失関数設計に起因する系統バイアスであり、アンサンブルメンバー間で「揃って間違える」ため、disagreementとして検出できない。

### 最終採用解（TD-16のRecon→GT外れ値対策として確定）

**R7-11 + R7-12 決定論的ゲートの組み合わせ**（Deep Ensemblesではなく）をユーザーが最終承認・確定（2026-06-17、「採用案はR7-11+R7-12の決定論的ゲートで確定します」）。`validate_refine.py`による本番コードパス（`reconstruct_midsurface()`）経由の検証値:

```
conf_threshold=0.0, input_dist_threshold_mm=10, prune_dist_threshold_mm=12, refine_rounds=3
Recon->GT: mean=8.57mm median=6.54mm p90=18.97mm max=47.27mm
  (R7-10単独ベースライン比 mean 53.83mm→8.57mm = 6.3倍改善、max 244.77mm→47.27mm = 5.2倍改善)
GT->Recon: mean=1.62mm median=1.13mm p90=3.42mm max=13.23mm
  (内部精度はR7-10単独ベースラインのmean=1.58mmからほぼ無劣化)
```

実プロダクションCLI（`reconstruct.py --conf-threshold 0.0 --input-dist-threshold-mm 10 --prune-dist-threshold-mm 12 --refine-rounds 3 --save-intermediate`）でbody002に対し出力ファイル生成・動作確認済み（4,080頂点の粗メッシュ→208,881頂点の最終メッシュ、`poc_result_viewer.py`での目視確認用に`body002_r711r712_*.ply`一式を`src/model/`に保存済み）。

残存課題: max=47.27mmは依然としてGT→Recon内部精度（mean 1.62mm）と比べて大きい。境界cut locus由来の外挿は完全には消えておらず、必要であればR7-09のEikonal正則化の重み増加や、TD-16記載のMeshUDF/CAP-UDF方式の導入が次の改善候補となる（未着手、ユーザー判断待ち）。

### 実装ファイル更新
- `src/model/reconstruct.py`: `prune_far_vertices()`（R7-12新規）、`reconstruct_midsurface()`に`prune_dist_threshold_mm`引数追加、`process()`/`main()`に`--prune-dist-threshold-mm` CLI引数追加
- `src/model/validate_refine.py`, `src/model/diagnose_outliers.py`: `--prune-dist-threshold-mm` CLI引数追加（インターフェース同期）
- `src/model/ensemble_compare.py`: 新規追加（Deep Ensembles比較専用、本番パイプライン外のアドホックスクリプト）
- `src/model/train.py`: `--seed`引数で複数シード学習に対応（既存、`checkpoints_r710_seed1〜4`生成に使用）
- `src/viewer/poc_result_viewer.py`: 既存（Round 7時点で多層トグル対応済み）。`--recon-dir`配下の`*_midsurface.ply`を自動discoverし、GT/最終メッシュ/各refineステージ/Stage A入力点群/クエリカテゴリ別点群を個別トグル表示。任意の2メッシュ層をReference/Targetに選んでper-vertex距離ヒートマップ表示可能。

---

## 16. トポロジーパッチ 多重seed分散検証 確定事項（Round 9）

### 記録日: 2026-06-18
### 前提
§14で確認したTD-16のゴーストメッシュ症状に対し、「Stage Aのray-castノイズ(GT境界トポロジーの断片化)が原因の一つではないか」という仮説のもと、`midsurface_sampler.py`に`patch_isolated_invalid_vertices()`（境界の孤立無効頂点を周囲80%以上が有効なら補間する処理、最大3反復）を実装した（Stage 1、ユーザー承認済み）。body002で検証したところGTの境界ループは211→71（-66%）に減少し、270頂点を回収した。

Stage 2として、このパッチを適用した状態でデータセットを再生成しseed=0で再学習・Stage C検証したところ、Recon→GT/GT→Recon指標は無変化〜やや悪化という結果になった。ただし単一seedの比較であり、再学習のnear_mae(0.0814mm)が元のseed0実行(0.0056mm)より大幅に悪かったため、「データセット変更の効果」と「訓練の通常の分散」を切り分けられないという限界があった。ユーザーの判断で複数seedによる分散の切り分けを実施した（Round 9、本セクション）。

### 実施内容
- unpatched(元データ、backup: `body002_filled_dataset_pretopopatch.h5`)で既存のseed0-4チェックポイント(`checkpoints_r710`, `checkpoints_r710_seed1`〜`_seed4`、Round 8のDeep Ensembles検証で作成済み)をStage C検証（R7-11+R7-12ゲート: `conf_threshold=0.0, input_dist_threshold_mm=10, prune_dist_threshold_mm=12`, `refine_rounds=3`）にかけ直した。
- patched(トポロジーパッチ適用後データ)でseed=1〜4を追加学習（`checkpoints_r710_patched_seed1`〜`_seed4`、既存の`checkpoints_r710_patched`=seed0と合わせて5seed）し、同一ゲート設定でStage C検証した。
- `validate_refine.py`に`--data`/`--gt`オプションを追加し、同一スクリプトで両条件を切り替えて評価できるようにした（後方互換: デフォルトは従来のハードコードパス）。

### 結果（n=5 seed、refine_rounds=3、production gate設定）

| 指標 | unpatched mean±std | patched mean±std | 差分 | t統計量(df=8, 対応なしt検定) |
|---|---|---|---|---|
| Recon→GT mean | 8.47±0.31mm | 8.41±0.23mm | -0.06mm | -0.34（有意差なし） |
| Recon→GT max | 47.09±0.57mm | 47.34±1.46mm | +0.25mm | 0.36（有意差なし） |
| GT→Recon mean | 1.65±0.11mm | 1.76±0.05mm | +0.11mm | 2.00（境界線、p≈0.08で非有意） |
| GT→Recon max | 15.63±2.39mm | 16.52±2.70mm | +0.89mm | 0.55（有意差なし） |

訓練収束の副次的観察: patched側5seedすべてのnear_mae(0.0739〜0.0837mm)が元のunpatched seed0(0.0056mm)より一貫して悪く、かつpatched内では狭い範囲に再現性高く収まっていた。ただしunpatched seed1-4のnear_mae値は当時記録しておらず比較不能であり、何より本観察はStage Cの最終復元精度（上表）には伝播していないため、TD-16の結論には影響しない。

### 確定設計判断（R9-01）

| 判断ID | 内容 | 根拠 | 状態 |
|--------|------|------|------|
| **R9-01** | 「Stage Aのray-castノイズ(GT境界トポロジー断片化)がTD-16のゴーストメッシュの主要因」という仮説を**棄却**する。トポロジーパッチ(`patch_isolated_invalid_vertices()`)はGTのboundary loop数を66%削減したにもかかわらず、Stage Cの最終復元精度(Recon→GT/GT→Recon)に統計的に有意な改善をもたらさなかった(4指標中3指標で通常のseed分散の範囲内、1指標で境界線上の弱い悪化シグナルのみ、n=5×2条件)。TD-16の支配的要因は引き続き§14で特定したcut locus(UDFの境界における非微分可能性)であるとの理解を維持する。 | 5seed×2条件の対応なしt検定。最大効果量を示したGT→Recon meanでもt=2.00(df=8)で有意水準0.05に届かず(p≈0.08)。 | **確定（棄却）** |

### 今後の方針
トポロジーパッチ自体は副作用がない(Stage Cの精度を統計的に悪化させない)ため実装は維持してよいが、TD-16解決のための主要な投資先としては不適切と判断する。次の一手はユーザー判断待ち: (a) cut locusへの直接対応(R7-09 Eikonal正則化の重み増加、MeshUDF/CAP-UDF等の導入)、(b) ソリッドSDFへの切り替え検討(2026-06-18のユーザー質問で議論した通り、薄殻表現の既知の弱点とのトレードオフがあり現時点では非推奨)、(c) 本件は保留し他の優先課題へ。

### 実装ファイル更新
- `biw_poc/src/preprocess/midsurface_sampler.py`: `patch_isolated_invalid_vertices()`, `_build_vertex_adjacency()`(Stage 1で追加、デフォルト有効)
- `biw_poc/src/model/validate_refine.py`: `--data`/`--gt` CLI引数追加（クロスコンディション比較用）
- チェックポイント: `checkpoints_r710_patched_seed1`〜`_seed4`(新規、patchedデータseed1-4)

---

## 17. 「虫食い」(GT→Recon被覆穴) 根本原因特定と解決 確定事項（Round 10）

### 記録日: 2026-06-18

### MeshUDF調査（参考、不採用で確定）
TD-16（境界cut locusでのオーバーシュート）への対策候補としてMeshUDF（ECCV 2022, cvlab-epfl）を調査した。アルゴリズムは勾配内積による疑似符号割当（`si = sgn(g₁·gᵢ) × uᵢ`）+ BFS/投票伝播 + 標準Marching Cubesルックアップテーブルで構成される。**不採用と結論**：MeshUDFは「適切に較正されたフィールドからメッシュトポロジーを抽出する」層の問題を解くものであり、TD-16の真因（境界外挿によるOOD/ghost UDF較正の崩れ）には対処しない。さらに原論文自身が、非ウォータータイト・開境界サーフェス（本プロジェクトの中立面そのもの）に対するMarching Cubesの不適合を課題として明記している。**代替候補としてDUDF**（CVPR 2024, Differentiable Unsigned Distance Fields with Hyperbolic Scaling）を提案——cut locus非微分可能性をメッシュ抽出時ではなくネットワーク学習時に修正する点で、既存のR7-09 Eikonal正則化と同じ層で対処できるため筋が良い。**未着手・未調査続行**：ユーザーが優先度を「虫食い(穴)問題」に振り替えたため、DUDFの実装検討は保留中。再開はユーザーの明示的指示があった場合のみ。

### 「虫食い」問題の症状定義
ユーザーがスクリーンショット（`poc_result_viewer.py`のGT-recon距離ヒートマップ）で確認: TD-16（境界付近の突起・オーバーシュート）とは別に、**部品表面全体に散らばる小さな被覆穴（本来メッシュがあるべき箇所に何もない状態）** が多数存在する。GT→Recon方向の問題であり、Recon→GT方向のTD-16とは原因も対策も独立。

### 確定設計判断テーブル（R7-13）

| 判断ID | 内容 | 根拠/経緯 |
|--------|------|----------|
| **R7-13** | 「虫食い」診断には**頂点間最近傍距離ではなく真の point-to-surface 距離**（`trimesh.proximity.closest_point()`を密かつ均一にリサンプルしたGT表面に対して使用）を用いる。さらに `refine_rounds`（メッシュ密度、R7-08のsubdivide-and-reproject回数）を**R7-12のprune閾値と並んで穴の支配的要因**として扱う。本番設定を `prune_dist_threshold_mm: 12→20`、`refine_rounds: 3→4` に変更（**ユーザー承認済み**、2026-06-18「採用する(推奨)」）。 | 当初仮説（R7-12の「3頂点中1つでも閾値超過なら三角形ごと削除」ルールが穴の唯一原因）は、頂点間距離から真のpoint-to-surface距離に測定方法を切り替え、かつpruneを完全無効化したテストを行った結果、部分的にしか正しくないと判明した（prune=None時でも真の穴>1mmが49.4%残存）。3-way trade-offを確認: prune閾値を締めると穴は増えるが突起は減る／緩めると穴は減るが突起(TD-16)が再爆発する（prune=None: 突起 mean=27.28mm, max=188.07mm）／refine_roundsを上げると穴は大幅に減り突起コストはごく僅か。(prune=20, refine_rounds=4)のPareto最適点で: 真の穴>5mm率 5.3%→1.1%、穴距離mean 1.43→0.89mm、突起mean 8.94→9.53mm（ほぼ不変）、突起max 46.95→46.42mm（横ばい〜微改善）。コスト: メッシュ頂点数が約6倍に増加（約20万→約123万頂点、約38.2万→約243万三角形）。 |

### 残存課題（許容済み、TDとしては未登録）
fix適用後も真の穴>5mm率は1.1%（ゼロではない）。これ以上の改善には不釣り合いなメッシュサイズ増大が必要（refine_rounds=5は実用的でない: 300万頂点超）と判断し、現状の1.1%を許容残差として受け入れる。TD-16同様、根治（境界cut locus由来の外挿）にはDUDF/MeshUDF系の対策が必要になる可能性があるが、これはRound 10時点では未着手。

### R7-13 追補: CLIデフォルト不整合の修正（同日、ユーザー承認済み）
実装直後の確認で、`reconstruct.py`の`main()`に未修正の既存不整合が見つかった: CLIデフォルトが`conf_threshold=0.5`・`input_dist_threshold_mm=None`(無効)のままになっており、R7-11〜R7-13で検証してきた本番想定値（`conf_threshold=0.0`・`input_dist_threshold_mm=10mm`）と一致していなかった。つまりこれらのデフォルトは一度も検証済みパイプラインに対して実走していなかった。ユーザー確認（「この不整合を修正する(推奨)」を選択）の上、`conf_threshold`のデフォルトを0.5→0.0に、`input_dist_threshold_mm`のデフォルトをNone→10.0mmに修正。`conf_threshold=0.0`とする根拠はRound 8のDeep Ensembles検証で学習済み確信度ヘッド単体のキャリブレーションが信頼できないと判明したため、OOD抑制は決定論的な`input_dist_threshold_mm`/`prune_dist_threshold_mm`ゲートに委譲する設計とした。

### 実装ファイル更新
- `biw_poc/src/model/diagnose_holes.py`: 新規追加。GT→Recon方向の被覆穴診断専用ツール（`diagnose_outliers.py`のRecon→GT方向と対をなす）。d_recon/d_input/d_cand/d_bndとモデル自身のpred_conf/pred_udfをワースト頂点ごとに報告する。
- `biw_poc/src/model/reconstruct.py`: `main()`のargparseデフォルトを `--refine-rounds` 0→4、`--prune-dist-threshold-mm` None→20.0、`--conf-threshold` 0.5→0.0、`--input-dist-threshold-mm` None→10.0 に変更（R7-13 + 追補）。
- `biw_poc/configs/model.yaml`: `reconstruct.refine_rounds` 3→4、新規キー `reconstruct.prune_dist_threshold_mm: 20.0`・`reconstruct.conf_threshold: 0.0`・`reconstruct.input_dist_threshold_mm: 10.0` 追加、R7-13判断根拠＋追補のコメントブロック追加。

---

## 18. R7-16 Charbonnier平滑化損失 実験結果（Round 11）

### 記録日: 2026-06-21
### 背景
ユーザーから「中立面の境界cut locus問題を、後処理ではなく損失関数のゼロ残差近傍の不連続性を均すことで緩和できないか」という提案（Phase 2a: GT/ターゲット側へのhyperbolic近似は cut locus を解消しないことを証明済み・却下→Phase 2b: 代わりにLOSS LANDSCAPE側にCharbonnier/pseudo-Huber平滑化を入れる案）を受けて実施した一回限りの検証実験。

### 実装内容（R7-16、`biw_poc/src/model/train.py`）
`weighted_losses()` の(クランプ後の)UDF残差L1計算をオプションでCharbonnier形に置換する新CLIフラグ `--loss-eps-mm`（既定0.0=元のL1損失とビット単位で同一）を追加。
```
udf_err = sqrt((pred_c - gt_c)^2 + eps^2) - eps   (eps = --loss-eps-mm, mm)
```
**R7-09のEikonal項とは別物**であることに注意: Eikonal項はネットワークが適合すべき「ターゲット形状」側を変える（cut locus付近でも∥∇UDF∥≈1に寄せる）のに対し、R7-16は target は一切変えず、pred==targetの残差ゼロ近傍での**損失関数の勾配の振る舞い**だけを変える、より保守的な介入。

### 副産物: チェックポイント保存のWindows実行時バグ修正（本実験中に発覧・解決）
学習中、`best.pt`が頻繁に上書きされる（near_mae改善のたびに保存）状況で、Windows Defenderのリアルタイム保護による断続的なファイルロック競合が発生し、`torch.save()`または`os.replace()`が `RuntimeError`/`PermissionError ([WinError 5])` で失敗することを確認（`Get-MpPreference`でリアルタイム保護有効を確認、並行実行が原因という仮説は単独実行でも再現したため棄却）。**解決**: `save_ckpt()` ヘルパーを新設し、(1) 一意な一時ファイル名へ`torch.save()`→(2) `os.replace()`でアトミックリネーム、の両ステップそれぞれに独立したリトライ(既定10回×1秒)を実装。一時ファイルへの書き込みはロック中のファイルを再オープンしないため衝突を回避でき、リネームも「対象パスを一瞬だけ専有する」操作のためリトライで解消する。1500エポックの学習を2回連続でクラッシュなく完走させ修正を確認済み。

### 検証条件と結果
body002（単一部品overfitサニティ、§14と同じ枠組み）、seed=0、データセット同一、1500エポック、`--loss-eps-mm` のみ変更。

| 設定 | best near_mae | overfit判定(閾値0.1mm) |
|---|---|---|
| baseline（L1, eps=0.0） | 0.2127mm | FAIL |
| Charbonnier（eps=0.1mm） | 0.1934mm | FAIL |

`near_mae` は損失の形に依存しない生の `\|pred_udf - gt_udf\|`（near領域）であり、両条件間の公平な比較指標として使用（train.py内の計算ロジックはloss_eps_mmと独立）。

### 結論・確定設計判断（R7-16）

| 判断ID | 内容 | 状態 |
|--------|------|------|
| **R7-16** | Charbonnier平滑化損失（eps=0.1mm）はbaseline（L1）比でnear_maeを約9%改善（0.2127mm→0.1934mm）した。一貫した方向の改善ではあるが、両条件とも既存のoverfitサニティ閾値（0.1mm）をFAILしており、かつ両条件ともRound 7で記録した同一部品(body002)の過去基準値（near_mae=0.0056mm）から大幅に劣化している（0.19〜0.21mm台）。後者の劣化はeikonal正則化(R7-09)・confidence head(R7-10)・トポロジーパッチ(R9-01)等、Round 7以降に追加された学習タスクの複雑化が主因と推測されるが未確認であり、本実験の比較対象（baseline vs Charbonnier）はあくまで同一の現行コードベース上での相対比較として有効。**ユーザー判断によりここで一旦区切る**：R7-16の効果自体は確認できたが、(a) 絶対水準の劣化原因切り分け、(b) Stage C再構成精度への伝播確認、はいずれも未実施・保留。 | **実験完了、評価は保留** |

### 今後の方針（保留、ユーザー判断待ち）
次の一手として提示済みだが未着手の選択肢: (a) near_mae絶対水準の劣化原因（eikonal/conf_bce導入、トポロジーパッチ等）を切り分けてからCharbonnier効果を再評価する、(b) 現状のCharbonnier効果（9%改善）を採用し`reconstruct.py`/`validate_refine.py`でStage C再構成精度（Recon→GT/GT→Recon）まで比較する、(c) 本件は保留し他の優先課題へ。再開はユーザーの明示的指示があった場合のみ。

なお、別途ユーザーから提案されたR7-17（softmin/LogSumExp GT-field平滑化、Stage AのGT構築自体を多分岐softminに置換する案）は、その後ユーザー承認（「この内容をベースに進めましょう」）を得て実装・検証が完了した。詳細は **§19** を参照。

### 実装ファイル更新
- `biw_poc/src/model/train.py`: `weighted_losses()`に`loss_eps_mm`引数追加（R7-16）、`save_ckpt()`新設（チェックポイント保存のtemp-write+atomic-rename化、両ステップ独立リトライ）、`--loss-eps-mm` CLI引数追加
- `biw_poc/experiments/loss_smoothing/`: 本実験専用の隔離ワークスペース（codexの並行作業ファイルとは非衝突）。`ckpt_baseline_l1/`、`ckpt_charbonnier_eps0p1/`、両ログファイル。

---

## 19. R7-17: softmin/LogSumExp GT-field平滑化（Stage A、Round 12）

### 記録日: 2026-06-21
### 背景・動機
§14（TD-16）以降、Recon→GTの境界オーバーシュート（cut locus由来のゴーストメッシュ）に対し、§15（R7-11/R7-12 決定論的ゲート）・§17（R7-13 prune/refine調整）で**復元（Stage C）側の後処理**による緩和を重ねてきた。これらはいずれも「歪んだGT場をどう後から扱うか」という対症療法であり、§14で確定した根本原因（**真のUDFは中立面の境界＝cut locusで数学的に非微分可能**、複数の最近傍枝が競合し勾配が不連続に切り替わる）そのものには手を付けていなかった。

ユーザーから、この不連続性を**GT構築そのものを多分岐の滑らかな近似に置き換えることで発生源で除去する**softmin/LogSumExp定式化が提案され、検討の上で承認された（「この内容をベースに進めましょう」）。R7-16（Charbonnier損失によるloss関数側の平滑化）とは平滑化を適用する層が異なる点に注意: R7-16は「ネットワークの予測誤差に対するペナルティの形」を変えるのに対し、R7-17は「ネットワークが回帰目標として与えられるGT値そのもの」を変える。

### 定式化
$$\tilde u_\tau(x) = -\tau\log\sum_i \exp\!\left(-\frac{f_i(x)}{\tau}\right)$$

数値安定版（実装に採用）:
$$\tilde u_\tau(x) = f_{\min}(x) - \tau\log\sum_i \exp\!\left(-\frac{f_i(x)-f_{\min}(x)}{\tau}\right)$$

勾配は重み付き合成: $\nabla\tilde u_\tau(x)=\sum_i w_i(x)\nabla f_i(x)$、$w_i \propto \exp(-f_i/\tau)$（softmax重み）。$\tau\to0$ で厳密な hard-min に収束する。理論上界: 全$n$分岐が完全に同距離でタイした場合、$\tilde u_\tau = f_{\min} - \tau\ln n$ が最大下方偏差となる。

### 確定設計判断テーブル（R7-17, R7-17追補）

| 判断ID | 内容 | 根拠/経緯 |
|--------|------|----------|
| **R7-17** | `midsurface_sampler.py` の `sample_udf_queries()`・`sample_boundary_shadow_queries()` に、opt-inのsoftmin/LogSumExp GTモードを追加する。デフォルト無効（`softmin_tau_mm=None`）で既存の hard-min パスとビット完全一致を維持し、後方互換性を保つ。有効時は `softmin_tau_mm`（温度パラメータ、mm単位）・`softmin_k_faces`（候補面数、デフォルト8）をCLI/関数引数で指定する。 | ユーザー承認（「この内容をベースに進めましょう」）。§14で特定したcut locus不連続性をGT構築の発生源で直接解消するため。R7-13の「測定は厳密に（point-to-surface距離）、対策は別レイヤーで」の原則を踏襲し、診断・評価ツールは引き続き厳密な非平滑距離で測定する（softminはあくまで学習用GTターゲットの構築のみに使う）。 |
| **R7-17追補（候補生成バグ修正）** | softminの候補枝集合に、centroid-kNNによるK近傍面（`softmin_k_faces`個）に加えて、**`trimesh.proximity.closest_point()`による厳密最近傍枝を必ず追加で含める**（`n_branches = k_faces + 1`）。 | 自己検証中に発見: centroid距離のみによるK近傍面探索は真のpoint-to-triangle距離の不完全な代理であり（特に細長い・歪んだ三角形で）、真の全体最近傍面がcentroid-kNN候補集合から漏れることがある。漏れた場合、候補集合自体の`f_min`が真の全体最小値を超過し、`softmin(x) ≤ true_min(x)`という定式化の根幹を成す数学的保証が破れる（実証: 修正前はNEARカテゴリの約半数で違反、平均+4.55mm・最大+59.8mm）。ユーザー承認（AskUserQuestion「提案通り修正する(推奨)」）を得て、厳密最近傍枝を無条件で候補集合に追加する形で修正。重複枝（厳密最近傍がcentroid-kNN候補にも含まれる場合）はsoftminの多重集合上の和として無害。 |

### 実装詳細
- `_softmin_udf_and_grad(tri_mesh, query_xyz, tau_mm, k_faces, face_kdtree)` を新規関数として追加。候補枝は (1) `trimesh.proximity.closest_point()` による厳密最近傍（正しさの担保、無条件採用）+ (2) `cKDTree(triangles_center)` によるK近傍面（`k_faces`個）の2系統を合成。数値安定softmin（$f_{\min}$を引いてからexp・log）でUDFを計算し、勾配は各枝の単位方向ベクトルをsoftmax重みで合成した後、**単位長に再正規化**して返す（既存の`query_grad`が単位ベクトルである既存データセットの慣例に合わせるため。理論上、真の$\nabla\tilde u_\tau$のノルムは競合枝が近いほど1未満に縮小する「確信度」信号を内包するが、`train.py`の`grad_loss`が`F.cosine_similarity`でスケール不変であるため、現状の学習ではこの縮小情報は利用されない設計になっている — ドキュメントとして関数docstringに明記済み）。
- `sample_udf_queries()` / `sample_boundary_shadow_queries()` は `softmin_tau_mm is None or <= 0` の場合は従来の単一最近傍点 `trimesh.proximity.closest_point()` パスを一切変更せず実行し（ビット完全一致）、それ以外の場合のみ `_softmin_udf_and_grad()` に分岐する。
- `process_body()` は `softmin_tau_mm`・`softmin_k_faces` を受け取り、有効時のみ共有 `face_kdtree` を1回構築して両クエリ関数に使い回す。h5属性（`gt_mode`, `softmin_tau_mm`, `softmin_k_faces`）とmeta JSONの両方に来歴を記録する。
- CLI: `--softmin-tau-mm`（デフォルト`None`=無効）・`--softmin-k-faces`（デフォルト8）を追加。

### 検証結果（body002、`experiments/loss_smoothing/softmin_r717/`）
| 検証項目 | 結果 |
|---|---|
| 中立面メッシュ（.ply）のhard-min版との一致 | md5一致（`be41a8dcba44720060e9367f87dea646`）— GT場の計算のみが変化し、メッシュ形状自体は無影響であることを確認 |
| 修正前: softmin(x) ≤ true_min(x) 保証の違反 | NEARカテゴリで約半数の点が違反（平均+4.55mm、最大+59.8mm）、FARカテゴリでも多数違反（全体4696点） |
| 修正後: 同保証の違反 | グローバル最大の正方向ずれ = 2.5e-5mm（2つの異なるtrimesh最近傍点実装間の浮動小数点誤差レベル、実質ゼロ）。1e-4mm許容で違反ゼロ |
| BOUNDARYカテゴリの最大ブレンド量と理論上界の一致 | 実測 ≈2.197mm、理論値 $\tau\ln(n_{\text{branches}})=1.0\times\ln(9)\approx2.1972$mm とほぼ完全一致（$\tau=1.0$mm、$n_{\text{branches}}=9$）— 実装の数学的正しさを裏付ける独立した根拠 |
| `python -m py_compile` | 修正前後とも clean |

### 状態と今後
データセット生成・数学的検証は完了。**未着手**: (a) softmin版データセットを用いた実際の学習実行とhard-min版との比較（R7-16で確立した「学習前に一旦立ち止まりユーザーに報告する」パターンを踏襲し、再開はユーザーの明示的指示を待つ）、(b) Stage-B評価軸の見直し（near_maeのみに依存しない分布統計＋カバレッジ＋ゴースト異常値チェック、ユーザーから別途要望済み・未着手）。

### 実装ファイル更新
- `biw_poc/src/preprocess/midsurface_sampler.py`: `_softmin_udf_and_grad()`新設、`sample_udf_queries()`/`sample_boundary_shadow_queries()`/`process_body()`にsoftmin引数追加、`--softmin-tau-mm`/`--softmin-k-faces` CLI引数追加、`SOFTMIN_K_FACES_DEFAULT=8`定数追加
- `biw_poc/experiments/loss_smoothing/softmin_r717/`: 本実験専用の隔離ワークスペース（codexの並行作業ファイルとは非衝突）。`body002_filled.stp`のコピー、生成された`body002_filled_midsurface.ply`/`body002_filled_dataset.h5`/`body002_filled_midsurface_meta.json`。

### R7-17追補2: 重複枝バグ + 遠方パララックスバイアスの発見と対策A+B（Round 13）

#### 発見の経緯
ユーザーから学習中の観察として「近似空間上の最小距離が集中し、板金から遠い所に異常値が存在しないこと」という品質要件が示されたため、softmin GT場を詳細診断したところ、3つの独立した問題が見つかった。

1. **重複枝の二重カウントバグ**: R7-17追補で「厳密最近傍枝」を候補集合に強制追加したことの副作用として、その厳密最近傍枝がcentroid-kNN候補集合の上位枝と（ほぼ）同一点を指すケースが高頻度で発生し、softmaxの和の中で実質的に同じ枝を2回カウントしてバイアスを最大 $\tau\ln 2 \approx 0.693$mm 過大評価していた。重複率: 全体55.1%（NEARカテゴリ37.8%／FARカテゴリ45.8%／BOUNDARYカテゴリ91.3%）。
2. **遠方パララックス（候補集中）バイアス**: 重複を除去してもなお、クエリ点が真の表面から離れるほどバイアスが単調に増大する構造的な効果が残った（固定τ・絶対空間での候補探索に内在する限界であり、バグではない）。
3. **表面近傍での物理的に無効な負値出力**: 真の表面距離が0.5mm未満のクエリ点のうち多数が、softmin UDFとして負値（最小-2.17mm）を出力していた。

#### 対策A+B（ユーザー承認: AskUserQuestion「A+B（推奨）: 重複排除 + クリップ半径5mm」）
- **対策A（重複枝排除）**: `_softmin_udf_and_grad()`内で、各クエリ点ごとに距離昇順ソート後、最近点座標が`dedup_tol_mm`（=1e-4mm、新定数`SOFTMIN_DEDUP_TOL_MM`）以内で一致する枝を貪欲法でマージしてからsoftmax重み・log-sum-expを計算する。
- **対策B（クリップ半径）**: 新パラメータ`softmin_clip_radius_mm`（新定数`SOFTMIN_CLIP_RADIUS_MM_DEFAULT=5.0`、ユーザー承認済みデフォルト）。真の最近傍距離がこの半径を超えるクエリ点は無条件に厳密hard-minパスにフォールバックし、softminブレンドを一切適用しない。これにより半径超過領域でのバイアスは構造的にゼロになる。
- 対策C（出力の下限クランプ、負値対策）は今回スコープ外として明示的に見送り。
- `softmin_clip_radius_mm`は`sample_udf_queries()`/`sample_boundary_shadow_queries()`/`process_body()`/CLI（`--softmin-clip-radius-mm`）を通じて配線し、h5属性とmeta JSONの両方に来歴記録した。

#### 検証結果（body002、修正前後比較、`experiments/loss_smoothing/softmin_r717/`）

| 検証項目 | 修正前 | 修正後 |
|---|---|---|
| softmin(x) ≤ true_min(x) 保証の違反 | （R7-17追補時点で既に解消済み） | 違反ゼロを再確認（全14,336クエリ点、最大正方向ずれ2.5e-5mm = 浮動小数点誤差レベル） |
| バイアス 0-1mm bin | 0.7469mm | **0.6001mm** |
| バイアス 1-2mm bin | 0.8416mm | **0.6906mm** |
| バイアス 2-5mm bin | 1.0162mm | **0.8213mm** |
| バイアス 5-10mm bin | 1.4533mm | **0.0000mm** |
| バイアス 10-20mm bin | 1.7658mm | **0.0000mm** |
| バイアス 20-50mm bin | 1.9119mm | **0.0000mm** |
| バイアス 50mm+ bin | 1.6831mm | **0.0000mm** |
| 負値出力率（true_dist<0.5mm帯） | 60.5% | 56.6%（残存課題、対策C未着手） |
| 負値出力 平均（同帯） | -0.5324mm | -0.3757mm |
| 負値出力 最小値（同帯） | -2.1709mm | -2.0132mm |
| `python -m py_compile` | — | clean |

**結論**: 対策Bはクリップ半径5mmを超える領域で設計通りバイアスを完全にゼロ化し、ユーザーが当初提起した「板金から遠い所に異常値が存在しないこと」という要件を構造的に満たした。対策Aは半径内のバイアスを縮小したが（重複排除による有限の改善であり全消去ではない）、表面近傍の負値出力は対策A+Bでは根本対応しておらず60.5%→56.6%の小幅改善に留まる——これは正規のマルチブランチ競合（複数の最近傍候補が拮抗する真の幾何学的状況）に起因し、対策Aが除去した「人為的な重複カウント」とも対策Bが境界とする「遠方パララックス」とも異なる第3の独立した残存課題である。対策C（出力クランプ）または他の手法が必要な場合はユーザー判断待ち。

#### データセット更新
- `body002_filled_dataset.h5`を対策A+B適用済みコードで再生成（`--softmin-tau-mm 1.0 --softmin-k-faces 8 --softmin-clip-radius-mm 5.0 --thickness-hint 1.5188`）。
- 修正前データセットは`body002_filled_dataset_preAB.h5`としてバックアップ保持。
- `body002_filled_midsurface.ply`はGT場計算のみの変更であるため形状自体は不変のはずだが、本ラウンドでは直接の前後diffは未実施（再生成前のply個別バックアップを取得しなかったため）。決定論的パイプラインであり`build_midsurface_mesh()`自体は変更していないため非回帰と判断しているが、未確認の軽微なギャップとして記録。

#### 状態と今後
未着手: (a) 対策A+B適用済みデータセットでの学習実行とhard-min版・対策前softmin版との比較、(b) 負値出力残存課題への対策C（クランプ）の要否判断、(c) Stage-B評価軸の見直し（near_maeのみに依存しない分布統計＋カバレッジ＋ゴースト異常値チェック）。いずれも再開はユーザーの明示的指示を待つ。

#### 実装ファイル更新（Round 13差分）
- `biw_poc/src/preprocess/midsurface_sampler.py`: `_softmin_udf_and_grad()`に`clip_radius_mm`/`dedup_tol_mm`引数追加（対策A: 貪欲マージによる重複枝排除、対策B: クリップ半径超過点は厳密hard-minにフォールバック）。新定数`SOFTMIN_CLIP_RADIUS_MM_DEFAULT=5.0`・`SOFTMIN_DEDUP_TOL_MM=1e-4`追加。`sample_udf_queries()`/`sample_boundary_shadow_queries()`/`process_body()`/CLIに`softmin_clip_radius_mm`（`--softmin-clip-radius-mm`）を配線、h5属性・meta JSONに来歴記録追加。
- `biw_poc/experiments/loss_smoothing/softmin_r717/body002_filled_dataset_preAB.h5`: 新規追加（対策A+B適用前のデータセットバックアップ）。

### R7-17追補3: 損失側での負値GT対策 — 対策D（GTスムーズフロア）+ 対策E（負値GT点の重み低減）実装（Round 14）

#### 経緯
対策A+B（R7-17追補2）適用後も、表面近傍（真の距離<0.5mm）のクエリ点の56.6%が物理的に無効な負値softmin UDFを出力する残存課題が残っていた。ユーザーから「モデルアーキテクチャ側で副作用の少ない形で根絶できないか」という検討依頼があり、調査の結果、`QueryDecoder`の出力ヘッド最終層は既に`softplus()`で正値のみを出力するよう設計済み（R7-04）であることが判明した。つまりネットワークの予測自体は構造的に常に正であり、不整合は「時々負になるGTターゲット」と「常に正の予測」の間にのみ存在する。よって「モデルが負値を予測してしまう」という問題は存在せず、対応すべきは損失計算・学習データの扱い側のみと整理した。

ユーザーへ提示した選択肢（AskUserQuestion）:
- **対策D**: 損失計算時のみGTにsoftplus型スムーズフロアを適用（データセット自体は無加工）
- **対策E**: 負値GT点の損失重みを下げる（値そのものは変更しない）
- **対策D+E併用**
- **対策F**: cut locus根治（DUDF等、大規模・別途）
- 見送り

**ユーザー選択: 「対策D+E併用」** — D・E両方を実装する方針で承認を得た。

#### 設計修正の明示的報告（対策E）
当初ユーザーへの説明では対策Eを「既存のconfidenceヘッド（R7-10）を拡張し、負値GTを低信頼度として重み低下させる」案として提示していたが、実装設計を詰める過程で、**学習途中のネットワーク自身が出力する予測確信度（`pred_conf`）をそのまま同じ学習ステップの損失重みとして使う設計は、確信度ヘッドがまだ収束していない学習初期に循環参照的な不安定さを生みうる**と判断した。そのため実装では、確信度ヘッドの予測値ではなく、**GTデータそのものから決定論的に計算される `gt_udf < 0` のブール判定**を重み係数の根拠とする設計に変更した。確信度ヘッド（R7-10）自体は従来通り独立した学習目標として維持し、対策Eの重み付けとは結合させていない。

#### 実装内容（`biw_poc/src/model/train.py`のみ、対象外ファイルへの変更なし）
- `weighted_losses()`に新規キーワード引数`gt_floor_tau_mm: float = 0.0`（対策D）・`neg_gt_weight: float = 1.0`（対策E）を追加。両方ともデフォルト値で従来のロジックとビット完全一致（`--loss-eps-mm`等これまでのopt-inフラグと同じ後方互換パターンを踏襲）。
- **対策D**: `gt_floor_tau_mm > 0`の場合のみ、R7-03の上限クランプ（`gt_c = torch.minimum(gt_udf, clip)`）の**前**に `gt_for_loss = gt_floor_tau_mm * softplus(gt_udf / gt_floor_tau_mm)` を適用し、`gt_c = torch.minimum(gt_for_loss, clip)`とする。`max(gt_udf, 0)`のようなハードクランプと異なり残差ゼロ近傍で微分可能（キンクなし）。**`near_mae`診断指標は意図的にフロア適用前の生の`gt_udf`を参照し続ける**よう実装（オーバーフィット健全性チェックが常に真の誤差を報告するようにするため）。
- **対策E**: `neg_gt_weight != 1.0`の場合のみ、`gt_udf < 0`のクエリ点について、既存のカテゴリ別重み`w`（near/far/boundary、R7-09）に`neg_gt_weight`を乗算する（`torch.where`で対象点のみ選択的に適用）。`udf_loss`・`grad_loss`双方が共有する重み`w`に作用するため、両損失に同時に効く。
- `main()`に`--gt-floor-tau-mm`（デフォルト0.0=無効）・`--neg-gt-weight`（デフォルト1.0=無効）のCLI引数を追加し、`run_epoch()`内の`weighted_losses(...)`呼び出しへ配線済み。
- モジュール先頭docstringに「R7-17追補3」のセクションを追加し、対策D/Eの設計意図・対策C（見送り済み）との違い・対策Eの設計修正理由を記述。

#### 検証結果（数値サニティチェック、`KMP_DUPLICATE_LIB_OK=TRUE python -c "..."`、N=8の合成テンソルで実施）
| 検証項目 | 結果 |
|---|---|
| デフォルト引数（`gt_floor_tau_mm=0.0, neg_gt_weight=1.0`）が旧来の呼び出しとビット完全一致 | `torch.allclose()`で一致確認（OK） |
| フロア`tau=0.5`適用: 負値GT（例 -1.5, -0.3, -2.0mm）が正値（0.024, 0.219, 0.009mm）に変換 | 設計通り（softplusは漸近的に0へ近づくが厳密な0にはならない） |
| フロア適用後も大きい正値（gt_udf=4.0mm）はほぼ無変化（4.0001mm） | 設計通り（softplusは正の引数で恒等関数に漸近） |
| フロア適用時、`near_mae`が無フロア版と一致（生のgt_udfを参照し続ける） | `torch.isclose()`で確認（OK） |
| `neg_gt_weight=0.2`適用で`udf_loss`/`grad_loss`がデフォルト版から変化 | 確認（OK、負値点の重みが選択的に下がっていることを裏付け） |
| `python -m py_compile train.py` | clean |

#### 状態と今後
実装・回帰防止サニティチェックは完了。**未着手・ユーザー判断待ち**: (a) フロア温度`tau`・down-weight係数の具体的な運用値の選定とユーザー確認、(b) 対策D/E適用版での実際の学習実行（hard-min版・無印softmin版・対策A+B版との比較）、(c) Stage-B評価軸の見直し（near_maeのみに依存しない分布統計＋カバレッジ＋ゴースト異常値チェック）、(d) `HANDOFF_codex.md`の最終化（D+E決定の反映）、(e) `AGENTS.md`/`CLAUDE.md`のセクション番号分岐の整理。いずれも再開はユーザーの明示的指示を待つ。

#### 実装ファイル更新（Round 14差分）
- `biw_poc/src/model/train.py`: `weighted_losses()`に`gt_floor_tau_mm`（対策D）・`neg_gt_weight`（対策E）引数追加、`--gt-floor-tau-mm`・`--neg-gt-weight` CLI引数追加、`run_epoch()`呼び出し配線、モジュールdocstring加筆。

---

## 20. SoftminGuidanceModelへの全面シフトと統合ロードマップ確定（Round 15）

### 記録日: 2026-06-22

### 背景
codex側ドキュメント（`AGENTS.md` §24〜26、Round 17〜19、本日付）と本ドキュメント（CLAUDE.md §18〜19、Round 12〜14）が、`HANDOFF_codex.md` §2で警告されていた通りRound 11以降に内容分岐していたことを確認した。突き合わせの結果、以下が判明した。

- `AGENTS.md` R18-01（Round 18）により、`SoftminGuidanceModel`はレガシーの`VecSetAE`/DCUDF系列とは**独立したチェックポイント・アーキテクチャを持つ「第一モデル」**として正式に位置づけられていた。両トラックはGT哲学が意図的に逆である: SoftminGuidanceModel（Track 1）はsoftmin/LogSumExpを**そのまま公式ガイダンス信号として採用**（R18-02）するのに対し、レガシー`VecSetAE`（Track 2）はR17-02によりsoftminを**診断用補助信号に格下げ**し、公式UDF GTはhard-min幾何距離のままとする、という正反対の設計判断が確定していた。
- `AGENTS.md` §26（Round 19、本日付）により、評価方針も刷新されていた: メッシュ生成/VTK再構成を評価パスから除外しモデル単体で評価する、固定検証分割を使う、ベスト複合指標を`potential MAE + step MAE + 0.5*direction cosine誤差 + 0.5*ambiguity MAE`に変更する、`branch_ambiguity`は認識論的不確実性ではなく幾何学的分岐競合であり真のOOD代理は入力点群距離である（R19-05）、support-gate ON/OFFは常に両方比較しGT被覆度の悪化を伴うpoint-to-GT改善は不合格扱いとする（R19-06）。
- `study/A_幾何処理・メッシュ数学の基礎/12_点群再構成とreconstruct_surface.md`（本日付）により、未解決だった「法線配向異常」（`HANDOFF_codex.md` §0）について新たな直接証拠を得た: 候補点選択→GT間の誤差はVTKの`reconstruct_surface()`を経由する直前まで段階的に改善する（1.56mm→0.68mm）のに対し、VTK通過直後に急激に悪化する（~8.9mm）。再構成済み頂点は自身の入力点群からも平均10.25mm離れており、これはモデルの予測精度とは独立にVTK側が疎な領域で外挿していることの直接証拠である。原因は`max_input_distance_mm`（既定10.0mm）のsupport gateがGT表面の25.9%を候補点から除外していること（被覆ギャップ）に遡れる。`nbr_sz`調整は平滑性と被覆完全性のトレードオフであり根治ではない。

これらを踏まえ、今後の作業優先順位をユーザーに提示し、3点の判断を仰いだ（`AskUserQuestion`、いずれも推奨案を選択）。

### 確定設計判断テーブル（R20-01〜R20-03）

| 判断ID | 内容 | 根拠/経緯 | 状態 |
|--------|------|----------|------|
| **R20-01** | 「法線配向異常」（`HANDOFF_codex.md` §0）への対応は、codexからの回答を待たずに**今すぐ着手**する。VTK側（候補点密度・`nbr_sz`・support-gate見直し）から着手し、codexへの問い合わせ自体は並行して継続する。 | `study/12`が示した直接証拠（VTK通過前後での誤差急増、再構成頂点の入力点群からの平均10.25mm乖離）により、原因はモデルの予測精度ではなくVTK再構成側にある可能性が高いと判断できる状態になった。codexの回答を待つブロッキング理由が薄れた。 | **確定** |
| **R20-02** | レガシートラック（Track 2: `VecSetAE`/DCUDF、TD-16、R7-16 Charbonnier損失、R7-17のsoftmin-A+B版データセットでの未実施学習実行）は**凍結し記録のみ残す**。SoftminGuidanceModelが正式な「第一モデル」（R18-01）となった以上、レガシートラックへの追加投資は行わない。 | レガシートラックは§14〜19で詳細に検証されたが、TD-16（境界cut locus外挿）の根治は依然未解決のまま。第一モデルがTrack 1に確定した以上、限られた工数をTrack 1に集中させるのが合理的。再開が必要になった場合のみユーザー判断で復帰する。 | **確定（凍結）** |
| **R20-03** | 「① branch_ambiguityヘッド除去アブレーション」（CLAUDE.md側で検討していたマルチタスク干渉切り分け）と、`AGENTS.md` Round 19が定義した「新複合指標＋固定検証分割での再学習」（R19の次実験順序①）は、**別々の実験ではなく単一の再学習ジョブに統合**し、同一スイープ内で「branch_ambiguityヘッドあり/なし」を比較する。 | 両者は「同一条件下での再学習＋比較」という点で実験基盤を共有でき、別々に2回学習を回すのは工数の無駄。固定検証分割・新複合指標という土台の上でheadあり/なしを比較すれば、マルチタスク干渉仮説の検証とR19の新評価軸への移行を同時に達成できる。 | **確定** |

### 優先順位ロードマップ（Tier付け）

| Tier | 項目 | 内容 | 対応するAGENTS.md/HANDOFF項目 |
|------|------|------|------------------------------|
| **Tier 0**（着手中） | 法線配向異常のVTK側対応 | 候補点密度の引き上げ・`nbr_sz`調整・support-gate（`max_input_distance_mm`）見直しによる被覆ギャップ縮小 | `HANDOFF_codex.md` §0、`study/12` |
| **Tier 1** | ①アブレーション統合再学習 | 固定検証分割＋新複合指標（R19-01〜R19-04）での再学習を、branch_ambiguityヘッドあり/なしの2系統で同時実施 | `AGENTS.md` R19 次実験順序① |
| **Tier 2** | Stage-A退行の根本確認＋難例サンプリング | Stage A occlusion修正後にnear-field精度が退行した件（`ab`との3-way比較で確認済み）の根本原因確認と難例サンプリングでの緩和 | `AGENTS.md` R19 次実験順序③ |
| **Tier 3** | 複数部品ホールドアウトデータセット | 単一部品（body002）evidenceからの脱却。`AGENTS.md` R19-09の信頼性要件（30部品以上・5ファミリー以上・複数シード）に向けた第一歩 | `AGENTS.md` R19 次実験順序④ |
| **保留** | HANDOFF §1（GT負値フロアのレイヤー分岐問題） | train.py側のみで対応すべきかStage A側まで踏み込むべきかの設計判断。codexの意見待ちで保留継続 | `HANDOFF_codex.md` §1 |
| **凍結（記録のみ）** | レガシートラック（Track 2） | TD-16、R7-16 Charbonnier、R7-17 softmin-A+B版での学習実行 | R20-02 |
| **長期** | `archi.md` Phase R1以降 | 拘束点条件付き再帰型UDFメッシュ生成アーキテクチャ（目次のみ確認済み、本文は未読） | `archi.md` |

### 今後の進め方
Tier 0の実装に着手する。ただし`reconstruct_softmin_guidance.py`はcodex主導実装であり（`HANDOFF_codex.md` §4で編集対象外と明記済み）、まずは非破壊の診断（VTK投入直前の候補点群密度・GT被覆をダンプして測定）から始め、具体的な修正案が固まった時点で、これまでの全変更と同様に**デフォルト値を変えないopt-inのCLI引数**として実装し、codexとの並行作業を妨げない形を維持する。

---

## 21. Tier 0診断結果：法線配向異常の原因切り分け（Round 16）

### 記録日: 2026-06-22
### 前提
§20 R20-01（「今すぐ着手」）に基づき、`HANDOFF_codex.md` §0 / `study/12` §8が未解決のまま残していた法線配向異常
（再構成メッシュの法線がGT法線と無相関、`cos`平均0.010〜0.015、60°超ズレが面の75〜79%、TD-16のcut locus
パターン（境界距離との相関係数≈1.0）とは異なり境界距離との相関係数がほぼゼロ）について、原因仮説
「(a) モデルの方向予測自体が誤っている」と「(b) 投影後の点群密度がVTKの局所法線推定に不適切」のどちらが
支配的かを、`checkpoints_softmin_r717_p1only/best.pt` + 現行データセット（post-CoverageFix、softmin A+B適用後）
を用いた直接介入実験で切り分けた。新規スクリプト`diagnose_candidate_coverage.py`（非破壊、`reconstruct_softmin_guidance.py`
の`reconstruct_coarse_surface()`をVTK呼び出し直前まで再現するのみで同ファイルは未編集）を作成し測定した。

### 主要な発見（仮説(b)の単純な「密度不足」説を実験的に棄却）

| 設定 | 候補点数 | 候補点→GT距離(mean) | GT被覆ギャップ（10mm超の割合） | 再構成メッシュ→GT距離(mean) | 法線cos(mean) | 60°超ズレ率 |
|---|---:|---:|---:|---:|---:|---:|
| 既定 (`grid_res=48, candidate_factor=3.0, nbr_sz=20`) | 6,912 | **0.71mm** | 2.6% | 9.08mm | 0.010 | 79.2% |
| 高密度化 (`grid_res=64, candidate_factor=6.0`) | 24,576 (3.5倍) | 0.72mm（**ほぼ不変**） | 0.5%（**5倍改善**） | 8.61mm（**ほぼ不変**） | 0.002（**改善せず**） | 79.4%（**改善せず**） |
| `nbr_sz=8`（既定の局所近傍を半分以下に縮小） | 6,912 | — | — | 9.29mm（**ほぼ不変**） | 0.005（**改善せず**） | 78.3%（連結成分は3→48に断片化、**悪化**） |

候補点群はVTK投入直前の時点で既にGTから平均0.71mmという高精度（`study/12` §3.1の「3回投影後: 0.68mm」と
ほぼ一致し、独立した再測定として整合性を確認）。にもかかわらず、候補点密度を3.5倍に増やしGT被覆ギャップを
5倍改善させても、再構成メッシュの精度・法線整合性はほぼ変化しなかった。これは`study/12` §3.2が単独の主要因と
した「被覆ギャップ→疎領域外挿」という説明では、本異常（法線配向異常）の大部分を説明できないことを示す
決定的な反証である（§3節が扱う「疎な点群からの外挿」失敗モードとは独立した、別の機構が支配的）。

### 符号分布チェック：Hoppe型陰関数のsign伝播がこの開いた中立面形状で機能していない可能性

再構成メッシュ頂点を最近傍の入力点群法線方向に射影した変位（along-normal成分）の符号分布を測定したところ：
- `+normal`方向への変位: 49.9%　／　`-normal`方向への変位: 50.1%（ほぼ完全に五分五分）
- 平均値はほぼゼロ（-0.025mm）だが標準偏差は10.72mmと非常に大きい

一方向への系統的なオフセット（例: 板厚分のズレ等）ではなく、**符号がランダムに反転している**パターンである。
これは`study/12` §2.1が指摘した「`reconstruct_surface()`は各点周りの局所接平面の法線符号を隣接平面間で
伝播的に揃えるという、SDF的な（向き付けを要求する）内部表現を構成する」という設計が、本プロジェクトの
中立面点群（**真の「内側/外側」を持たない、厚みゼロの開いた両面シート**）に対しては、符号伝播の基準点が
構造的に存在しないため、向き付けが事実上ランダムに決定されてしまっている可能性を強く示唆する。

### 確定設計判断（R21-01）

| 判断ID | 内容 | 状態 |
|--------|------|------|
| **R21-01** | 法線配向異常の支配的原因を「(b) 投影後の点群密度不足」という単純な密度不足仮説から**棄却**し、「`reconstruct_surface()`（Hoppe型符号付き陰関数再構成）が、内側/外側を持たない開いた両面の中立面ジオメトリに対して、局所平面の符号（向き）を構造的に安定して決定できていない」という、より具体的な仮説に更新する。候補点の密度・精度を上げる対症療法（`nbr_sz`調整、`candidate_factor`増加等）では解決しないことを実験的に確認済み。根治には、VTKのsign伝播に依存せず、Stage Aで既に得られている向き付き法線（`cloud.normals`）を再構成アルゴリズム側に直接注入する設計（例: 各候補点に最近傍入力点の法線を割り当てて`reconstruct_surface()`の内部推定をバイパスする、または法線整合が不要な別の再構成手法に置き換える）が必要になる可能性が高い。 | **確定（仮説更新）、具体的修正案は未実装** |

### 状態と今後
Tier 0の診断フェーズは完了。次（タスク#66）は、R21-01の更新仮説に基づいた具体的な修正案（候補点に入力点群の
法線を直接割り当てる、または`reconstruct_surface()`以外の再構成手法を検討する等）の設計に進む。
`reconstruct_softmin_guidance.py`はcodex主導実装であるため、修正案を固めた段階でcodexとの調整を行う
（既存の`HANDOFF_codex.md`運用方針を継続）。

### 実装ファイル更新
- `biw_poc/src/model/diagnose_candidate_coverage.py`: 新規追加。VTK投入直前の候補点群密度・GT被覆ギャップを
  非破壊で測定する診断スクリプト（`reconstruct_softmin_guidance.py`は未編集）。

---

## 22. 板厚条件付け(thickness conditioning)の実装と効果検証（Round 17）

### 記録日: 2026-06-22
### 背景
ユーザーから提起された問い「拘束点・板厚・断面二次モーメント・面積等の部品データはモデル入力に含まれているか、含めることで精度を高められるか」（§20以前、未要約部分）に対し、AskUserQuestionで提示した4選択肢から**「まず板厚をモデル入力に配線(推奨)」**が選ばれ、実装・学習・評価を完了した。

### 実装内容
- `SoftminGuidanceModel`（`softmin_guidance.py`）に `use_thickness_in_encoder` フラグを追加。有効時、`thickness_norm`（`thickness_mm / scale_mm`の正規化板厚スカラー）を `thickness_enc`（2層MLP、最終層ゼロ初期化）でトークン次元へ埋め込み、`PointTokenizer`出力の全トークンに加算する。
- **後方互換パターン**: `SoftminGuidanceModel.__init__`のデフォルトは`use_thickness_in_encoder=False`（旧チェックポイントのメタデータにキーが無くても復元可能）。一方`TrainConfig.use_thickness_in_encoder`のデフォルトは`True`（新規学習はデフォルトでopt-in）。CLIには無効化用の`--no-thickness-in-encoder`フラグを追加（プロジェクトの`--is-cut`/`--no-cut`規約に倣う）。
- **ゼロ初期化no-opパターン**: `thickness_enc`最終層の重み・バイアスをゼロ初期化することで、学習開始時点では板厚条件付けが厳密な恒等写像（no-op）として振る舞い、学習が進むにつれてのみ寄与が立ち上がる設計とした。
- テスト5件追加（`test_softmin_guidance_model.py`）: 無効時の無視確認、有効時の検証エラー、shape/勾配/逆伝播、ゼロ初期化no-op性、チェックポイントメタデータ反映。既存8件と合わせ計13件全PASS。`test_train_softmin_guidance.py`の`evaluate_epoch`系テストで`thickness_mm`バッチフィールド欠如によるリグレッション1件を検出・修正。4ファイル計40テスト全PASS確認済み。

### 検証方法
`checkpoints_softmin_r717_p1only`（既存baseline、板厚条件付けなし）と全く同一のハイパーパラメータ（epochs=2000, batch_size=2, lr=0.0002, seed=0, 全loss重み・clip値・アーキテクチャ寸法が同一）で、板厚条件付けのみを有効化（新デフォルト）して`checkpoints_softmin_r717_p1only_thickness`を再学習。学習完了後（best epoch=1875、near composite=0.56067）、`analyze_softmin_guidance_field.py`に`body002_r717_p1only.json`の`configuration`ブロックと完全に同一の解析パラメータ（grid_res=48, candidate_count=4096, projection_iterations=5, bbox_pad_fraction=0.15, input_dist_threshold_mm=10.0等）を与えて17ゲート品質レポートを再生成し、baseline（`body002_r717_p1only.md`）と直接比較した。

### 結果（`body002_r717_p1only.md` → `body002_r717_p1only_thickness.md`）

| 指標 | baseline（板厚なし） | thickness条件付け | 変化 |
|---|---:|---:|---|
| 品質ゲート合格数 | 3/17 | **4/17** | +1（`near_clear_direction_p95_deg`が50.11°→34.71°でFAIL→PASSに転換） |
| near potential MAE | 0.2485mm | **0.1863mm** | -25% |
| near potential RMSE | 0.4550mm | 0.3773mm | 改善 |
| near step MAE | 0.2589mm | **0.2264mm** | -13% |
| near direction angle mean | 22.25° | 20.64° | 改善 |
| near ambiguity MAE | 0.2281 | 0.2141 | 改善 |
| near ambiguity Spearman | 0.4527 | 0.4971 | 改善（閾値0.70まで依然遠い） |
| near_clear potential MAE | 0.1719mm | **0.1229mm** | -28% |
| dense grid ghost fraction（potential/step） | 0.0221/0.0250 | 0.0174/0.0183 | 改善 |
| candidate ranking, without_ood_gate, raw_signed, initial point→GT mean | 8.32mm | **4.92mm** | 大幅改善 |
| autograd cos(-∇potential, predicted direction) | 0.7227 | 0.7353 | 改善 |
| coverage_p95_mm（ゲート） | 9.3987mm | 9.6173mm | 微悪化（誤差範囲） |
| projection_worsened_fraction（ゲート） | 0.0623 | 0.0664 | 微悪化（誤差範囲） |
| input_cloud_coverage系2ゲート | PASS（不変） | PASS（不変） | 不変（入力点群由来でモデル非依存のため当然） |

ほぼ全ての near/near_clear カテゴリ精度・方向予測・曖昧度較正・候補ランキング品質・autograd整合性が一貫して改善し、1ゲートがFAIL→PASSに転換した。悪化したのは`coverage_p95_mm`と`projection_worsened_fraction`の2ゲートのみで、いずれも微小（既存の0.0623→0.0664等）。

### 確定設計判断（R22-01）

| 判断ID | 内容 | 状態 |
|--------|------|------|
| **R22-01** | 板厚をエンコーダ入力として条件付けすることは、全く同一の学習条件下でnear系カテゴリの精度・方向予測・曖昧度較正・候補ランキング品質を広範かつ一貫して改善する、有効な追加入力であることを実証した。ただし合格ゲート数は3/17→4/17に留まり、`not_ready`判定は継続する。ユーザーが当初提起した仮説（部品メタデータを入力/損失に使うことで精度向上が可能か）に対する最初の実証的回答として、「単独要因では精度を大きく押し上げないが、悪化させる方向のトレードオフも実質的に生まない、安全な改善」と結論する。 | **確定** |

### 状態と今後
板厚条件付けは本番採用可能な改善として確定（新規学習のデフォルトでopt-in済み）。ユーザーが同時に提起していた他の部品メタデータ（拘束点・断面二次モーメント・面積）の入力/損失への組み込みは未着手。Tier 1/Tier 2/Tier 3（§20参照）の優先順位ロードマップには未統合であり、次の一手はユーザー判断待ち。

### 実装ファイル更新
- `biw_poc/src/model/softmin_guidance.py`: `use_thickness_in_encoder`フラグ、`thickness_enc`サブモジュール（ゼロ初期化）、`encode()`/`forward()`配線、`checkpoint_metadata()`の`input_contract`反映。
- `biw_poc/src/model/train_softmin_guidance.py`: `TrainConfig.use_thickness_in_encoder`（デフォルトTrue）、`_thickness_norm()`ヘルパー、`--no-thickness-in-encoder` CLIフラグ。
- `biw_poc/src/model/reconstruct_softmin_guidance.py`: `_ARCHITECTURE_KEYS`に追加、`encode_input()`が`thickness_norm`を常時算出・配線。
- `biw_poc/tests/test_softmin_guidance_model.py`: 板厚条件付け検証テスト5件追加（計13件）。
- `biw_poc/tests/test_train_softmin_guidance.py`: `EvalContractModel.forward`/バッチfixtureの`thickness_mm`欠如リグレッション修正。
- `checkpoints_softmin_r717_p1only_thickness/best.pt`: 新規学習済みチェックポイント（best epoch=1875）。
- `biw_poc/reports/softmin_guidance/body002_r717_p1only_thickness.{md,json,npz,png}`: 新規分析レポート一式。

---

## 重要リンク集

- pycatia: https://github.com/evereux/pycatia
- FreeCAD SheetMetal: https://github.com/shaise/FreeCAD_SheetMetal
- NXOpen Python API: https://docs.plm.automation.siemens.com/tdoc/nx/10/nx_api/
- SolidWorks API (Sheet Metal): https://www.codestack.net/solidworks-api/document/sheet-metal/
- OpenCASCADE Sheet Metal: https://www.opencascade.com/components/sheet-metal-operations/
- Analysis Situs: https://analysissitus.org/
- pythonocc-core: https://github.com/tpaviot/pythonocc-core
- build123d: https://github.com/gumyr/build123d
- CATIA CAA Installation: https://www.3ds.com/support/documentation/resource-library/installation-catia-v5-and-caa-api
