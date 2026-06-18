# CATIA V5 × Python COM — 中立面自動生成 決定版アーキテクチャ
## (1 CATPart = 1 ソリッド方針 / ベストプラクティス集約版)

対象: STEPインポートされた MANIFOLD_SOLID_BREP ソリッドを1つ持つ CATPart。
基準面(スキン面)を検出し、t/2 オフセットの中立面を生成して STP/PLY 出力する。

---

## 1. アーキテクチャ全体図

```mermaid
flowchart TD
    subgraph PhaseA ["Phase A: 接続・対象特定"]
        A1["Dispatch('CATIA.Application')<br>DisplayFileAlerts = False"]
        A2["Documents.Open(catpart)"]
        A3["solid_feat = part.MainBody.Shapes.Item(1)<br>(永続参照の支持フィーチャとして後で使う)"]
        A1 --> A2 --> A3
    end
    subgraph PhaseB ["Phase B: フェイス列挙 (一時参照)"]
        B1["sel.Search('Topology.CGMFace,all')<br>※1パート1ソリッドなので ,all = そのソリッドの全面"]
        B2["ref = sel.Item2(i).Reference ← Valueは使わない<br>dn = ref.DisplayName / area = GetMeasurable(ref).Area"]
        B3["sel.Clear() ← 回収完了後に初めて実行<br>(Clearで一時参照は無効化)"]
        B1 --> B2 --> B3
    end
    subgraph PhaseC ["Phase C: スキン面特定・永続化"]
        C1["面積Top2 = 表裏スキン面<br>(面積比チェックでサニティ確認)"]
        C2["DisplayName を永続形に変換<br>Selection_RSur→RSur<br>WithTemporaryBody→WithPermanentBody"]
        C3["CreateReferenceFromBRepName(name, solid_feat)<br>→ 永続Reference ×2面のみ"]
        C4["t = GetMinimumDistance(表, 裏)<br>実測板厚 (範囲チェック)"]
        C1 --> C2 --> C3 --> C4
    end
    subgraph PhaseD ["Phase D: 中立面生成 (GSD)"]
        D1["形状セット 'MidSurface' を確保"]
        D2["AddNewExtract(永続ref)<br>PropagationType=1 / UpdateObjectで局所更新"]
        D3["AddNewOffset(extract, t/2)"]
        D4{"中立面→反対スキンの<br>実測距離 ≒ t/2 ?"}
        D5["Orientation 反転して再Update"]
        D1 --> D2 --> D3 --> D4
        D4 -- No --> D5 --> D4
    end
    subgraph PhaseE ["Phase E: エクスポート"]
        E1["中立面のみ Copy →<br>新規Part に PasteSpecial('CATPrtResult')"]
        E2["ExportData(xxx_mid.stp, 'stp')"]
        E3["(任意) gmsh/trimesh で PLY 変換"]
        E1 --> E2 --> E3
    end
    PhaseA --> PhaseB --> PhaseC --> PhaseD --> PhaseE
```

---

## 2. ベストプラクティス一覧(実験で確認した落とし穴と対策)

| # | 落とし穴(実験で確認) | 対策(確定) |
|---|---|---|
| 1 | `Search` のクエリ・スコープはロケール/環境依存。日本語UIで `,sel`・日本語クエリは E_FAIL | **`"Topology.CGMFace,all"` 一本に固定**(日本語UIでも動作確認済)。スコープ機能には頼らない |
| 2 | `sel.Item2(i).Value` + `CreateReferenceFromObject` は B-Rep セルに対して**全件失敗** | B-Rep には **`sel.Item2(i).Reference`** を使う。`CreateReferenceFromObject` はツリー上のフィーチャ(Extract等)専用と区別する |
| 3 | 選択由来の Reference は一時参照(`Selection_RSur` / `WithTemporaryBody`)。`sel.Clear()` で無効化、フィーチャ入力に使うと更新で壊れる | 計測は一時参照で済ませ、**フィーチャ入力に使う2面だけ** `CreateReferenceFromBRepName` で永続化。Clear 前に DisplayName/面積を回収 |
| 4 | BRepName の永続化変換はリリース差あり(`MonoFond` トークン等) | 変換候補を複数用意して順に試すフォールバック(①単純置換 ②MonoFond除去 ③Face記述子から標準フラグで再構成) |
| 5 | スキン面の判定: 平面判定は曲面パネルで破綻 | **面積Top2ヒューリスティック** + 面積比チェック + 板厚実測の範囲チェック |
| 6 | 板厚は図面公称値とズレることがある | `GetMinimumDistance` の**実測値**を採用 |
| 7 | `AddNewOffset` の正方向は法線依存で事前に不明 | **生成後に実測検証**(中立面→反対面 ≒ t/2)、外れたら `Orientation` 反転(不可なら負値オフセットで再生成) |
| 8 | `part.Update()` は全フィーチャ更新で遅い・他形状を巻き込む | **`part.UpdateObject(feature)`** で局所更新 |
| 9 | `ExportData` はドキュメント全体を出力(ソリッドが混入) | 中立面のみ **`PasteSpecial("CATPrtResult")`** で新規Partへリンクなしコピーしてから出力 |
| 10 | PLY は CATIA ネイティブ非対応 | STP をハブに **gmsh → trimesh** で後段変換(またはOCC直テッセレーション) |
| 11 | バッチ中のダイアログでハング | `catia.DisplayFileAlerts = False`、処理後 `doc.Close()`、失敗パートは記録してスキップ続行 |

---

## 3. COM 呼び出しシーケンス

```mermaid
sequenceDiagram
    participant Py as Python (pywin32)
    participant Sel as Selection
    participant SPA as SPAWorkbench
    participant Part as Part
    participant HSF as HybridShapeFactory

    Py->>Sel: Search("Topology.CGMFace,all")
    loop i = 1..Count2
        Py->>Sel: Item2(i).Reference
        Sel-->>Py: 一時Reference
        Py->>SPA: GetMeasurable(ref).Area
        Py->>Py: DisplayName と面積を記録
    end
    Py->>Sel: Clear()
    Py->>Py: 面積Top2 → 永続BRepName へ変換
    Py->>Part: CreateReferenceFromBRepName(name, Solid)
    Part-->>Py: 永続Reference (表/裏)
    Py->>SPA: GetMinimumDistance(表, 裏) → t
    Py->>HSF: AddNewExtract(表ref)
    Py->>Part: UpdateObject(extract)
    Py->>HSF: AddNewOffset(extract, t/2)
    Py->>Part: UpdateObject(offset)
    Py->>SPA: 中立面→裏面 距離検証 (≒t/2、NGなら反転)
    Py->>Sel: Copy → 新規Partへ PasteSpecial("CATPrtResult")
    Py->>Py: ExportData(stp) → (gmsh/trimesh → ply)
```

---

## 4. オブジェクト/参照の使い分けマップ

```mermaid
graph TD
    subgraph 一時参照 ["一時参照 (Selection 生存中のみ)"]
        T["sel.Item2(i).Reference<br>Selection_RSur / WithTemporaryBody"]
    end
    subgraph 永続参照 ["永続参照 (フィーチャ入力に使用可)"]
        P["CreateReferenceFromBRepName<br>RSur / WithPermanentBody<br>支持フィーチャ = Solid"]
    end
    subgraph フィーチャ参照 ["ツリー上フィーチャの参照"]
        F["CreateReferenceFromObject(extract)<br>※GSDフィーチャにはこちらでOK"]
    end
    T -->|"用途: 面積計測・DisplayName取得"| U1[全フェイスのスクリーニング]
    P -->|"用途: AddNewExtract 入力 / 距離計測"| U2[スキン面2枚のみ]
    F -->|"用途: AddNewOffset 入力"| U3[Extract→Offset の連鎖]
```

---

## 5. 運用チェックリスト

- [ ] CATIA 起動済み / `DisplayFileAlerts = False`
- [ ] 1 CATPart = 1 ソリッド(`Bodies.Count` と `MainBody.Shapes.Count` を起動時に検証)
- [ ] フェイス回収は `sel.Clear()` の**前**に完了している
- [ ] 永続化は2面のみ(性能のため全面はやらない)
- [ ] 板厚 t が想定レンジ内(例 0.2〜10mm)、面積比 Top2/Top1 ≥ 0.5
- [ ] 中立面の位置検証(≒t/2)が通っている
- [ ] 出力 STP に中立面のみ含まれる(ソリッド混入なし)
- [ ] バッチ時: 失敗パートのログ + continue、最後に成功/失敗サマリ
