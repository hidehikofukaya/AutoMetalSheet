# -*- coding: utf-8 -*-
"""
catia_midsurface.py
板金ソリッド(CATPart)から基準面を探索し、t/2 オフセットで中立面を生成して
STP / PLY に出力する CATIA V5 自動化スクリプト。

前提:
  - Windows 上で CATIA V5 が起動済み、対象の CATPart がアクティブであること
  - pip install pywin32          (必須)
  - pip install gmsh trimesh     (PLY 変換を使う場合のみ)

使い方:
  python catia_midsurface.py --out D:/work/out --ply --mesh-size 3.0
"""

import argparse
import math
import os
import sys

import pythoncom
from win32com.client import Dispatch, VARIANT

# ---------------------------------------------------------------- 定数
AREA_RATIO_MIN = 0.5      # 表裏スキン面の面積比の下限(サニティチェック)
T_RANGE_MM = (0.2, 10.0)  # 板厚として妥当な範囲 [mm]
DIST_TOL_RATIO = 0.10     # 中立面位置の許容誤差(t に対する比率)


def r8_array(n):
    """CATIA の byref double 配列引数用 VARIANT を作る(pywin32 の定石)"""
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8 | pythoncom.VT_BYREF,
                   [0.0] * n)


class CatiaMidSurface:
    def __init__(self):
        self.catia = Dispatch("CATIA.Application")
        self.catia.DisplayFileAlerts = False  # バッチ処理用にダイアログ抑止
        self.doc = self.catia.ActiveDocument
        if self.doc is None or not self.doc.Name.lower().endswith(".catpart"):
            raise RuntimeError("アクティブな CATPart がありません。対象パートを開いてください。")
        self.part = self.doc.Part
        self.spa = self.doc.GetWorkbench("SPAWorkbench")
        self.sel = self.doc.Selection
        print(f"[INFO] 対象: {self.doc.Name}")

    # ------------------------------------------------------------ 1. フェイス列挙
    def collect_faces(self):
        """ソリッドの全フェイスを列挙し、面積・平面情報を付与して返す"""
        sel = self.sel
        sel.Clear()
        # MainBody に限定して検索(失敗したらドキュメント全体)
        try:
            sel.Add(self.part.MainBody)
            sel.Search("Topology.CGMFace,sel")
        except pythoncom.com_error:
            sel.Clear()
            sel.Search("Topology.CGMFace,all")

        faces = []
        for i in range(1, sel.Count2 + 1):
            obj = sel.Item2(i).Value
            ref = self.part.CreateReferenceFromObject(obj)
            meas = self.spa.GetMeasurable(ref)
            info = {"ref": ref, "meas": meas, "area": float(meas.Area),
                    "plane": None}
            # 平面なら原点+2方向ベクトルを取得(曲面は COM エラーになる)
            arr = r8_array(9)
            try:
                meas.GetPlane(arr)
                info["plane"] = list(arr.value)
            except pythoncom.com_error:
                pass
            faces.append(info)
        sel.Clear()

        if len(faces) < 3:
            raise RuntimeError(f"フェイス数が不足しています: {len(faces)}")
        print(f"[INFO] フェイス数: {len(faces)} "
              f"(うち平面 {sum(1 for f in faces if f['plane'])})")
        return faces

    # ------------------------------------------------------------ 2. スキン面ペア探索
    def find_skin_pair(self, faces):
        """面積トップ2を表裏スキン面とみなし、(基準面, 反対面, 板厚t) を返す"""
        faces = sorted(faces, key=lambda f: f["area"], reverse=True)
        f1, f2 = faces[0], faces[1]

        ratio = f2["area"] / f1["area"]
        if ratio < AREA_RATIO_MIN:
            print(f"[WARN] 面積比 {ratio:.2f} が小さい。板金形状でない、"
                  "またはフランジ面を誤検出している可能性があります。")

        t = float(f1["meas"].GetMinimumDistance(f2["ref"]))
        print(f"[INFO] スキン面面積: {f1['area']:.1f} / {f2['area']:.1f} mm^2, "
              f"板厚 t = {t:.3f} mm")

        if not (T_RANGE_MM[0] <= t <= T_RANGE_MM[1]):
            raise RuntimeError(f"計測板厚 t={t:.3f}mm が想定範囲 {T_RANGE_MM} 外です。")

        if f1["plane"]:
            o, d1, d2 = f1["plane"][0:3], f1["plane"][3:6], f1["plane"][6:9]
            n = (d1[1]*d2[2]-d1[2]*d2[1],
                 d1[2]*d2[0]-d1[0]*d2[2],
                 d1[0]*d2[1]-d1[1]*d2[0])
            ln = math.sqrt(sum(c*c for c in n)) or 1.0
            print(f"[INFO] 基準面は平面。法線 = "
                  f"({n[0]/ln:.3f}, {n[1]/ln:.3f}, {n[2]/ln:.3f})")
        else:
            print("[INFO] 基準面は曲面(Extract+Offset は曲面でも動作します)")
        return f1, f2, t

    # ------------------------------------------------------------ 3. 中立面生成
    def build_midsurface(self, base, other, t):
        """基準面を Extract し t/2 オフセット。向きを実測で検証して必要なら反転。"""
        part = self.part
        hsf = part.HybridShapeFactory

        try:
            gs = part.HybridBodies.Item("MidSurface")
        except pythoncom.com_error:
            gs = part.HybridBodies.Add()
            gs.Name = "MidSurface"

        # --- 基準面の抽出(点連続伝播でフェイス全体)
        ext = hsf.AddNewExtract(base["ref"])
        ext.PropagationType = 1
        ext.ComplementaryExtract = False
        ext.IsFederated = False
        gs.AppendHybridShape(ext)
        part.InWorkObject = ext
        part.Update()

        # --- t/2 オフセット = 中立面
        ext_ref = part.CreateReferenceFromObject(ext)
        off = hsf.AddNewOffset(ext_ref, t / 2.0)
        gs.AppendHybridShape(off)
        part.InWorkObject = off
        part.Update()

        # --- 向き検証: 中立面→反対側スキンの最短距離は t/2 のはず
        def dist_to_other():
            m = self.spa.GetMeasurable(part.CreateReferenceFromObject(off))
            return float(m.GetMinimumDistance(other["ref"]))

        d = dist_to_other()
        tol = max(0.01, DIST_TOL_RATIO * t)
        if abs(d - t / 2.0) > tol:
            print(f"[INFO] オフセット向きが外側 (d={d:.3f}mm)。反転します。")
            try:
                off.Orientation = -off.Orientation
            except pythoncom.com_error:
                # Orientation が使えない環境向けフォールバック
                off.Offset.Value = -t / 2.0
            part.Update()
            d = dist_to_other()

        if abs(d - t / 2.0) > tol:
            raise RuntimeError(f"中立面位置の検証に失敗 (d={d:.3f}, 期待 {t/2:.3f})")

        off.Name = f"MidSurface_t{t:.2f}"
        print(f"[OK] 中立面生成完了 (反対面までの距離 {d:.3f} mm ≒ t/2)")
        return off

    # ------------------------------------------------------------ 4. STP 出力
    def export_stp(self, off, out_dir, basename):
        """中立面のみを新規 CATPart へ As Result コピーして STP 出力"""
        os.makedirs(out_dir, exist_ok=True)
        sel = self.sel
        sel.Clear()
        sel.Add(off)
        sel.Copy()

        newdoc = self.catia.Documents.Add("Part")
        s2 = newdoc.Selection
        s2.Clear()
        s2.Add(newdoc.Part)
        s2.PasteSpecial("CATPrtResult")  # リンクなし結果コピー
        newdoc.Part.Update()

        stp_path = os.path.abspath(os.path.join(out_dir, basename + "_mid.stp"))
        newdoc.ExportData(stp_path, "stp")
        newdoc.Close()
        print(f"[OK] STP 出力: {stp_path}")
        return stp_path


# ---------------------------------------------------------------- PLY 変換
def stp_to_ply(stp_path, ply_path, mesh_size=3.0):
    """STP を gmsh で表面メッシュ化し、trimesh 経由で PLY に変換する。
    CATIA は PLY を直接出力できないため、この後処理で対応する。"""
    import gmsh
    import trimesh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(stp_path)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.2)
        gmsh.model.mesh.generate(2)
        tmp_stl = ply_path + ".tmp.stl"
        gmsh.write(tmp_stl)
    finally:
        gmsh.finalize()

    mesh = trimesh.load(tmp_stl)
    mesh.export(ply_path)
    os.remove(tmp_stl)
    print(f"[OK] PLY 出力: {ply_path} "
          f"(頂点 {len(mesh.vertices)}, 面 {len(mesh.faces)})")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="CATIA 板金ソリッド → 中立面抽出")
    ap.add_argument("--out", default=".", help="出力ディレクトリ")
    ap.add_argument("--ply", action="store_true", help="STP に加えて PLY も出力")
    ap.add_argument("--mesh-size", type=float, default=3.0,
                    help="PLY 変換時の最大メッシュサイズ [mm]")
    args = ap.parse_args()

    app = CatiaMidSurface()
    faces = app.collect_faces()
    base, other, t = app.find_skin_pair(faces)
    off = app.build_midsurface(base, other, t)

    basename = os.path.splitext(app.doc.Name)[0]
    stp_path = app.export_stp(off, args.out, basename)

    if args.ply:
        ply_path = os.path.splitext(stp_path)[0] + ".ply"
        stp_to_ply(stp_path, ply_path, mesh_size=args.mesh_size)

    print("[DONE] 完了")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
