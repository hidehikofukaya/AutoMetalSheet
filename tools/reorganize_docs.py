"""Reorganise docs/ into four categories (decision 2026-09-05):
  decisions/  design documents and their decision history
  knowledge/  everything the research established
  learning/   material for the user to learn the machinery
  tips/       operational notes
Merged sources are concatenated verbatim (headings kept, so "KB 21.24"-style
references still resolve by grep) under one file with provenance comments;
retired documents go to docs/_archive/ (git mv, history preserved).
Dry-run by default; --apply performs git mv / rm and writes files.
"""
from __future__ import annotations
import argparse, pathlib, re, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
D = REPO / "docs"
KB = D / "KB"

# new file -> (title, [source files in order])
MERGE = {
    "decisions/bend_design.md": ("曲げ表現の設計(KB 20)", [KB/"20_bend_design.md"]),
    "decisions/face_loops.md": ("面ループ表現(KB 21)— 設計と決定の履歴", [KB/"21_face_loops.md"]),
    "decisions/roadmap.md": ("現在地とロードマップ(KB 22)", [KB/"22_status_roadmap.md"]),
    "decisions/language_integration.md": ("言語モデル統合 PoC 計画(KB 23)", [KB/"23_language_integration.md"]),
    "decisions/staged_generator_design.md": ("段階生成器の設計", [D/"staged_generator_design.md", D/"mesh_route_design.md"]),
    "decisions/requests_partmaker.md": ("データ側(PartMaker/抽出)との依頼・回答の記録", [
        D/"requests"/"2026-08-30_wireframe_extraction.md", D/"requests"/"2026-08-30_wireframe_extraction_REPLY.md",
        D/"requests"/"2026-08-30_wireframe_extraction_REPLY2.md", D/"requests"/"2026-09-01_face_ids.md",
        D/"requests"/"2026-09-01_face_ids_REPLY.md"]),
    "decisions/rejected_alternatives.md": ("棄却・保留した設計案(案A/B/C、反復ループ、次期案)", [
        D/"plan_ab_design.md", D/"plan_b_graph_design.md", D/"plan_c_revised.md",
        D/"loop_architecture_proposal.md", D/"next_architecture_proposals.md"]),
    "knowledge/01_task_and_data.md": ("課題とデータ、一般化(KB 01, 09)", [KB/"01_task_and_data.md", KB/"09_generalisation.md"]),
    "knowledge/02_representations.md": ("表現の変遷(KB 03, 10, 14, 15 + 理論メモ)", [
        KB/"03_representations.md", KB/"10_sidecar_finding.md", KB/"14_delta_tokens.md", KB/"15_micro_structure_design.md",
        D/"stage2_bend_result.md", D/"final_wireframe_theory.md", D/"wireframe_theory_rev_curve_major.md"]),
    "knowledge/03_experiment_ledger.md": ("施策台帳と作業記録(KB 04, 06, 07, 08 + ログ)", [
        KB/"04_experiment_ledger.md", D/"research_log.md", KB/"06_status_and_numbers.md", KB/"07_plan_8h.md",
        KB/"08_night_202608_30.md", D/"overnight_plan_202608.md", D/"overnight_results.md", D/"progress_202608.md",
        D/"session_knowledge_202608.md"]),
    "knowledge/04_measurement_pitfalls.md": ("測定の事故台帳(KB 05)", [KB/"05_measurement_pitfalls.md"]),
    "knowledge/05_laws.md": ("法則集: 条件付け・汎用性・針(KB 11, 12, 19)", [KB/"12_conditioning_rule.md", KB/"11_generality_audit.md", KB/"19_generality_and_needles.md"]),
    "knowledge/06_rationality_eval.md": ("工学的成立性と合理性評価(KB 16, 17)", [KB/"16_engineering_structure.md", KB/"17_rationality.md"]),
    "knowledge/07_teacher_data.md": ("教師データ: 力学×幾何、CATIAエッジ(KB 13, 18)", [KB/"13_mechanics_geometry.md", KB/"18_catia_edges.md"]),
    "knowledge/08_past_architectures.md": ("過去アーキテクチャの結末(KB 02 + 各報告の要約)", [KB/"02_architecture.md", D/"current_architecture.md", D/"retrieval_floor_and_direction.md", D/"frontier_gaps_and_reform.md"]),
    "learning/02_survey.md": ("文献サーベイと提案メモ", [D/"architecture_survey_202608.md", D/"fable5_alternative_generation_paradigms.md",
        D/"constraint_geometric_attention_wireframe_proposal.md", D/"mechanical_attention_data_collection_proposal.md"]),
    "learning/03_scaling_and_capacity.md": ("スケーリングと容量の見積り", [D/"model_scaling_estimate.md", D/"next_architecture_overview.md"]),
    "tips/compute.md": ("計算資源の運用(vast.ai / Kaggle)", [D/"vast_ai_setup.md", D/"kaggle_user_manual.md", D/"kaggle_training_handoff.md"]),
    "tips/workspace.md": ("ワークスペースの整理", [D/"workspace_cleanup_plan.md"]),
    "tips/annotation_tool.md": ("締結点アノテーションツールの設計", [D/"annotation_tool_design.md"]),
}
# retired: summarised in knowledge/08, originals archived
ARCHIVE = [D/f for f in (
    "cae_adaptive_shell_mesh_model.md", "cae_midsurface_autoencoder_experiment_001.md",
    "cae_structured_q20_60_cov_scaf_seed13_report.md", "cae_structured_q20_60_e1000_diagnostic_report.md",
    "cae_structured_scaffold_autoencoder_architecture.md", "cae_tangent_frame_decoder_design.md",
    "cae_typed_cross_attention_lattice_design.md", "fable5_diagnostic_response.md", "fable5_local_diagnostic_report.md",
    "fable5_progress_to_constraint_point_generation.md", "fable5_response_constraint_generation.md",
    "elastic_proposal_review.md", "two_point_generation_model_design.md", "wireframe_flow_ae_plan.md")]
ARCHIVE += [KB/"00_INDEX.md"]

def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, check=True, capture_output=True, text=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
    all_src = [s for _, srcs in MERGE.values() for s in srcs] + ARCHIVE
    missing = [s for s in all_src if not s.exists()]
    if missing:
        sys.exit(f"missing sources: {missing}")
    seen = set()
    for s in all_src:
        if s in seen: sys.exit(f"duplicate source: {s}")
        seen.add(s)
    leftovers = sorted(set(D.rglob("*.md")) - seen - {D/"README.md"})
    print(f"{len(MERGE)} merged files from {sum(len(v[1]) for v in MERGE.values())} sources, {len(ARCHIVE)} archived, leftovers: {[str(p.relative_to(D)) for p in leftovers]}")
    if not a.apply:
        return
    for new, (title, srcs) in MERGE.items():
        out = D / new; out.parent.mkdir(parents=True, exist_ok=True)
        parts = [f"# {title}\n\n> 統合日 2026-09-05。以下は元文書を順に**そのまま**収録(見出し番号は元のまま。`KB 21.24` のような参照はこのファイル内を検索)。\n> 収録元: " + ", ".join(f"`{s.relative_to(D)}`" for s in srcs) + "\n"]
        for s in srcs:
            body = s.read_text(encoding="utf-8")
            parts.append(f"\n\n---\n\n<!-- 元文書: {s.relative_to(D)} -->\n\n" + body)
        out.write_text("".join(parts), encoding="utf-8")
        for s in srcs:
            git("rm", "-q", str(s.relative_to(REPO)))
    arc = D / "_archive"; arc.mkdir(exist_ok=True)
    for s in ARCHIVE:
        git("mv", str(s.relative_to(REPO)), str((arc / s.name).relative_to(REPO)))
    for new in MERGE:
        git("add", str((D / new).relative_to(REPO)))
    # empty dirs
    for d in (KB, D / "requests"):
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    print("applied")

if __name__ == "__main__":
    main()
