#!/usr/bin/env bash
# Ingest new PartMaker groups registered in build_synthetic.GROUPS into the G1 teacher set.
# Resumable: every step skips parts that already exist.
#   nohup bash tools/ingest_occt.sh > runs/ingest_occt.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD/cae_mesh_generator/src" WTOK_FACES_MODE=all
LIMIT=${LIMIT:-4807}
echo "=== 1 build_synthetic (limit $LIMIT)"
python -m cae_mesh_generator.wtok.build_synthetic --output-dir runs/wtok_synth_g1 --limit "$LIMIT" 2>&1 | grep -vE "Warning|^\s*$" | tail -15
echo "=== 2 copy v4.5 wireframes"
cp runs/wtok_synth_g1/wireframes/o1*.json runs/wtok_synth_v45/wireframes/ && echo "v45 now holds $(ls runs/wtok_synth_v45/wireframes | grep -c '^o1') occt wireframes"
echo "=== 3 mesh npz"
python tools/build_mesh_dataset.py --wtok runs/wtok_synth_g1 --out runs/mesh_synth 2>&1 | tail -2
echo "=== 4 face cache"
python -c "from cae_mesh_generator.wtok import faces; faces.export_cache('runs/wtok_synth_g1/face_targets')" 2>&1 | grep -v Warning | tail -3
echo "=== 5 counts"
echo "parts $(ls runs/wtok_synth_g1/parts | wc -l)  wf45 $(ls runs/wtok_synth_v45/wireframes | wc -l)  npz $(ls runs/mesh_synth/parts | wc -l)  faces $(ls runs/wtok_synth_g1/face_targets | wc -l)"
echo "=== 6 spec table"
python tools/export_spec_vectors.py --wtok runs/wtok_synth_g1 2>&1 | tail -1
echo DONE_ALL
