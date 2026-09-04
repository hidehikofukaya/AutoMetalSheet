#!/usr/bin/env bash
# Three-stage training on the OCCT teacher (KB 21.20): outline -> 2a -> 2b, then E2E on
# the OCCT holdouts and on the CATIA families as unseen-family probes.
#   nohup bash tools/train_occt.sh > runs/pipeline_occt.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$PWD/cae_mesh_generator/src" WTOK_FACES_MODE=all WTOK_FACES=runs/wtok_synth_g1/face_targets
TRAIN=${TRAIN:-1940}
COMMON="--use-spec --rel-attn --seed 0 --dataset runs/mesh_synth --wtok runs/wtok_synth_g1 --val-list runs/wtok_synth_g1/val_occt_60.json --train-parts $TRAIN --val-parts 50 --dim 256 --layers 8 --heads 8 --cfg-scale 1.0 --probe-parts 16"
echo "=== spec rows: $(python -c "import json;print(len(json.load(open('runs/wtok_synth_g1/spec_vectors.json'))['spec']))")"
echo "=== outline on OCCT $TRAIN parts"
python -u -m cae_mesh_generator.wtok.staged --stage outline_frame $COMMON --output-dir runs/frame_occt --epochs 900 --batch-size 16 --probe-every 30 --max-hours 3.0 > runs/frame_occt.log 2>&1
echo "=== 2a on OCCT $TRAIN parts"
python -u -m cae_mesh_generator.wtok.staged --stage face_set $COMMON --output-dir runs/faceset_occt --epochs 600 --batch-size 16 --probe-every 30 --max-hours 3.0 > runs/faceset_occt.log 2>&1
echo "=== 2b on OCCT $TRAIN parts"
python -u -m cae_mesh_generator.wtok.staged --stage face_ring $COMMON --output-dir runs/facering_occt --epochs 140 --batch-size 64 --probe-every 10 --max-hours 4.0 > runs/facering_occt.log 2>&1
for VL in val_occt11_30.json val_occt12_30.json val_names_100.json val_flange_30.json; do
  echo "=== E2E OCCT stages on $VL"
  python -m cae_mesh_generator.wtok.face_eval --outline runs/frame_occt/best.pt --ckpt2a runs/faceset_occt/best.pt --ckpt2b runs/facering_occt/best.pt --ring-k 3 --ring-despike --val-list "$VL" --out "runs/facering_occt/face_eval_$VL" 2>&1 | sed -n "/val parts/,/picture/p"
done
echo DONE_ALL
