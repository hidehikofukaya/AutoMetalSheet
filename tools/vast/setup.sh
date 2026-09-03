#!/usr/bin/env bash
# One-time setup on a fresh vast.ai instance (pytorch template image, SSH in).
#   bash setup.sh            # expects wtok_code.zip + wtok_synth_data.zip in /workspace/upload
set -euo pipefail
WS=/workspace
mkdir -p $WS/code $WS/data $WS/runs $WS/upload
cd $WS
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip install -q numpy scipy matplotlib
unzip -qo $WS/upload/wtok_code.zip -d $WS/code
unzip -qo $WS/upload/wtok_synth_data.zip -d $WS/data
echo "parts: $(ls $WS/data/parts | wc -l)  face_targets: $(ls $WS/data/face_targets 2>/dev/null | wc -l)"
ls $WS/data/*.json
nproc; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "setup done. run:  cd $WS/code && nohup python tools/vast/run_arms.py $WS/upload/sweep.json --data $WS/data --out $WS/runs --workers 8 --parallel 4 > $WS/runs/arms.log 2>&1 &"
