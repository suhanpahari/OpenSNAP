#!/bin/bash
# Checkpoints + OpenScene-processed data into ../downloads
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); D=$ROOT/downloads; mkdir -p $D/ckpt $D/data
export PYTHONNOUSERSITE=1 HF_HOME=$D/hf
cd $D/ckpt
$ROOT/env/bin/gdown 1xGzyOxUPhXLRO8rdOWX9NOetAdBg9q0w -O SNAP_C.pth
for m in scannet_lseg matterport_openseg; do wget -c https://cvg-data.inf.ethz.ch/openscene/models/$m.pth.tar; done
cd $D/data
for z in scannet_processed/scannet_3d matterport_processed/matterport_3d_160; do
  wget -c https://cvg-data.inf.ethz.ch/openscene/data/$z.zip && unzip -qo $(basename $z).zip; done
$ROOT/env/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-3B-Instruct')"
