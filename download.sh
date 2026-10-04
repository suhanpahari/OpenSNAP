#!/bin/bash
# Checkpoints + OpenScene-processed data into ../downloads
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); D=$ROOT/downloads; mkdir -p $D/ckpt $D/data
export PYTHONNOUSERSITE=1 HF_HOME=$D/hf
cd $D/ckpt
$ROOT/env/bin/gdown 1xGzyOxUPhXLRO8rdOWX9NOetAdBg9q0w -O SNAP_C.pth
for m in scannet_lseg matterport_openseg; do wget -c https://cvg-data.inf.ethz.ch/openscene/models/$m.pth.tar; done
for m in scannet_openseg; do wget -c https://cvg-data.inf.ethz.ch/openscene/models/$m.pth.tar; done
cd $D/data
for z in scannet_processed/scannet_3d matterport_processed/matterport_3d_160 replica_processed/replica_3d; do
  wget -c https://cvg-data.inf.ethz.ch/openscene/data/$z.zip && unzip -qo $(basename $z).zip; done
$ROOT/env/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-3B-Instruct')"

# Instance-labelled data for interactive / query benchmarks and training (Pointcept releases).
# ScanNet requires accepting its terms on huggingface.co (auto-approved); set HF_TOKEN first.
mkdir -p $D/data/pointcept && cd $D/data/pointcept
$ROOT/env/bin/python -c "from huggingface_hub import hf_hub_download as h; h('Pointcept/s3dis-compressed', 's3dis.tar.gz', repo_type='dataset', local_dir='.')" \
  && mkdir -p s3dis && tar -xzf s3dis.tar.gz -C s3dis --strip-components=0 && rm s3dis.tar.gz
[ -n "$HF_TOKEN" ] && $ROOT/env/bin/python -c "from huggingface_hub import hf_hub_download as h; h('Pointcept/scannet-compressed', 'scannet.tar.gz', repo_type='dataset', local_dir='.')" \
  && mkdir -p scannet && tar -xzf scannet.tar.gz -C scannet && rm scannet.tar.gz
$ROOT/env/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-VL-7B-Instruct')" || true
