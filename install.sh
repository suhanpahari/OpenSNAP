#!/bin/bash
# Creates ./env (next to this repo) with torch 2.4 / CUDA 12.4, SNAP deps, MinkowskiEngine (patched for CUDA 12), viser and Qwen2.5-VL.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
E=$ROOT/env
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR=$ROOT/.cache/pip CONDA_PKGS_DIRS=$ROOT/.cache/conda_pkgs
conda create -y -p $E python=3.10
$E/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
$E/bin/pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
$E/bin/pip install spconv-cu120 fire timm addict einops scipy plyfile h5py pyyaml tensorboardx hdbscan pandas ftfy regex tqdm \
  gdown open3d torch-geometric python-dateutil ninja nuscenes-devkit viser "transformers>=4.49,<4.57" accelerate qwen-vl-utils
$E/bin/pip install git+https://github.com/openai/CLIP.git
$E/bin/pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
conda install -y -p $E -c conda-forge openblas
conda create -y -p $ROOT/downloads/gcc10 -c conda-forge gcc_linux-64=10 gxx_linux-64=10
mkdir -p $ROOT/downloads && cd $ROOT/downloads
[ -d MinkowskiEngine ] || git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine && git apply $ROOT/opensnap/patches/minkowski_cuda12.patch || true
G=$ROOT/downloads/gcc10/bin
CC=$G/x86_64-conda-linux-gnu-gcc CXX=$G/x86_64-conda-linux-gnu-g++ NVCC_PREPEND_FLAGS="-ccbin $G/x86_64-conda-linux-gnu-g++" \
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0} MAX_JOBS=16 \
LIBRARY_PATH=$E/lib LDFLAGS="-L$E/lib -Wl,-rpath,$E/lib" \
$E/bin/python setup.py install --blas=openblas --blas_include_dirs=$E/include --blas_library_dirs=$E/lib --force_cuda
