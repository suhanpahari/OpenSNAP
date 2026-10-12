#!/bin/bash
# Data for the comparative study (OVI-MAP, Search3D/OpenMask3D, HouseLayout3D, OpenScan protocols) into ../downloads.
# Run after download.sh. SigLIP / CLIP / SAM weights are fetched on first use into ../downloads/hf.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); D=$ROOT/downloads
export PYTHONNOUSERSITE=1 HF_HOME=$D/hf

# Replica, NICE-SLAM release: GT meshes + posed RGB-D frames (OVI-MAP protocol)
mkdir -p $D/replica_nice && cd $D/replica_nice
wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip && unzip -qo Replica.zip

# OVI-MAP evaluation code and Replica GT label files
B=https://raw.githubusercontent.com/OVI-MAP/OVI-MAP/main
mkdir -p $D/papers/ovimap && cd $D/papers/ovimap
for f in scripts/eval_sem_seg.py scripts/eval_utils.py scripts/eval_inst_seg.py scripts/utils/semantic_const.py scripts/utils/scannet200_splits.py; do
  curl -s --create-dirs -o $f $B/$f; done
for s in office0 office1 office2 office3 office4 room0 room1 room2; do
  curl -s --create-dirs -o scripts/datasets/replica_gt_semantics/semantic_labels_$s.txt $B/scripts/datasets/replica_gt_semantics/semantic_labels_$s.txt
  curl -s --create-dirs -o scripts/datasets/replica_gt_instances/instance_labels_$s.txt $B/scripts/datasets/replica_gt_instances/instance_labels_$s.txt
done

# ScanNet posed frames (OpenScene release: every 20th frame, 320x240, all train + val scenes)
cd $D/data
wget -c https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip && unzip -qn scannet_2d.zip

# HouseLayout3D (16 Matterport3D buildings: layouts, doors, levels) and OpenScan (attribute GT on ScanNet200 val)
$ROOT/env/bin/python -c "from huggingface_hub import snapshot_download as s; s('houselayout3d/HouseLayout3D', repo_type='dataset', local_dir='$D/hl3d/data')"
$ROOT/env/bin/python -c "from huggingface_hub import snapshot_download as s; s('garrying/OpenScan', repo_type='dataset', local_dir='$D/openscan')"
