#!/bin/bash
# Comparative study (results/compare/*, tables C1-C15). Run from the repo root after download.sh and download_compare.sh.
# Each eval_standard run writes results/compare/<name>.json (+ _recs.pt); --dump saves per-object expert terms that the
# selection scripts re-score without re-running the model. Settings are chosen on development scenes only
# (S3DIS Area 1, 30 ScanNet train scenes) and applied once to the test sets.
set -e
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=../env/bin/python; R=../downloads/runs
SC="--openscene_ckpt ../downloads/ckpt/scannet_openseg.pth.tar"
MP="--openscene_ckpt ../downloads/ckpt/matterport_openseg.pth.tar"
U="--proposals union --c2f_levels 4"
ev() { out=$1; shift; $PY scripts/eval_standard.py "$@" --out results/compare/$out.json; }

# Development scenes (never reported): proposal mode and 3D labelling setting
ev dev/s3dis_ul4_dump --bench s3dis_dev $MP $U --dump $R/dump_s3dis_dev
ev dev/scannet_ul4_dump --bench scannet_sel $SC $U --dump $R/dump_scannet_sel
ev dev/scannet_sel_img --bench scannet_sel $SC $U --images 1 --dump $R/dump_scannet_sel_img

# Test sets: OVI-MAP protocol (Replica 8 scenes, ScanNet 18 scenes), Search3D/OpenMask3D protocol (ScanNet200 val)
$PY scripts/sam_frames.py                                                   # SAM masks of Replica frames -> $R/sam_replica
ev replica_ul4 --bench replica $MP $U --images 1 --dump $R/dump_replica_ul4
ev replica_lift --bench replica $MP $U --images 1 --lift2d $R/sam_replica --dump $R/dump_replica_lift
ev scannet18 --bench scannet18 $SC
ev scannet18_ul4 --bench scannet18 $SC $U --dump $R/dump_scannet18
ev scannet18_img --bench scannet18 $SC $U --images 1 --dump $R/dump_scannet18_img
ev scannet200_val --bench scannet200 $SC
ev scannet200_ul4 --bench scannet200 $SC $U --dump $R/dump_scannet200
ev scannet200_img --bench scannet200 $SC $U --images 1 --dump $R/dump_scannet200_img

# Selection on development scenes, applied once to test dumps
$PY scripts/tune_dev.py \
  --dev $R/dump_s3dis_dev:results/compare/dev/s3dis_ul4_dump_recs.pt:s3dis_dev/9 \
        $R/dump_scannet_sel:results/compare/dev/scannet_ul4_dump_recs.pt:scannet200/198 \
  --test scannet18:$R/dump_scannet18:results/compare/scannet18_ul4_recs.pt:scannet200/200 \
         replica:$R/dump_replica_ul4:results/compare/replica_ul4_recs.pt:replica51
$PY scripts/apply_setting.py --bench scannet18 --inst 200 --dump $R/dump_scannet18 --recs results/compare/scannet18_ul4_recs.pt --out results/compare/scannet18_tuned.json
$PY scripts/apply_setting.py                                                # ScanNet200, 198 classes, head/common/tail
for d in replica_ul4 replica_lift; do $PY scripts/image_terms.py --dump $R/dump_$d; done
for b in scannet_sel:dump_scannet_sel_img scannet18:dump_scannet18_img scannet200:dump_scannet200_img; do
  $PY scripts/image_terms.py --bench ${b%%:*} --dump $R/${b#*:} --encoders siglip_so400m; done
$PY scripts/tune_image_dev.py                                               # image column chosen on dev, applied to all test sets

# Leave-one-scene-out variants on Replica (sensitivity; reported with their protocol)
$PY scripts/loso_lift.py --base results/compare/tuned.json --out results/compare/replica_lift_loso_tuned.json
$PY scripts/loso_image.py --dump $R/dump_replica_ul4 $R/dump_replica_lift --recs results/compare/replica_ul4_recs.pt results/compare/replica_lift_recs.pt \
  --out results/compare/replica_joint_loso_image.json

# Buildings, scale, calibration, workload, relational queries, attributes, guard, video
$PY scripts/eval_partition.py                                               # HouseLayout3D joins (builds ../downloads/catalogs/hl3d)
$PY scripts/eval_scale.py                                                   # campus replication, pushdown join
$PY scripts/eval_calibration.py cache && $PY scripts/eval_calibration.py analyze
$PY scripts/eval_workload.py
$PY scripts/eval_queries.py --dataset replica --proposals c2f --agg_sweep 1 --out results/compare/agg_replica.json
$PY scripts/eval_queries.py --dataset s3dis --agg_sweep 1 --out results/compare/agg_s3dis.json
$PY scripts/eval_openscan.py
$PY scripts/eval_guard.py
$PY scripts/video_replica.py                                                # RGB-only Replica videos via VGGT
ev replica_video --bench replica_video $MP --proposals c2f --max_dist 0.1 --images 1
ev replica_video_oracle_scale --bench replica_video $MP --proposals c2f --max_dist 0.1 --images 1 --oracle_scale 1

# Figure and tables
$PY scripts/qualitative_priors.py                                           # assets/compare/relational_priors.jpg
$PY scripts/make_tables.py                                                  # results/TABLES.md
