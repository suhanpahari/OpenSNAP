# OpenSNAP: Open-Vocabulary Segment-and-Query for 3D Point Clouds

<p align="center">
  <img src="assets/video/opensnap_explainer.gif" width="90%" alt="OpenSNAP system explainer"><br>
  <em>System explainer (60 s) — full video: <a href="assets/video/opensnap_explainer.mp4">assets/video/opensnap_explainer.mp4</a></em>
</p>

<p align="center">
  <img src="assets/video/viewer_demo.gif" width="90%" alt="Viewer demo on Replica room0"><br>
  <em>Real viewer session on Replica room0 (held out from training): 5 random objects clicked and named, then text queries
  — full video: <a href="assets/video/viewer_demo.mp4">assets/video/viewer_demo.mp4</a>. Demo clicks skip walls/floors;
  the gallery below is unfiltered.</em>
</p>

<p align="center">
  <img src="assets/gallery.png" width="90%" alt="Interactive viewer"><br>
  <em>Viewer: click an object → mask + name; type a phrase → its objects; type a class list → open-vocabulary labels.</em>
</p>

**OpenSNAP makes the promptable point-cloud segmenter [SNAP](https://github.com/neu-vi/SNAP) open-vocabulary and queryable.**
Click any object and get its mask *and its name*; type any phrase and get the matching objects; run declarative
queries such as `SELECT WHERE IS('chair') AND NEAR(IS('table'), 0.5)` over a single room or a whole building.

## What it does

| | |
|---|---|
| **Click → mask + name** | SNAP gives the mask; the name comes from fusing SNAP's language token with mask-pooled 3D CLIP features, and a small VLM chooses among the 3D-proposed names from renders of the object (*3D-propose / VLM-choose*). |
| **Text → objects** | A trained text decoder (SNAP+ v4) answers "segment the &lt;phrase&gt;" in one pass while SNAP's click path stays exactly frozen; a training-free fallback uses the phrase as SNAP's click oracle. |
| **Object relation + OVQL** | Each scene is materialised once into a table of objects (voxels, box, confidence, two language embeddings). OVQL queries combine `IS`, `CONTAINS`, `NEAR/ABOVE/BELOW/ON`, `AND/OR/NOT` with probabilistic semantics; answers are ranked by probability in ~1 ms. |
| **Semantic cracking** | If no stored object explains a phrase, one is materialised at query time and persisted (with a negative cache), so later queries reuse it. |
| **Building-scale search** | Many rooms share one persisted catalog; spatial predicates run as a batched per-room join. |

## Key results

All numbers are produced by the scripts in this repo; every table, with its source file, is in
**[results/TABLES.md](results/TABLES.md)** (regenerate with `python3 scripts/make_tables.py`).

| task | benchmark | SNAP (its own text head) | other baseline | **OpenSNAP** |
|---|---|---|---|---|
| open-vocab 3D segmentation (mIoU) | ScanNet val | 41.5 | OpenScene 52.0 | **58.2** |
| | Matterport-160 (unseen classes) | 6.9 (1.9) | OpenScene 9.5 (5.5) | **11.6 (5.7)** |
| | nuScenes LiDAR val, 6019 keyframes (outdoor, 16 classes) | 23.8 | OpenScene 26.9 | **32.0** |
| | SemanticKITTI seq 08, 19 classes (mIoU / mAcc) | **32.8** / 43.7 | OpenScene (nuScenes model) 20.8 / 37.1 | 31.5 / **45.8** |
| open-vocab instance segmentation (AP, same protocol) | S3DIS Area 5, 13 classes | 10.5 | — | **15.4** |
| | Replica, 8 scenes (AP / AP50) | 14.8 / 19.7 | OpenMask3D 13.1 / 18.4 (published) | **19.9 / 25.9** |
| | nuScenes LiDAR, 10 thing classes (AP / AP50) | 7.9 / 12.1 | — | **13.9 / 18.8** |
| | SemanticKITTI seq 08, 8 thing classes (AP / AP50) | 13.9 / 16.3 | — | **20.6 / 23.0** |
| click → name (accuracy) | Matterport-160, 212 clicks | 30.7 | VLM alone 18.9 | **40.1** (7B VLM) |
| | Matterport-160, unseen classes only | 12.2 | VLM alone 6.5 | **17.2** |
| text → mask (mIoU) | ScanNet200 val (head/common/tail) | — | training-free 11.8 | **18.2** (32.8 / 13.8 / 2.7) |
| | S3DIS (zero-shot dataset) | — | training-free 20.0 | **27.2** |
| query latency | S3DIS room, stored relation | — | re-processing the scan 8.8 s | **0.8 ms** |
| | building: 68 rooms, 5,149 objects | — | — | **0.9–3.4 ms** |
| storage | 68-room catalog vs raw XYZ | — | 945 MB | **171 MB** |
| semantic cracking (repeat queries / accuracy) | S3DIS workload, v4 refiner | — | refine every time 26.8 ms | **3.7 ms**, AP50 38.7 vs 38.9 |
| interactive click segmentation | S3DIS, IoU@1/3/5 | 53.3 / 69.1 / 74.0 | — | identical (SNAP frozen) |
| | S3DIS object crops (SNAP Table 2), IoU@1/3/5 | 56.1 / 72.0 / 76.8 | AGILE3D 58.7 / 77.4 / 83.6 (published) | identical (SNAP frozen) |

## Open-vocabulary gallery

Unfiltered results from the viewer on 17 scenes from four datasets (none of them SNAP's example scenes). Each grid shows
10 text queries (★ = word absent from SNAP's training vocabulary), 4 randomly chosen objects clicked and named
(the VLM's choice shown below), open-vocabulary labels with the scene's own query words, and segment-everything.
In query tiles the best match is red and other strong matches orange; when no stored object matches confidently the
text decoder's mask is shown instead. Failures are kept (e.g. `rug` → sofa, `sideboard` → coffee table first).

<details open><summary><b>Replica</b> (8 scenes, held out from all training)</summary>

![replica_room0](assets/openvocab_gallery/replica_room0.jpg)
![replica_room1](assets/openvocab_gallery/replica_room1.jpg)
![replica_room2](assets/openvocab_gallery/replica_room2.jpg)
![replica_office0](assets/openvocab_gallery/replica_office0.jpg)
![replica_office1](assets/openvocab_gallery/replica_office1.jpg)
![replica_office2](assets/openvocab_gallery/replica_office2.jpg)
![replica_office3](assets/openvocab_gallery/replica_office3.jpg)
![replica_office4](assets/openvocab_gallery/replica_office4.jpg)
</details>

<details><summary><b>Matterport3D</b> (3 regions; SNAP never trained on Matterport)</summary>

![mp3d_pa4otMbVnkk_06](assets/openvocab_gallery/mp3d_pa4otMbVnkk_06.jpg)
![mp3d_yqstnuAEVhm_11](assets/openvocab_gallery/mp3d_yqstnuAEVhm_11.jpg)
![mp3d_wc2JMjhGNzB_34](assets/openvocab_gallery/mp3d_wc2JMjhGNzB_34.jpg)
</details>

<details><summary><b>S3DIS</b> (3 rooms; zero-shot for SNAP)</summary>

![s3dis_conferenceRoom_2](assets/openvocab_gallery/s3dis_conferenceRoom_2.jpg)
![s3dis_office_19](assets/openvocab_gallery/s3dis_office_19.jpg)
![s3dis_pantry_1](assets/openvocab_gallery/s3dis_pantry_1.jpg)
</details>

<details><summary><b>ScanNet</b> (3 val scenes)</summary>

![scannet_scene0050_00](assets/openvocab_gallery/scannet_scene0050_00.jpg)
![scannet_scene0231_00](assets/openvocab_gallery/scannet_scene0231_00.jpg)
![scannet_scene0568_00](assets/openvocab_gallery/scannet_scene0568_00.jpg)
</details>

<details open><summary><b>Outdoor LiDAR</b> (nuScenes val, XYZ-only, free vocabulary)</summary>

![outdoor](assets/outdoor_gallery.jpg)

Labelled with a free vocabulary: road, grass, tree, building, street light, fence, … (★) are not nuScenes class names.
</details>

<details><summary><b>Building-level search</b>: one query over all 68 rooms of S3DIS Area 5</summary>

![building](assets/building_hits.png)

`sofa` and its synonym `couch` retrieve the same objects across rooms; `blackboard` and `pillar` (flat / structural)
fail with low probabilities, shown as-is.
</details>

## Install

```bash
git clone --recursive https://github.com/suhanpahari/OpenSNAP opensnap
bash opensnap/install.sh            # ../env: torch 2.4 + CUDA 12.4, MinkowskiEngine (patched for CUDA 12), viser, Qwen2.5-VL
WITH_MEDIA=1 bash opensnap/install.sh   # optional: Playwright (screenshots/demo video) and Manim (explainer video)
HF_TOKEN=... bash opensnap/download.sh  # checkpoints, OpenScene data, Replica, Pointcept S3DIS/ScanNet, Qwen2.5-VL -> ../downloads
export PYTHONNOUSERSITE=1
```

Trained OpenSNAP checkpoints are kept locally in `../downloads/checkpoints/` (see its README):
`opensnap_text_v4.pth` (text decoder, 6.9 MB) and `opensnap_click_v3_6k.pth`. Both load SNAP-C and OpenScene separately.

## Viewer

```bash
cd opensnap
../env/bin/python scripts/app.py --scene ../downloads/demo/replica_room0   # dir with coord.npy/color.npy, or .pth/.ply/.pcd/.bin/.npy
# open http://localhost:8080
```

- **Click** an object → mask, instant 3D name, then the VLM-verified name; **shift-click** refines; *keep previous objects* collects several.
- **Text query** → the text decoder's mask for the phrase (or ranked proposals when `--text_ckpt ''`).
- **Open-vocabulary labels** → type any class list. **Show segments** → colour all segments, click one to name it.
- Options: `--vlm Qwen/Qwen2.5-VL-7B-Instruct`, `--vlm ''`, `--text_ckpt ''`, `--openscene_ckpt ''` with `--domain Outdoor` for LiDAR (SNAP only).

## Python API

```python
from opensnap.snap_model import SnapModel
from opensnap.openscene_model import OpenSceneModel
from opensnap.text import TextEncoder
from opensnap.text_decoder import TextDecoder
from opensnap.core import OpenSNAP
from opensnap.namer import VLMNamer
from opensnap.data import prior_vocab, snap_train_vocab
from opensnap.query.catalog import materialize
from opensnap.query.engine import Engine

clip = TextEncoder("ViT-B/32")
dense = OpenSceneModel("../downloads/ckpt/matterport_openseg.pth.tar", "openseg")
m = OpenSNAP(SnapModel("../downloads/ckpt/SNAP_C.pth"), clip, dense, TextEncoder(dense.clip_name),
             anchors=snap_train_vocab(),
             text_decoder=TextDecoder("../downloads/checkpoints/opensnap_text_v4.pth", clip))
m.encode(coord)                                   # (N, 3) XYZ
obj = m.click([point_index])                      # object from a click
name, ranked, vlm_pick, _ = VLMNamer().name_choose(m, obj, 0, coord, color, prior_vocab("snap+lvis"))
mask, source, conf = m.text_segment("wicker chair")          # text -> voxel mask
table = materialize(m)                                       # object relation
answers, stats = Engine(m, table).run("SELECT WHERE IS('chair') AND NEAR(IS('table'), 0.5) LIMIT 5")
```

## Reproduce

```bash
python scripts/eval_semseg.py --dataset scannet --split val                       # T1/T2/T15
python scripts/eval_queries.py --dataset s3dis [--keep_all 1 --proposals c2f]      # T3/T4
python scripts/prep_replica.py; python scripts/eval_queries.py --dataset replica --proposals c2f   # T4b
python scripts/prep_nuscenes.py --root <nuScenes v1.0-trainval with lidarseg>      # official val -> ../downloads/data
python scripts/eval_semseg.py --dataset nuscenes --domain Outdoor --dense_voxel 0.05 --feature openseg \
       --openscene_ckpt ../downloads/ckpt/nuscenes_openseg.pth.tar                 # T2b
python scripts/eval_queries.py --dataset nuscenes --domain Outdoor --dense_voxel 0.05 --stride 4 \
       --openscene_ckpt ../downloads/ckpt/nuscenes_openseg.pth.tar                 # T4c
python scripts/eval_interactive.py --dataset nuscenes --domain Outdoor --stuff 10 --min_points 10 --stride 10   # T5d
python scripts/prep_kitti.py --root <SemanticKITTI dataset/sequences>; then the same three scripts with
       --dataset semantickitti (--stuff 8 for interactive)                                              # T2c/T4d/T5e
python scripts/eval_interactive.py --model snap [--crops 1]                       # T5/T5c
python scripts/eval_naming.py --dataset matterport160 --split test --prior snap+lvis --feature openseg \
       --openscene_ckpt ../downloads/ckpt/matterport_openseg.pth.tar [--unseen_only] # T6/T7
python scripts/eval_text.py --dataset scannet --method trained --ckpt ../downloads/checkpoints/opensnap_text_v4.pth  # T8
python scripts/eval_proposals.py; python scripts/eval_systems.py; python scripts/eval_building.py                 # T10-T16
python scripts/train.py --dataset scannet --arch v4 --num_masks 0 --text_classes 12 --steps 16000                 # text decoder
python scripts/capture_demo.py ...; python scripts/make_gallery.py; python scripts/building_hits.py              # visuals
python scripts/make_tables.py                                                                                     # results/TABLES.md
```

`--gpus` accepts physical GPU ids or MIG UUIDs; each worker is pinned with `CUDA_VISIBLE_DEVICES`.

## Limitations

- **Segmentation itself is SNAP's.** Fine-tuning SNAP's decoder on ScanNet alone loses its multi-dataset generalisation,
  so OpenSNAP keeps SNAP frozen; click segmentation is identical to SNAP-C (T5, T9).
- **Flat objects on walls/floors** (paintings, whiteboards, rugs, maps) are hard: geometry cannot separate them from the wall.
- **Truly novel concepts** (vase, sculpture) remain hard; most unseen-class gains come from near-synonyms of seen classes.
- **Semantic cracking** with the training-free refiner costs instance AP (T12, T13); with the trained text decoder as
  refiner the cost is ≈0.2 AP50 (T12b, T12c).
- SNAP's published S3DIS open-vocabulary AP (16.1) uses its own evaluation; under one shared protocol OpenSNAP is 15.4 vs 10.5.
- Click-to-name maps free-form names to labels with CLIP text similarity; run-to-run noise is about ±3 points.

## Acknowledgements

Built on [SNAP](https://github.com/neu-vi/SNAP), [OpenScene](https://github.com/pengsongyou/openscene),
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [CLIP](https://github.com/openai/CLIP),
[Pointcept](https://github.com/Pointcept/Pointcept), [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine),
[viser](https://github.com/nerfstudio-project/viser) and [Manim](https://www.manim.community/).
