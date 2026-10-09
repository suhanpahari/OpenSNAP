# OpenSNAP — Technical Notes

Working notes for the ICDE 2027 submission (research track, area: *Data models, semantics, and query languages*).
Numbers are filled in only from runs in `results/`; anything not yet measured is marked **TBD**.

## 1. Problem

3D point clouds (indoor scans, LiDAR) are stored and exchanged as raw coordinates. Answering even simple questions —
*"where are the fire extinguishers?"*, *"which chairs are next to a table?"* — requires a custom perception pipeline
per question and per vocabulary. We want a **queryable representation** of a point cloud that supports
open-vocabulary, spatial and interactive queries with probabilistic answers, without re-processing the scan per query.

## 2. Data model: the object relation

A scene `S` (voxelised at 2 cm, `N` voxels) is materialised into a relation

```
Objects(oid, V ⊆ {1..N}, bbox, centroid, conf, e ∈ R^512, p ∈ R^D, prov)
```

| attribute | meaning |
|---|---|
| `V` | voxel set (stored as sorted int32 ids) |
| `bbox`, `centroid` | geometry for spatial predicates |
| `conf` | promptable segmenter's predicted IoU (mask quality) |
| `e` | segmenter's language token (CLIP text space) |
| `p` | mask-pooled 3D vision–language feature `p = norm(Σ_{i∈V} w_i d_i)` |
| `prov` | provenance: `auto` (ingest), `click`, or `query:<text>` (created by semantic cracking) |

Semantics are **open-vocabulary and probabilistic**: no label column is stored. For any text `t`,
`P(t | o)` is computed at query time from `(e, p)` (Sec. 4), calibrated against a background vocabulary `B`.

## 3. Query language (OVQL)

```
SELECT [WHERE expr] [LIMIT k] [THRESHOLD τ]
expr   := term (OR term)*          term := factor (AND factor)*
factor := NOT factor | (expr) | pred
pred   := IS('text') | CONTAINS(x,y,z) | NEAR(expr, d) | ABOVE(expr) | BELOW(expr) | ON(expr[, d])
```

Example: `SELECT WHERE IS('chair') AND NEAR(IS('table'), 0.5) LIMIT 5`.

## 4. Semantics

Each predicate maps an object to a probability; composition follows the tuple-independence convention of
probabilistic databases.

* `IS(t)(o) = P(t | o)`: product of experts over the pooled feature and the language token,
  `log P ∝ α · log softmax_τ(p·T_d(t, B)) + (1−α) · log P_tok(t | e)`, where `P_tok` reads the token **through the
  segmenter's training vocabulary** (vocabulary relay: posterior over training anchors × anchor–query text similarity).
* `CONTAINS(x)(o) = conf(o) · [x ∈ V_o]`
* Spatial `R ∈ {NEAR, ABOVE, BELOW, ON}`: `R(E)(o) = max_{o'≠o} R(o, o') · E(o')` over bounding-box relations.
* `AND = ×`, `OR = 1 − (1−a)(1−b)`, `NOT = 1 − a`.

## 5. Query processing

Two-tier plan:

1. **Phase 1 – coverage & semantic cracking.** For every `IS(t)` in the plan, if `max_o P(t|o) < θ`, no stored object
   explains `t`. The engine materialises one *at query time* by using the text as a click oracle for the promptable
   segmenter: clicks are placed where text relevance is high but the current mask is missing (Sec. 6). If geometry
   cannot separate the object (agreement < 0.3, e.g. flat objects on walls), the relevance region is used instead.
   The new tuple is **persisted** with provenance `query:t`, so later queries reuse it — database cracking applied to
   semantics.
2. **Phase 2 – probabilistic evaluation** over the (now fixed) relation; ranking by probability, `LIMIT`/`THRESHOLD`.
3. **Optional verification.** Cycle consistency: a name for an object is accepted only if `IS(name)` re-segments the
   same object (IoU / click containment).

Cost split: the backbone and dense features run once per scene (ingest); every query touches only the small decoder
and vector arithmetic over `|Objects|` rows.

## 6. Language as the click oracle

Promptable point-cloud segmenters are trained to follow *simulated users* who click in the error region of the
current mask. We replace the user with a text relevance map `r_t(i)`: the first click is the densest high-relevance
voxel; each further click goes to the most relevant voxel inside the target region `{r_t > τ·max}` not yet covered by
the mask; we keep the click set with the best mask/relevance agreement. This turns a click-only segmenter into a
text-promptable one without training.

## 7. Trained decoders (SNAP+)

Backbone and SNAP decoder frozen in the final system; ScanNet200 train only (S3DIS, Matterport, Replica stay zero-shot).

* **v1/v2 (rejected).** Fine-tuning SNAP's decoder on ScanNet alone loses its 7-dataset generalisation (control run:
  IoU@5 71.0 vs 72.8 on S3DIS). v1 also blurred same-class instances by adding semantics to the mask embeddings.
* **v3 (click parity).** SNAP frozen; extra single-click tokens run in a separate pass so the multi-click mask is
  exactly SNAP's. Oracle headroom of the extra tokens is only +2.0 (ScanNet) / +3.0 (S3DIS) and the IoU head cannot
  realise it, so v3 is kept only as a parity reference.
* **v4 (text decoder, shipped).** Clicks → exact frozen SNAP. Text → a trainable copy of SNAP's two-way decoder and
  mask head; per-voxel 3D CLIP features enter only its attention keys; a CLIP text embedding is the prompt token.
  Trained on head+common classes (66 tail classes held out). Used for "segment the X", as the fallback of the viewer's
  text query, and as the query-time refiner of semantic cracking (where it removes cracking's accuracy cost).

## 7b. Corrective clicking: negative clicks on a positive-only segmenter

SNAP accepts positive clicks only, so once its mask spills onto a neighbour, further clicks cannot remove the spill and
IoU plateaus (S3DIS crops: 76.8 → 80.4 from 5 to 10 clicks). Its mask is a dot product `z_i = E_i · m` between the refined
voxel embeddings `E` (output of the two-way decoder) and one mask vector `m`. We add negative clicks in two ways.

* **Test-time token tuning (training-free).** Decode the positive clicks once with frozen SNAP, keep `E` fixed, and take
  30 Adam steps on `m` (plus a scalar bias) minimising a margin softplus loss on the clicked voxels (positives above +2,
  negatives below −2) with an anchor `λ‖m − m₀‖² / ‖m₀‖²`, λ = 10. Only a D-dim vector is optimised over cached
  embeddings: ~43 ms per click on a 267k-voxel room. Hyper-parameters chosen on ScanNet val only (T5g).
* **Corrective decoder (trained, v5).** Positive-only click sets → exact frozen SNAP (parity guaranteed). Once a negative
  click exists → a trainable copy of SNAP's two-way decoder and mask head, with learned positive/negative click-type
  embeddings added to the prompt tokens and the previous mask injected as a dense prompt (`pe + σ(z_prev)·e_prev`).
  Trained with simulated corrective clicking (each click in the larger error region), frozen backbone, 9k steps on an
  equal mix of ScanNet, nuScenes-train and SemanticKITTI-train. The two compose: token tuning on top of the decoder's
  `(E, m)`.
* **Baselines.** SNAP under the same click sequence (negatives ignored) and negative-mask subtraction (decode each
  negative click alone and remove its mask where it dominates).

## 7c. Backbone adapters, click-jitter TTA, video ingest

* **Backbone adapters (v6).** Rank-16 LoRA on `qkv` and `proj` of every serialized-attention layer of PTv3, B = 0 at
  init (exact SNAP at step 0). Trained jointly with the corrective decoder on ScanNet + nuScenes-train +
  SemanticKITTI-train; losses: positive-click masks through SNAP's frozen decoder (1–3 clicks), corrective masks, and
  feature distillation `‖f − f₀‖² / ‖f₀‖²` to the un-adapted backbone. Gains in the training domains (nuScenes IoU@1
  +3.5), small first-click cost on held-out indoor crops (T5i) → use for the Outdoor domain.
* **Click-jitter TTA.** The positive click set is decoded 8 more times with each click moved to a random voxel within
  10 cm; logits are averaged. Radius chosen on ScanNet val (T5k).
* **Video ingest.** `scripts/video2cloud.py`: sharpest frame per window → VGGT-1B depth + cameras → unprojection →
  confidence filter → metric scale = median ratio of Depth-Anything-V2 (metric, indoor) to VGGT depth on 8 frames →
  RANSAC plane whose normal is within ~35° of the first camera's up axis → rotate to z-up. Cameras, depth and frame
  paths are stored in `views.npz`.
* **Dense collapse guard.** At ingest, the dense features label every voxel with SNAP's training vocabulary; if one
  label that is not wall/floor/ceiling-like takes > 75 % of the object voxels, the dense branch is dropped and text
  relevance is computed from the segmenter's own proposals (soft-mask-weighted vote of their token probabilities).
  Benchmark scans: 0.14–0.66; the phone-video scan: 0.80 (T18).
* **Photo-grounded naming.** For scenes with `views.npz`, a selected 3D mask is projected into all frames, visibility is
  checked against VGGT depth, the two frames with the largest visible fraction (penalising off-centre views) are cropped
  around the mask's 2–98 % pixel extent with a red box, and the VLM names the photo; its ranked answer is used directly.

## 8. Experimental protocol

* **Interactive segmentation** (S3DIS Area 5 full rooms, zero-shot): XYZ-only input; first click at the object
  centre (farthest from the boundary), next clicks at the centre of the missed region; IoU@{1,3,5,10}, NoC@{80,85,90}.
  Our evaluation of the released checkpoint reproduces the published numbers (53.3/69.1/74.0 vs 53.6/71.1/77.6 IoU@1/3/5).
* **Corrective clicking**: same click placement, but each click goes to the larger of the missed / false-positive
  regions (positive or negative), as in AGILE3D / Point-SAM; S3DIS crops (SNAP Table 2 setting), S3DIS rooms, Replica,
  nuScenes and SemanticKITTI (every 10th scan, things = instances, stuff = one segment per class).
* **Query workload** (S3DIS Area 5): concept queries `IS(c)` and relational `IS(a) AND NEAR(IS(b), 0.5)` with answer
  sets derived from instance annotations; interpolated AP over IoU 0.50:0.95, AP50, AP25; latency; storage.
* **Open-vocabulary naming**: click-to-name accuracy on Matterport3D-160 (seen/unseen classes).
* **Text prompts**: ScanNet200 val head/common/tail mIoU; S3DIS zero-shot.

## 9. Results

All results, with their source files, are generated into [`results/TABLES.md`](../results/TABLES.md) by
`scripts/make_tables.py` (tables include semantic segmentation and ablation, query workloads, SNAP-protocol instance AP,
interactive segmentation and its diagnosis, click-to-name with two VLM sizes, text prompts on ScanNet200 and S3DIS,
single-click selection, proposal recall, systems, cracking and its refiners, calibration sweeps, dense-feature source,
building-level catalog).

## 10. References (to cite in the paper)

* Promptable point-cloud segmentation: SNAP (Gupta et al., 3DV 2026, arXiv:2510.11565); AGILE3D (Yue et al., ICLR 2024);
  Interactive4D (Fradlin et al.); Point-SAM (Zhou et al., 2024); InterObject3D (Kontogianni et al.).
* Segment Anything (Kirillov et al., ICCV 2023) — multi-mask tokens, IoU head.
* Open-vocabulary 3D: OpenScene (Peng et al., CVPR 2023); OpenMask3D (Takmaz et al., NeurIPS 2023); OpenIns3D;
  SAI3D; LSeg; OpenSeg; CLIP (Radford et al., ICML 2021).
* Vision–language models: Qwen2.5-VL.
* Backbone: Point Transformer V3 (Wu et al., CVPR 2024); Pointcept; MinkowskiEngine.
* Datasets: ScanNet / ScanNet200, S3DIS, Matterport3D, Replica, LVIS (vocabulary).
* Data management: database cracking (Idreos, Kersten, Manegold, CIDR 2007); probabilistic databases
  (tuple-independence; Suciu, Olteanu, Ré, Koch); spatial query processing; vector databases.
