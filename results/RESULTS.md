# OpenSNAP — results, analysis and paper notes

Everything needed to write the paper, in one place. Numbers are copied from [`TABLES.md`](TABLES.md), which is
generated from the JSON files in `results/` by `scripts/make_tables.py`; table IDs (T1, T5f, …) refer to it. Method
details are in [`docs/TECHNICAL_NOTES.md`](../docs/TECHNICAL_NOTES.md). When a number here and in `TABLES.md` disagree,
`TABLES.md` wins.

---

## 0. One-paragraph summary

OpenSNAP turns SNAP, a promptable point-cloud segmenter with a closed text head, into an **open-vocabulary 3D object
database**. A scan is ingested once into an **object relation** (one tuple per segment: geometry, a segmenter token and
mask-pooled 3D CLIP evidence); open-vocabulary predicates get **probabilistic semantics** and are answered by an
**open-vocabulary query language (OVQL)** with concept, spatial-relation and boolean operators, processed in milliseconds
and refined lazily by **semantic cracking**. On top of that, OpenSNAP improves SNAP's own interactive segmentation by
adding **negative / corrective clicks** to a positive-only model (test-time token tuning, a trained corrective decoder,
backbone adapters), names clicked objects with a **3D-propose / VLM-choose** scheme, and runs on **phone videos**
(VGGT reconstruction + photo-grounded naming + a dense-feature collapse guard). It is evaluated on 7
datasets (ScanNet, ScanNet200, Matterport3D-160, S3DIS, Replica, nuScenes, SemanticKITTI) with 32 tables.

## 1. Headline results (safe to claim)

| claim | evidence | numbers |
|---|---|---|
| Open-vocab 3D semantic segmentation beats both SNAP's text head and OpenScene | T1/T2, T2b | ScanNet 58.2 vs 41.5 (SNAP) / 52.0 (OpenScene); nuScenes 32.0 vs 23.8 / 26.9 |
| Generalises to classes the segmenter never saw | T1/T2 | Matterport-160 unseen classes 5.7 vs 1.9 (SNAP), 5.5 (OpenScene) |
| OVQL query accuracy beats SNAP's text token on every dataset (query workload, classes present) | T4, T4b–d | S3DIS 15.4 vs 10.5; Replica 19.9 vs 14.8; nuScenes 13.9 vs 7.9; SemanticKITTI 20.6 vs 13.9 (not comparable to OpenMask3D's closed-set protocol, see C1) |
| Under OVI-MAP's exact protocol, stored objects beat OVI-MAP on strict AP | C1, C2 | Replica AP 16.9 (image column) / 12.9 (XYZ only) vs 8.5; ScanNet-18 AP 9.8 (XYZ only) vs 7.2 |
| OVQL beats phrase-embedding retrieval on relational queries | C9 | NEAR AP50 21.7 vs 14.3 (S3DIS), 24.8 vs 21.7 (Replica); ECE 0.02–0.03 vs 0.07–0.09 |
| IS(c) is calibrated across 5 datasets | C5 | ECE 0.015–0.022 with the 234-word background; 0.004–0.010 after Platt scaling fitted on ScanNet train |
| Building joins: halo makes room partitioning exact; cross-floor vertical answers removed | C4 | 16 HouseLayout3D buildings: room-only NEAR recall 68–85%, room+halo 100%; ~50% of global ABOVE/BELOW answers are cross-floor |
| Campus scale | C8 | 1.22 M objects, NEAR 2.6 ms with pushdown vs 92.6 ms padded |
| Click → name: 3D-propose / VLM-choose beats the segmenter and the VLM alone | T6/T7 | 40.1% vs 30.7% (segmenter) vs 18.9% (VLM free); unseen 17.2% vs 12.2% |
| Text → mask on a zero-shot dataset | T8b | S3DIS 27.2 mIoU vs 20.0 training-free |
| Negative clicks fix SNAP's positive-only plateau, on all 5 datasets | T5f | IoU@5 / IoU@10: S3DIS crops 81.8 / 88.8 vs 73.1 / 74.6; Replica 74.8 / 81.9 vs 59.5 / 60.1; nuScenes 73.4 / 79.2 vs 63.8 / 65.4 |
| Querying the stored relation is 4 orders of magnitude faster than re-processing | T11 | 0.8 ms vs 8.8 s per query; relation 6.7× smaller than raw XYZ |
| Building-scale: one query over 68 rooms / 5,149 objects | T16 | 0.9–3.4 ms per query; 171 MB catalog vs 945 MB raw |
| Cracking makes repeated queries cheap at ≈ no accuracy cost (v4 refiner) | T12c | 2nd pass 3.7 ms vs 26.8 ms (refine every time); AP50 38.7 vs 38.9 |

## 2. ICDE 2027 framing (research track, "Data models, semantics, and query languages")

**Deadline**: Round 2, 2026-11-11 5pm PT. IEEE format, 12 pages + refs, no appendix, single-blind, public artifact
expected, mandatory AI-use disclosure. Desk-reject risk: papers that "purely advance ML" without a data-engineering
contribution. Write it as a **data model + query language + query processing** paper whose substrate happens to be 3D
scans; the vision components are *operators* inside the system.

### 2.1 Problem statement (DB terms)
Point clouds are raw, unstructured, and every semantic question ("where are the chairs near a whiteboard?") today means
re-running a segmentation network over the whole scan with a fixed label set. We want a **queryable, open-vocabulary
representation**: ingest once, ask arbitrary concept and spatial queries later, with calibrated answers and
interactive latency, at building scale.

### 2.2 Contributions mapped to the track
| track topic | OpenSNAP contribution | evidence |
|---|---|---|
| **Data models** | Object relation `Obj(oid, points, token, pooled_evidence, bbox, room, cracked)` materialised once per scan; building = union of room relations with a room partition | T11, T16 |
| **Semantics** | Open-vocab predicate `IS(c)` as a calibrated probability (product-of-experts of segmenter token and dense CLIP evidence; vocabulary relay for unseen words); tuple-independent probabilistic semantics for AND / OR / NOT | T1/T2, T3, T4 |
| **Query languages** | OVQL: `SELECT WHERE IS('chair') AND NEAR(IS('whiteboard'), 0.5) LIMIT k`; concept, spatial (NEAR, ABOVE), boolean operators | T3, T16 |
| **Query processing** | Two-phase plan (score, then refine), negative cache of refined concepts, room-partitioned batched spatial join (exact vs per-room: 5149/5149) | T11, T16 |
| **Adaptive indexing** | Semantic cracking: low-confidence answers are refined by a query-time operator (language-as-click-oracle, or the trained text decoder) and written back into the relation | T12/T14, T12b, T12c, T13 |
| **Uncertain data** | Probabilities, AP over ranked answers, confidence-gated refinement (γ, θ sweeps) | T13 |
| **Multi-modal / vector data** | Per-object CLIP vectors, text → vector predicates, VLM-in-the-loop naming | T6/T7 |
| **Robustness of the data model** | Dense-feature collapse guard (out-of-distribution ingest); phone-video ingest path | T18, §8 |

### 2.3 Scope-justification paragraph (≤ 1000 chars, draft)
> OpenSNAP is a data-management system for 3D scans: it defines an object relation that is materialised once per scan,
> an open-vocabulary query language (OVQL) with probabilistic semantics for concept, spatial and boolean predicates, and
> query processing techniques (a two-phase plan, room-partitioned spatial joins, and semantic cracking that refines and
> stores answers lazily). Learned models act as operators inside this system. We evaluate query accuracy (instance AP),
> latency (0.8 ms vs 8.8 s re-processing), storage (6.7× compression) and building-scale workloads on seven datasets.

### 2.4 Suggested 12-page outline and table budget
| section | pages | content | tables / figures |
|---|---|---|---|
| 1 Introduction | 1.25 | problem, why re-running a network per question fails, contributions | Fig 1: system overview (`assets/video/opensnap_explainer.gif` frames) |
| 2 Data model & semantics | 1.5 | object relation, IS(c) probability, relay, tuple independence | — |
| 3 OVQL | 1 | grammar, operators, examples | Fig: query examples (`assets/building_hits.png`) |
| 4 Query processing | 1.5 | two-phase plan, partitioned join, cracking | algorithm box |
| 5 Ingest operators | 1.25 | proposals, corrective clicks, naming, video ingest + collapse guard | Fig: `assets/demo_lab_desk/click_naming.jpg` |
| 6 Experiments | 4.5 | RQ1–RQ6 below | T1/T2, T2b, T3, T4+T4b–d (merge), T5f (condensed), T6/T7, T11, T12c, T16, T18 |
| 7 Related work | 0.75 | 3D open-vocab, promptable segmentation, DB for unstructured/multimedia, cracking, probabilistic DBs | — |
| 8 Conclusion | 0.25 | | |

ICDE readers care most about T3, T11, T12/T12c, T16 (systems), then T4-family (query accuracy). Put the
segmentation-only tables (T5 family, T8) as compact evidence that the operators are strong, not as the headline.

### 2.5 Things a DB reviewer will ask (and the answer)
* *Why not a vector DB over per-point CLIP features?* — per-point features answer "where is chair-ness", not "which
  objects"; T1/T2 shows point-wise OpenScene at 52.0 vs object-anchored 58.2 mIoU, and instance AP needs objects (T4).
* *Is materialisation worth it?* — 10,936× speed-up per query, 6.7× smaller than the raw scan (T11); ingest 2.3 s encode
  + 6.6 s materialise per room is paid once.
* *Does cracking converge / cost accuracy?* — with the v4 refiner: 2nd-pass latency back to 3.7 ms, AP50 −0.2 (T12c);
  with the training-free refiner it costs AP (T12/T14: 35.2 vs 38.9) — report both.
* *Building scale?* — 68 rooms, 5,149 objects, 0.9–3.4 ms per query; partitioned join verified exact (T16).
* *What are the semantics of NOT / OR?* — tuple independence; document it as an assumption.

## 3. System and method inventory

| component | what it does | status |
|---|---|---|
| SNAP-C backbone + decoder | promptable masks, XYZ-only input (colour/normals zeroed, as in training) | frozen; reproduces published click numbers (T5) |
| Mask-anchored pooling + dense blend | per-object 3D CLIP evidence from OpenScene features | shipped (T1/T2) |
| Product-of-experts fusion | segmenter token × dense evidence | shipped |
| Vocabulary relay | maps unseen words through SNAP's training vocabulary | shipped; +2.0 on ScanNet and +0.6 on nuScenes for the full system (T1/T2, T2b) |
| Coarse-to-fine proposals | grid prompts refined only in uncovered regions | shipped (T10: 69.7% recall@0.5) |
| 3D-propose / VLM-choose naming | segmenter proposes top-5 names, Qwen2.5-VL picks by letter log-probs, cycle consistency | shipped (T6/T7) |
| Language-as-click-oracle | text relevance drives simulated clicks | training-free text → mask (T8) |
| v4 text decoder | trainable copy of SNAP's decoder for text prompts, 3D CLIP features in attention keys | shipped (T8, T8b), also cracking refiner (T12b/c) |
| OVQL engine | parser, two-phase plan, negative cache, partitioned joins, cracking | shipped (T3, T11–T16) |
| **Test-time token tuning** | 30 Adam steps on SNAP's mask vector over frozen voxel embeddings to honour positive and negative clicks | shipped (T5f, T5g, T5h) |
| **Corrective decoder (v5)** | positive-only clicks → exact SNAP; any negative → trainable decoder copy with click-type embeddings + previous-mask prompt | shipped, multi-domain checkpoint (T5f) |
| **Backbone LoRA adapters (v6)** | rank-16 adapters in every PTv3 attention layer, zero-init, feature distillation | experimental; outdoor gains, small indoor first-click cost (T5i, T5j) |
| **Click-jitter TTA** | average SNAP's logits over 8 jittered copies of the click set (10 cm) | indoor first-click gain (T5k) |
| **Video ingest** | frames → VGGT point maps → metric scale (Depth-Anything-V2 metric) → gravity alignment | `scripts/video2cloud.py` (§8) |
| **Dense collapse guard** | detects out-of-distribution dense features at ingest and falls back to segmenter tokens | `OpenSNAP.check_dense` (T18) |
| **Photo-grounded naming** | projects a 3D mask into the source video frames and names the real photo crop | `opensnap/photo.py` (§8) |

Rejected variants (keep as ablations — they explain design decisions):
* **v1** semantics added to point features + min-loss tokens: IoU@5 66.4 vs 74.0 (T5) — semantic features blur
  same-class instances (T5b: removing them collapses IoU@1 to 39.9).
* **v2** semantic keys + role tokens, fine-tuned on ScanNet: loses SNAP's multi-dataset generalisation (IoU@5 69.5 vs
  72.8 on the S3DIS subset; the decoder-only control also drops, 71.0).
* **v3** frozen SNAP with extra single-click tokens: parity only; oracle headroom +2 (T9) not realisable by the IoU head.

## 4. Experimental setup

| dataset | split / size | used for | protocol notes |
|---|---|---|---|
| ScanNet (20 cls) | val, 312 scenes | semantic seg (T1/T2, T15) | OpenScene label set |
| ScanNet200 | val 156 (queries), 104 (text) | queries T3, text → mask T8 | head / common / tail split 66/68/66; tail held out of text training |
| Matterport3D-160 | test, 406 regions | semantic seg (unseen classes), naming T6/T7 | 63 classes unseen by the segmenter |
| S3DIS | Area 5, 68 rooms | queries T3/T4, interactive T5/T5c/T5f, text T8b, systems T11–T16 | zero-shot for every trained component |
| Replica | 8 scenes | instance AP T4b, interactive T5f | instances transferred from habitat meshes (0.00 cm median NN) |
| nuScenes | official val, 6019 keyframes | LiDAR semseg T2b, instance AP T4c, interactive T5d/T5f | lidarseg labels, SNAP's 16-class map; instances = points in annotated boxes |
| SemanticKITTI | seq 08, 4071 scans | LiDAR semseg T2c, instance AP T4d, interactive T5e/T5f | panoptic labels; OpenScene has no KITTI model → dense branch is cross-dataset |
| Phone video (ours) | 1 video, 83 s, lab desk | in-the-wild demo, collapse guard T18 | VGGT-1B reconstruction; qualitative |

* **Metrics**: mIoU / mAcc (semantic); AP, AP50, AP25 over IoU 0.50:0.95 (instance / query); IoU@k, NoC@80/85/90
  (interactive); top-1 accuracy after CLIP name → label mapping (naming); latency (ms), storage (MB).
* **Interactive protocols**: *positive* (SNAP's): first click at the object centre, then centre of the missed region.
  *Corrective* (AGILE3D / Point-SAM style): each click at the centre of the larger of the missed / false-positive
  regions, labelled positive or negative. Crops: ±1.5 m around the object (SNAP Table 2).
* **Hyper-parameters** for every test-time method were selected on ScanNet val only (T5g, T5k); no tuning on any
  reported test set.
* **Noise**: naming ±3 points (212 clicks); interactive IoU ±0.5 between identical runs (Replica repeat: 55.8 / 58.6 /
  59.1 / 59.5 vs 56.3 / 58.9 / 59.5 / 60.1). Treat differences below these as ties.
* **Hardware**: 40 GB MIG (3g.40gb) slices for evaluation; training on one 80 GB GPU.

## 5. Results by research question

### RQ1 — Does object-anchored evidence give better open-vocabulary semantics? (T1/T2, T15, T2b, T2c)
* ScanNet: 58.2 mIoU vs OpenScene 52.0 and SNAP's text token 41.5. Most of the gain comes from **mask-anchored
  pooling** (52.0 → 57.2); relay adds +2.0 on top of the full fusion (56.2 → 58.2).
* Matterport-160: 11.6 (unseen 5.7) vs OpenScene 9.5 (5.5) and SNAP 6.9 (1.9) — unseen gain is from relay + pooling.
* nuScenes (6019 keyframes): 32.0 vs 26.9 / 23.8. On OpenScene's own processed sweeps 34.2 vs 31.1. Our OpenScene number
  is below its published 36.7 because we evaluate single keyframes with our script — say this explicitly.
* SemanticKITTI: **loss** — 31.5 vs SNAP's 32.8 mIoU (mAcc 45.8 vs 43.7 is a win). Cause: no KITTI OpenScene model; the
  nuScenes model applied cross-dataset collapses on ground classes (terrain 0.0, parking 0.1 IoU). OpenSNAP still gains on
  objects (other-vehicle 2.9 → 11.9). Lesson: the dense branch helps only when a dense model exists for the sensor/domain.
* Dense source matters (T15): with Matterport-trained OpenSeg features on ScanNet, OpenScene drops to 37.3 but OpenSNAP
  keeps 47.6 — the segmenter token makes the system robust to a mismatched dense model.

### RQ2 — Query accuracy of OVQL and instance segmentation (T3, T4, T4b, T4c, T4d)
* S3DIS (13 classes, SNAP Table 4 setting): 15.4 AP (c2f) vs 10.5 for SNAP's own text token under the same protocol.
  SNAP's published 16.1 uses its own evaluation; do not claim a win over the published number.
* Replica: 19.9 / 25.9 AP / AP50 vs SNAP 14.8 / 19.7 (ours) or 10.1 / 11.7 (published). This is the *query workload*
  (classes present in the scene are asked); it is **not** comparable to OpenMask3D's closed-set 13.1 / 18.4. Under the
  closed-set protocol (C1) OpenSNAP XYZ-only gives 12.9 / 15.1 and with the image column 16.9 / 21.8.
* nuScenes: 13.9 AP with cracking vs 7.9; SemanticKITTI 20.6 vs 13.9. SNAP's text head was trained on these exact
  classes, so these are wins against an in-domain baseline.
* Relational queries (NEAR) also improve: S3DIS 14.5 vs 12.2 AP; nuScenes 17.0 vs 13.3.
* ScanNet200 tail: OpenSNAP 2.5 vs 4.4 AP for the segmenter token (T3) — the segmenter's head was trained on all 200
  classes including tail, so it is in-domain there; report honestly.

### RQ3 — Naming clicked objects (T6/T7)
* 3D-propose / VLM-choose with Qwen2.5-VL-7B: 40.1% (unseen 17.2%) vs segmenter text head 30.7% (12.2%) vs VLM free
  naming 18.9% (6.5%). The VLM alone is weak on point-cloud renders; the 3D side alone lacks recognition; combining
  them wins.
* Closed-vocabulary upper references: 39.2–41.0% — the open-vocab system is at the closed-set level.
* Noise ±3 points (212 clicks) — claim the ordering, not the third digit.

### RQ4 — Text → mask (T8, T8b)
* ScanNet200: training-free language-as-click-oracle 11.8 mIoU; the v4 text decoder reaches 18.2 at 16k steps (head
  32.8 / common 13.8 / tail 2.7). Tail (held out of text training) stays hard.
* S3DIS zero-shot: 27.2 vs 20.0 (chair 79.0, table 51.4). Flat classes (board, beam, column) are 0 for both — geometry
  cannot separate them from walls (XYZ-only input).
* v3 (frozen decoder, extra tokens) could not learn text at all (1.1 mIoU) — motivates v4's separate decoder.

### RQ5 — Interactive segmentation beyond SNAP (T5, T5c–T5k)
* **Parity**: our click path reproduces SNAP (S3DIS 53.3 / 69.1 / 74.0 vs published 53.6 / 71.1 / 77.6; crops 56.1 /
  72.0 / 76.8 vs 56.6 / 73.8 / 80.9). Positive-only click sets still go through the exact SNAP decoder.
* **SNAP's failure mode**: under the corrective protocol SNAP wastes every negative click and plateaus (Replica 59.5 →
  60.1 from 5 to 10 clicks; S3DIS rooms 69.0 → 70.5).
* **Test-time token tuning** (training-free): IoU@5 / IoU@10 gains over SNAP under identical clicks — S3DIS crops
  +8.7 / +14.2, S3DIS rooms +9.5 / +14.2, Replica +15.3 / +21.8, nuScenes +8.5 / +13.0, SemanticKITTI +3.4 / +6.1.
  Chosen on ScanNet only (T5g); 43 ms per click (T5h).
* **Corrective decoder**: ScanNet-only training overfits indoors (weaker than token tuning on held-out and outdoor
  data); multi-domain training fixes most of it and is best on nuScenes / SemanticKITTI when combined with token tuning
  (nuScenes 73.4 / 79.2, NoC@90 4.78; SemanticKITTI 71.9 / 76.9, NoC@90 4.72). 16 ms per click.
* **Backbone adapters** (T5i, T5j): first real change to SNAP's features. Positive clicks only: nuScenes +3.5 IoU@1 (53.5
  → 57.0) and +3.3–4.0 at every click count; SemanticKITTI +0.8; S3DIS rooms neutral (53.3 = 53.3, later clicks +0.5);
  S3DIS crops −1.9 IoU@1 (later clicks +0.8–1.2). Adapters help where their training domains are and cost a little on
  held-out indoor first clicks → recommend enabling them for Outdoor only (SNAP has an explicit domain input).
* **Click-jitter TTA** (T5k): +1.1 IoU@1 on the ScanNet selection set, +1.5 on held-out S3DIS crops (best crops row
  overall: 57.6 / 74.4 / 82.2 / 89.0, NoC@90 5.82); no effect on nuScenes (10 cm is small relative to sparse LiDAR).
* **vs. published interactive methods** (S3DIS crops): AGILE3D 58.7 / 77.4 / 83.6 and Point-SAM 45.9 / 77.6 / 84.6
  (IoU@1/3/5). OpenSNAP best 57.6 / 74.4 / 82.2: within 1.1–1.4 of AGILE3D at 1 and 5 clicks, still 3.0 behind at 3
  clicks. Do not claim to beat AGILE3D; claim closing most of SNAP's gap while staying a single promptable model across
  indoor + outdoor.
* Negative-mask subtraction (decode the negative click alone, remove its mask) is a weak baseline (crops IoU@5 75.3).

### RQ6 — Systems (T11, T12/T14, T12b, T12c, T13, T16)
* Ingest: 2.3 s encode + 6.6 s materialise per S3DIS room; query on the stored relation 0.8 ms vs 8.8 s re-processing
  (10,936×); relation 1.98 MB vs 13.3 MB raw XYZ (6.7×).
* Building: 68 rooms / 5,149 objects as one database, catalog 171 MB vs 945 MB raw, loads in 0.29 s; IS 1.41 ms (0.87
  with LIMIT 10), NEAR 3.40 ms, ABOVE 2.95 ms, OR/NOT 2.08 ms; partitioned join identical to per-room evaluation.
* Cracking: with the training-free click-oracle refiner, repeated queries get cheap (2nd pass 1.7 ms) but AP50 drops
  38.9 → 35.2 (T12/T14). With the trained v4 refiner + dedup: 2nd pass 3.7 ms vs 26.8 ms refine-every-time, AP50 38.7 vs
  38.9 (T12c). Calibration sweep (T13): confidence-weighting γ and trigger θ trade refinements for AP; dedup ≥ 0.3
  recovers AP50.
* Outdoor cracking helps accuracy: nuScenes +1.2 AP, SemanticKITTI +2.6 AP (T4c, T4d); on Replica it is neutral (T4b).

### RQ7 — In the wild: phone video → queryable scene (T18, §8)
* VGGT reconstruction of an 83 s phone video → 1.09 M points, metric scale from a monocular metric-depth model
  (per-frame spread 13.9%), gravity from the dominant up-facing plane.
* The dense 3D CLIP model collapses on this reconstruction (80% of object voxels → "stair rail"); every benchmark scan
  stays ≤ 0.66 (38 scans). The guard (threshold 0.75) switches OpenSNAP to segmenter tokens for relevance and naming.
* Photo-grounded naming projects the 3D mask into the source frames (VGGT cameras, depth-tested occlusion) and shows
  the VLM a real photo crop: correct names for chair, monitor, bottle, PC-tower side ("vent"), desk, printer; errors
  "browser" (laptop screen content), "screwdriver" (PC-case box). Text queries are weaker on this scan (office chair ✓;
  keyboard, printer, water bottle ✗) because they rely on relevance without dense features.
* Qualitative only (one video); claim "works end-to-end on a phone video", not accuracy.

## 6. Ablations and negative results worth reporting
| finding | where | why it matters |
|---|---|---|
| Fine-tuning SNAP's decoder on one dataset loses cross-dataset generalisation | T5 (v1/v2 rows) | justifies keeping SNAP frozen for positive clicks |
| Semantic features inside mask embeddings blur instances | T5b | justifies keeping semantics out of the mask path |
| Extra single-click tokens have little oracle headroom (+2.0) | T9 | single-click ambiguity is not the bottleneck; corrective clicks are |
| Relay hurts the token-only variant on ScanNet (41.5 → 40.7) but helps the full system (56.2 → 58.2) and out-of-domain data | T1/T2, T2b | relay is for unseen vocabulary |
| Training-free cracking costs AP; trained refiner removes the cost | T12/T14 vs T12c | refiner quality determines cracking's value |
| Cross-dataset dense features can hurt (SemanticKITTI stuff classes) | T2c | limits of the dense branch; motivates the collapse guard |
| Corrective decoder trained on one domain overfits | T5f | multi-domain training is required |
| Backbone adapters: in-domain gains, small held-out indoor cost | T5i | domain-gated adapters |
| Jitter TTA is radius- and density-dependent | T5k | indoor-only default |

## 7. Comparisons to published numbers — protocol caveats (put these in the paper)
* SNAP published S3DIS click numbers (53.6 / 71.1 / 77.6 / 83.2) are higher than our reproduction at 10 clicks (77.4) —
  click placement differs; all our comparisons use one protocol for every method.
* SNAP@auto S3DIS AP 16.1 (published) uses the authors' evaluation; ours 15.4 vs their model's 10.5 under our protocol.
* OpenMask3D Replica 13.1 AP is the closed-set 51-class protocol; compare it only with C1 (OpenSNAP 12.9 XYZ-only, 16.9 with
  the image column), never with the T4b query workload.
* OpenScene nuScenes 36.7 (published) vs 26.9 (our run, single keyframes, our script).
* AGILE3D / Point-SAM numbers are copied from their papers (crop protocol); their click placement may differ slightly.

## 8. Phone-video demo (artifact)
* Inputs / outputs: `assets/demo_lab_desk/` — `lab_desk.mp4` (720p re-encode of the 179 MB original), `frames.jpg`
  (8 source frames), `cloud.jpg` (3 views of the reconstruction), `click_naming.jpg` (3D mask, what the VLM sees, the
  name; 8 objects), `click_naming.txt` (raw VLM replies), `viewer.jpg` (segment everything, open-vocab labels, 4 text
  queries incl. misses).
* Reproduce: `python scripts/video2cloud.py --video <clip.mp4> --conf_pct 65 --max_depth 2.5` then
  `python scripts/app.py --scene ../downloads/demo/<clip>` and `python scripts/make_video_demo.py`.
* Assumptions: indoor, static, phone roughly upright at the start, a level surface in view, enough parallax.

## 9. Figure inventory
| figure | file |
|---|---|
| System explainer video / GIF (Manim) | `assets/video/opensnap_explainer.mp4`, `.gif` |
| Viewer demo (Replica) | `assets/video/viewer_demo.mp4`, `.gif` |
| 17-scene open-vocab gallery (ScanNet, S3DIS, Replica, Matterport) | `assets/openvocab_gallery/*.jpg` |
| Outdoor LiDAR labelling (nuScenes) | `assets/outdoor_gallery.jpg` |
| Building-level query hits | `assets/building_hits.png` |
| Phone-video demo | `assets/demo_lab_desk/*` |

Figures still worth making for the paper: architecture diagram (vector), latency/storage bar chart (T11, T16),
IoU-vs-clicks curves for T5f (SNAP vs token tuning vs decoder, per dataset), cracking latency over a workload (T12c).

## 10. Threats to validity / limitations
* Positive-only clicking is SNAP's; segmentation gains come from negative clicks, test-time methods and adapters.
* Several baselines (SNAP's text head on nuScenes / SemanticKITTI / ScanNet200) are in-domain; held-out datasets
  (S3DIS, Replica, Matterport-160 unseen) are the fairest tests.
* Flat objects on walls/floors (boards, rugs) are hard with XYZ-only input.
* Truly novel concepts remain hard; most unseen-class gains come from near-synonyms of seen classes.
* The collapse-guard threshold was set from 38 benchmark scans and one phone video.
* The phone-video demo is a single qualitative example.
* Checkpoints (text decoder, corrective decoders, adapters) are kept local (`downloads/checkpoints/`, SHA256SUMS), not
  in the repository.

## 12. Comparative study and response to the review

The full comparative study against Search3D, OVI-MAP and HouseLayout3D, and the point-by-point answers to the review
(calibration, sensitivity, cracking workload, cross-room joins, attribute queries, scaling, collapse guard, multi-capture
video), is in **[COMPARISON.md](COMPARISON.md)** with tables C1–C13 in `TABLES.md`. Short version:

* **Ahead (same protocol):**
  * Strict-IoU instance quality vs OVI-MAP (Replica AP 16.9 / 12.9 vs 8.5; ScanNet AP 9.8 vs 7.2).
  * Relational queries vs phrase retrieval (C9).
  * Calibration (C5).
  * Building and campus joins (C4, C8).
* **Behind:**
  * mIoU / AP25 vs OVI-MAP (coverage).
  * ScanNet200 long tail vs Search3D (6.1 vs 23.0 AP, XYZ only).
  * OpenScan attributes (2.7 vs 9.9 mean AP).
  * RGB-only video (metric scale off 20–42%).
* **Method changes:**
  * image-evidence column;
  * admission control for cracking (fixes a 60% AP drift under long workloads);
  * level-scoped halo joins;
  * cost-based pushdown join;
  * 234-word calibration background.

## 11. Reproduction (key commands)
```bash
python scripts/eval_semseg.py --dataset scannet --split val                      # T1/T2/T15 (+ nuscenes, semantickitti)
python scripts/eval_queries.py --dataset s3dis --keep_all 1 --proposals c2f      # T3/T4 (+ replica, nuscenes, semantickitti)
python scripts/eval_interactive.py --dataset s3dis --crops 1 --protocol pn --method ttt [--tta 8]   # T5f/T5k
python scripts/eval_interactive.py ... --method v5ttt --corr_ckpt ../downloads/checkpoints/opensnap_corrective_v5_multi.pth
python scripts/train_corrective.py --dataset scannet,nuscenes,semantickitti --steps 9000
python scripts/train_adapt.py --steps 6000                                       # T5i/T5j adapters
python scripts/eval_naming.py --dataset matterport160 --split test --prior snap+lvis   # T6/T7
python scripts/eval_systems.py; python scripts/eval_building.py                  # T11–T16
python scripts/video2cloud.py --video clip.mp4; python scripts/make_video_demo.py       # §8
python scripts/make_tables.py                                                    # regenerate TABLES.md
```
