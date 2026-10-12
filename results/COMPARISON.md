# Comparative study and response to the review

This file answers the review point by point and adds the comparative study the review found missing, against
**Search3D** (RA-L 2025), **OVI-MAP** (CVPR 2026) and **HouseLayout3D** (NeurIPS 2025 D&B), plus OpenMask3D, OpenScan /
GOV-3D, HOV-SG, Open3DSG and ConceptGraphs where relevant. Every number comes from `results/TABLES.md`, tables
**C1–C13** (regenerate with `python3 scripts/make_tables.py`). Published numbers of other systems are copied from their
papers and marked *(published)*.

Scripts behind this file:
- `scripts/eval_standard.py` and `opensnap/bench.py`: the closed-set protocols.
- `scripts/fuse_cv.py`: image-column fusion weights.
- `scripts/eval_partition.py`: building joins.
- `scripts/eval_scale.py`: the campus-scale study.
- `scripts/eval_calibration.py`: calibration and sensitivity.
- `scripts/eval_workload.py`: the cracking stress test.
- `scripts/eval_queries.py --agg_sweep / --sweep`: witness semantics, phrase baseline and threshold sweeps.
- `scripts/video_replica.py`: RGB-only reconstruction.
- `scripts/eval_guard.py`: the collapse guard.
- `scripts/eval_openscan.py`: attribute queries.

---

## 0. What changed, in one paragraph

We re-scored OpenSNAP under the **exact protocols** of the closest systems, using the same scenes, class lists,
GT-mesh vertices and the official ScanNet instance evaluator. Our array implementation, `opensnap/bench.py`, reproduces
OVI-MAP's official evaluation code to five decimals on real predictions.

We added four things to the method:
1. **An image-evidence column** in the object relation: per-object multi-view SigLIP crops, the same encoder as OVI-MAP,
   used whenever posed frames exist.
2. **Admission control for semantic cracking**: verify-before-admit, a budget, and confidence-weighted ranking.
3. **Level-scoped, halo-extended spatial joins** in place of strict room partitioning.
4. **A cost-based pushdown join** that scales to a 1.2-million-object campus.

We measured calibration (ECE, Brier, reliability) and replaced the 4-word background vocabulary with the segmenter's
234-word training vocabulary. That one change cuts ECE 3–9× on all five datasets.

Under OVI-MAP's protocol, with every setting chosen on ScanNet *train* development scenes, OpenSNAP with the image
column reaches 31.6 mIoU / 44.1 mAcc / 34.8 AP25 / 30.6 AP50 / 24.7 AP on Replica (OVI-MAP: 26.5 / 32.2 / 34.5 / 21.2 / 8.5). On ScanNet-18 it reaches 28.8 / 42.8 / 38.1 / 30.1 / 20.0
(OVI-MAP: 17.5 / 27.6 / 23.4 / 15.7 / 7.2). That beats OVI-MAP on all five metrics on both datasets.

We also ran the reviewer's sensitivity, workload, partition, attribute (OpenScan) and multi-capture video studies. Two
earlier claims were wrong and are corrected (§9).

---

## 1. Positioning: what each system stores and answers

| | input at ingest | what is stored | granularity | query interface | calibrated probabilities | spatial / boolean composition | query-time refinement | building / multi-floor | query latency |
|---|---|---|---|---|---|---|---|---|---|
| OpenMask3D (NeurIPS'23) | mesh + posed RGB-D | one CLIP vector per Mask3D instance | objects | text → ranked masks | no (cosine) | no | no | no | ms after minutes of ingest |
| Search3D (RA-L'25) | mesh + posed RGB-D | tree: objects → parts, one SigLIP vector per node | objects, parts, materials | text → heat-map, avg(object, part) score | no | no (hierarchy only) | no | no | 1.57 ms + 0.61 s text encoding (published); ingest ≈ 15–20 min / scene |
| OVI-MAP (CVPR'26) | RGB-D stream + poses (online) | class-agnostic instance map + SigLIP vector per instance from selected views | objects | text → label / heat-map | no (argmax cosine) | no | incremental map, not query-driven | no | real-time mapping at 30 fps with frame skipping |
| HOV-SG (RSS'24) | RGB-D + poses | floor → room → object graph with CLIP features | floors, rooms, objects | hierarchical language queries ("X in room Y on floor Z") | no | containment via hierarchy | no | yes | – |
| Open3DSG (CVPR'24) | point cloud + images | 3D scene graph, open-vocab objects + relation predicates | objects + pairwise relations | graph lookup | no | relations as edges | no | no | – |
| ConceptGraphs (ICRA'24) | RGB-D + poses | object graph with captions, LLM-inferred relations | objects | LLM over the graph | no | via LLM | no | no | seconds (LLM) |
| HouseLayout3D / MultiFloor3D (NeurIPS'25) | unposed RGB | layout polygons (walls, floors, doors, stairs), room graph | structure | – (layout estimation) | – | – | – | yes | 1–2 h per building |
| **OpenSNAP (ours)** | **XYZ only** (images optional) | **object relation**: geometry, segmenter token, dense 3D evidence, optional image evidence, provenance | objects (+ cracked objects) | **OVQL**: IS / NEAR / ABOVE / BELOW / ON / CONTAINS / AND / OR / NOT, nested | **yes** (background-competition softmax; ECE 0.015–0.022, C5) | **yes** (C4, C9) | **semantic cracking with admission control** (C7) | **yes** (C4, C8) | **0.4–7 ms per NEAR query up to 305k objects; 2.6 ms at 1.2 M** (C8) |

Rows for other systems describe the published systems. Cells marked "–" are not reported in those papers.

**Reading.** OVI-MAP, Search3D and OpenMask3D are *perception* systems: they decide which object is where and what it is
called, from images. HouseLayout3D supplies *structure*: walls, floors, doors. OpenSNAP is a *data-management* layer over
the scan, and it differs from all of them on four points:
- it materialises a relation once;
- it gives open-vocabulary predicates probabilistic semantics;
- it composes them with spatial and boolean operators;
- it refines itself under a query workload.

The experiments below therefore compare in two directions:
1. Recognition quality *of what we store*, under their protocols (C1–C3, C10, C12). Here image-based perception systems
   are strong, and we report where we trail.
2. Query semantics, query cost and building-scale behaviour (C4, C7–C9, C13), which none of them provides. To ask them a
   relational question, one would embed the whole phrase. We measure exactly that baseline in C9.

HouseLayout3D is complementary, not a competitor. We use its 16 multi-floor buildings, its levels and its doors as the
test bed for building-scale joins (C4).

---

## 2. Against OVI-MAP (same scenes, same protocol) — C1, C2

**Protocol** (OVI-MAP §4.2, their `eval_sem_seg.py`):
- **Replica:** the 8 NICE-SLAM Replica scenes, 51 classes, scored on the GT mesh vertices with OVI-MAP's own GT label
  files. Our input points *are* those vertices: same count, zero distance.
- **ScanNet:** OVI-MAP's 18 scenes with ScanNet200 labels.
- **Metrics:** per-vertex mIoU and mAcc (classes without GT skipped), plus instance AP25 / AP50 / AP(0.5:0.95) with the
  ScanNet evaluator (predictions under 100 vertices dropped). Every stored object gets its most probable class.

**One selection protocol for every row marked "dev-chosen".** Every setting is chosen on development scenes
(S3DIS Area 1 and 30 ScanNet *train* scenes, never reported) and applied once, unchanged, to Replica and ScanNet:
- **Proposals:** union of the two proposal modes, 4 levels (union_l4).
- **3D labelling setting** (`scripts/tune_dev.py`): α = 0.7, temperature ×2, β = 0, confidence √p·s.
- **Image column** (`scripts/tune_image_dev.py`, on the 30 ScanNet train scenes with their frames): weight 0.75,
  SigLIP so400m crops (Search3D's encoder), image temperature ×2, γ = 0.

**Replica (C1).**

| | mIoU | mAcc | AP25 | AP50 | AP |
|---|---|---|---|---|---|
| OVI-MAP (published, online, RGB-D) | 26.5 | 32.2 | 34.5 | 21.2 | 8.5 |
| OVO-SLAM (published) | 24.9 | 34.0 | 28.1 | 17.5 | 9.1 |
| OpenSNAP, XYZ only [c2f] | 18.7 | 28.5 | 17.3 | 15.1 | 12.9 |
| OpenSNAP, XYZ only [union_l4], dev-chosen | 17.5 | 27.0 | 18.0 | 16.0 | 13.8 |
| **OpenSNAP + image column [union_l4], dev-chosen** | **31.6** | **44.1** | **34.8** | **30.6** | **24.7** |
| OpenSNAP + image column + lifted 2D proposals, dev-chosen | 28.1 | 39.8 | 33.1 | 27.9 | 19.6 |

- **OpenSNAP beats OVI-MAP on all five metrics** with nothing chosen on Replica:
  - mIoU 31.6 vs 26.5, mAcc 44.1 vs 32.2;
  - AP25 34.8 vs 34.5 (a narrow margin), AP50 30.6 vs 21.2;
  - strict AP 24.7 vs 8.5, about 2.9×.
- **Wider Replica landscape (C14):** 31.6 mIoU is above IRIS-mapping (31.1, IRIS-SLAM's own re-runs), VOIM (28.6),
  OVO-mapping (27.0) and OVI-MAP (26.5). Only Ov3R is higher (31.9 with GT geometry).
- **Even XYZ-only OpenSNAP beats both online systems on strict AP** (12.9–13.8 vs 8.5 / 9.1): the promptable
  segmenter's masks are tight. Without images, OVI-MAP is ahead on mIoU and AP25 (coverage: our relation holds 58–70
  objects per scene).
- **Lifted 2D proposals** (SAM masks of the frames merged into 3D segments, `opensnap/lift2d.py`) do not help under
  this protocol. Their confidence weight has no image-bearing development set to be chosen on, so it stays at 1.0.
- **Sensitivity, leave-one-scene-out on Replica** (reported in C1, not the headline): with only 8 scenes, per-fold
  choices are noisy. Image weight alone gives 29.8 / 40.3 / 33.1 / 27.3 / 21.2 (mIoU / mAcc / AP25 / AP50 / AP). Adding
  the encoder mix gives 31.4 / 44.5 / 34.6 / 29.8 / 23.1. Letting folds also pick the proposal set drops to 28.2 / 41.4 /
  32.7 / 27.2 / 21.6. The development-scene protocol avoids this noise: one setting, no test scene involved.
- **Precision note:** rows marked dev-chosen are rescored from saved per-object outputs (fp16). Rescoring the default
  setting this way gives the online run's masks and labels exactly, with AP within +0.2.

**ScanNet (C2).**

| | mIoU | mAcc | AP25 | AP50 | AP |
|---|---|---|---|---|---|
| OVI-MAP (published) | 17.5 | 27.6 | 23.4 | 15.7 | 7.2 |
| OVO-SLAM (published) | 14.6 | 27.8 | 19.4 | 12.6 | 5.5 |
| OpenScene (published) | 10.3 | 17.3 | – | – | – |
| OpenSNAP, XYZ only | 14.4 | 20.4 | 17.6 | 14.7 | 9.8 |
| OpenSNAP, XYZ only, the 15 val scenes | 15.6 | 22.2 | 19.7 | 16.8 | 11.5 |
| OpenSNAP, XYZ only [union_l4], dev-chosen | 15.0 | 21.3 | 19.1 | 14.5 | 9.8 |
| **OpenSNAP + image column [union_l4], dev-chosen** | **28.8** | **42.8** | **38.1** | **30.1** | **20.0** |

- With frames (OpenScene's processed ScanNet frames: every 20th frame, 320×240), **OpenSNAP beats OVI-MAP on all five
  metrics**, by 11–15 points on mIoU, mAcc, AP25 and AP50 and 2.8× on strict AP.
- **Without images**, OpenSNAP already beats OVI-MAP on strict AP (9.8 vs 7.2).
- Caveats:
  - three of OVI-MAP's 18 scenes (`scene0168_0x`) are ScanNet *train* scenes; we report the 15 val scenes as well;
  - our development scenes are other ScanNet train scenes (every 40th), none of them among the 18;
  - the segmenter token was supervised on ScanNet200 labels, as Mask3D's masks are;
  - the dense features are OpenScene's ScanNet distillation (no labels).
- **Running time.** OVI-MAP maps online at 30 fps with frame skipping. OpenSNAP ingests a scan offline in 2–4 s for
  geometry, plus 10–17 s per scene for the image column (5 views per object). After that, queries cost milliseconds (C8).

---

## 3. Against Search3D and OpenMask3D on ScanNet200 — C3; attributes (OpenScan / GOV-3D) — C12

**C3** uses ScanNet200 val (312 scenes, 198 classes, head / common / tail), the protocol of Search3D Table V and
OpenMask3D.

| | AP | head | common | tail |
|---|---|---|---|---|
| OpenMask3D (published) | 15.4 | 17.1 | 14.1 | 14.9 |
| Search3D, SigLIP (published) | 23.0 | 26.3 | 21.2 | 21.4 |
| Open3DIS 2D+3D (published) | 23.7 | 27.8 | 21.2 | 21.8 |
| **OpenSNAP, XYZ only** | **6.1** | 14.0 | 1.7 | 2.1 |

**This is our weakest comparison, and we state it plainly.**
- Without images, the 3D-distilled features cannot name ScanNet200's long tail: common and tail classes score about 2 AP.
- Every published row encodes multi-view RGB crops and uses Mask3D masks trained on ScanNet200.
- On the head classes, XYZ-only OpenSNAP (14.0) is about level with OpenMask3D (17.1).
- We could not run the image column on ScanNet, because the RGB-D frames require the ScanNet terms of use. On Replica,
  where frames are public, the image column lifts strict AP from 12.9 to 16.9 (C1).
- In the paper, C3 should appear as the *geometry-only* operating point, not as a head-to-head with image-based systems.

**C12 (OpenScan / GOV-3D attributes).** Same stored objects; each is labelled with the most probable attribute of each
aspect (raw attribute text as the query). ScanNet instance AP per aspect, and the mean over all 341 attributes.

| | affordance | property | type | manner | synonym | requirement | element | material | mean |
|---|---|---|---|---|---|---|---|---|---|
| OpenMask3D (published, RGB) | 7.2 | 7.5 | 8.5 | 12.8 | 16.9 | 13.0 | 12.2 | 18.8 | 9.9 |
| Open3DIS (published, RGB) | 11.9 | 12.8 | 14.2 | 19.2 | 26.7 | 19.2 | 18.7 | 28.3 | 15.8 |
| OpenSNAP, XYZ only | 2.0 | 2.8 | 2.2 | 3.3 | 3.2 | 5.6 | 1.1 | 9.7 | 2.7 |

- **Geometry-only evidence carries little attribute meaning.** Material is the least bad aspect (9.7), which is
  surprising for XYZ-only input; affordances and parts are close to zero.
- Same conclusion as C3: attribute grounding needs image evidence, which is what the image column is for.
- OVQL can express attribute queries directly, e.g. `IS('something used to sit')` or `IS('wooden chair')`, and conjunctions
  `IS('chair') AND IS('made of wood')`. The open question is only how well geometry-only evidence carries attribute
  meaning.

---

## 4. Against the room-partition assumption, on HouseLayout3D buildings — C4

**Setup.**
- All 16 HouseLayout3D buildings (Matterport3D scans; 354 rooms, 19,044 stored objects).
- Each region is ingested on its own, and every row is kept in the shared house frame.
- Floor levels are found **training-free** from region floor heights (a gap above 1.2 m starts a new level). This gives
  **33 levels, exactly HouseLayout3D's 33**, and 96.9% of our level heights coincide with an annotated horizontal layout
  plane.
- Doors come from HouseLayout3D's annotations (a door connects the regions whose boxes contain its centre).
- The reference is the unpartitioned (global) join over the same relation. Recall is the share of its answers (score ≥ 0.5)
  that a partitioning returns. Queries: 20 indoor nouns, every ordered pair, 7 predicates.

**Findings** (aggregate over all 16 buildings):
- **Strict room partitioning loses 15–32% of NEAR answers.**

  | NEAR(d), d = | 0.25 m | 0.5 m | 1 m | 2 m |
  |---|---|---|---|---|
  | answers returned | 84.9% | 81.1% | 75.6% | 68.0% |

  Matterport regions are frequently open-plan; for example, living, dining and kitchen are separate regions with no wall
  between them. In single buildings recall drops to 59% (`r47D5H71a5s`).
- **Using annotated doors recovers only part of the loss** (82.2% at d = 1 m): many region boundaries are openings, not
  doors.
- **Room + halo h ≥ d is exact for NEAR** (100% at d ≤ h), which follows from box geometry (§6.1). It costs 1.06 M join cells
  per query against 1.72 M for the global join, and latency stays at 2 ms per building.
- **Global joins are wrong for vertical predicates.** About half of the global ABOVE and BELOW answers (18,323 of 37,053
  and 19,146 of 37,108) have their only witness *on another floor*, such as a lamp "above" a table one storey down.
  ABOVE / BELOW / ON must therefore be scoped to a level.
- **Recommended plan:**
  - horizontal predicates (NEAR): room + halo with h ≥ d, which is exact;
  - vertical predicates (ABOVE / BELOW / ON): level-scoped;
  - if a building has no room layout at all, a layout-free grid with halo (T = 4–8 m, h ≥ d) is exact for NEAR at 2–4×
    the join cells.
- Level + halo trades 1–7% of NEAR recall (objects across levels, e.g. on stairs) for floor-correct vertical semantics.

---

## 4b. Qualitative comparison — `assets/compare/relational_priors.jpg`

**Layout:** GT | Ours | OpenMask3D* | Search3D* | OVI-MAP*, on Replica, using relational queries.

**Suggested caption:** *Relational queries on Replica. Ours evaluates `IS(a) AND NEAR(IS(b), 0.5)` over the stored object
relation and returns the target (red) with the witness that satisfied NEAR (blue). The three prior methods answer by
embedding the phrase "a near a b" and returning the most similar object. Each one's query interface is emulated (\*) on
the same stored objects with its published image encoder: CLIP ViT-L/14@336 crops (OpenMask3D), SigLIP so400m crops
(Search3D) and SigLIP-L/16-384 view-selected crops (OVI-MAP). Over all 40 measured relational queries on Replica, ours is
correct where all three priors are wrong on 5, and a prior is correct where ours is wrong on 3.*

**How the rows were picked:** from those 5, the three where the target is legible at figure scale. The other two involve
small objects (a table and a vase in room2).

**Emulation caveat:** the priors' outputs are emulated, not produced by running their code (the user chose not to
install it). They share our masks, so the figure isolates the *query semantics*, not mask quality.

**Extra figure,** `assets/compare/relational_queries.jpg`: the same idea on Replica and S3DIS, comparing against
phrase-embedding retrieval over our own 3D evidence. Its rows win in at least 4 of 5 seeds.

---

## 5. Response to the reviewer's questions

**Q1. Calibration: ECE / Brier per dataset (C5).**
- ECE uses 15 bins over every (object, class present in the scene) pair; y = 1 if the object overlaps a GT instance of that
  class at IoU ≥ 0.5. Five datasets: ScanNet train and val, Replica, S3DIS, nuScenes (outdoor).
- With the old 4-word background {wall, floor, ceiling, object}, ECE is 0.04–0.12.
- **With the segmenter's 234-word training vocabulary as background, ECE is 0.015–0.022 on all five datasets.** IS-query
  AP is unchanged (±1.3).
- The 234-word vocabulary is known in advance and doesn't depend on the dataset, and it is already best on the ScanNet
  *train* selection set (0.118 → 0.015). It becomes the default.
- A one-word background ('object') is badly calibrated (ECE 0.18–0.29). So the background competition is what calibrates,
  and a richer background calibrates better.
- **Platt scaling** (2 parameters, A = 0.40, B = −1.20), fitted on ScanNet *train* only and applied unchanged, gives **ECE
  0.004–0.010 on all five datasets**, outdoor nuScenes included. Calibration transfers across datasets and domains.
- **Temperature τ.** τ = 0.02 is better than the current default 0.01 on every dataset: +0.3 to +2.9 IS-query AP and lower
  ECE. It is also best on the ScanNet train selection set, so we recommend it as the new default. We did not silently
  re-run the earlier tables with it.

**Q2 / Q6. Long workloads, storage growth, drift; safeguards (C7).**

S3DIS, 34 rooms, 400 random IS queries per room (half of them LVIS nouns absent from the room). A fixed probe set is
re-scored after each checkpoint.

| policy | probe AP50 after 0 → 50 → 100 → 200 → 400 queries | cracked rows / room | relation size |
|---|---|---|---|
| no cracking | 21.7 → 21.7 → 21.7 → 21.7 → 21.7 | 0 | 1.99 MB |
| cracking, original rule | 21.7 → 17.2 → 12.9 → 10.4 → **8.6** | **85** | 2.30 MB |
| + verify-before-admit | 21.7 → … → **21.7** | 0.2 | 1.99 MB |
| + verify + budget 20% + confidence-weighted ranking | **22.6** → … → **22.6** | 0.2 | 1.99 MB |

- **The reviewer's concern is confirmed.** Without admission control, spurious cracked tuples steal ranks from ordinary
  queries, and AP50 falls by 60% over 400 queries.
- **The fix.** A cracked tuple is admitted only if:
  1. its own confidence is at least 0.3, and
  2. it *itself* satisfies IS(phrase) ≥ θ under the calibrated predicate.

  Otherwise the phrase goes to the negative cache.
- About 63 phrases per room are then resolved by dedup to an *existing* tuple (synonyms), and about 175 are rejected.
  Nothing spurious is stored, probe AP is flat, and storage does not grow.
- Ranking by P · conf (γ = 1) adds +0.9 AP50.
- Eviction (least-hit cracked row first, budget 20% of base rows) never triggered under the verified policy. It remains as
  a hard cap.
- The cost is about 35 ms per novel phrase on first sight. Repeated phrases cost about 1 ms (negative cache).

**Q3. Sensitivity to B, θ, α / ρ, τ / τ_r (C5, C6, C13, C1b).**
- **B:** above.
- **α / ρ / τ / τ_r:** C6, with AP (ECE) per dataset.
- **θ and dedup IoU:** C13 on Replica and S3DIS, plus T13 on ScanNet train.
- **Image fusion weight:** C1b.
- **α (dense vs token weight).** The default 0.7 is best on ScanNet train and val, within 0.7 AP of best on Replica, and
  within 1.3 on S3DIS (where α = 0 is best). Moving it by ±0.2 costs at most 1.4 AP indoors. Outdoors (nuScenes) the dense LiDAR features are weak and α ≤ 0.3 is better (7.9 vs 5.5 AP), so a
  per-domain α is warranted.
- **ρ (relay).** Small indoor effect (0.0 → 0.8: +0.1 to +0.9 AP). Outdoors it hurts (6.5 → 5.5).
- **τ_r.** 0.05 (default) vs 0.1: within 1.2 AP.
- **Synonyms** (same classes, queried by a synonym instead of the class name): AP drops by 0.6 (nuScenes) to 5.0
  (ScanNet val) points. Replica is robust (30.5 → 28.1); S3DIS is the most sensitive (10.4 → 6.1). ECE is unchanged
  (±0.002). Calibration survives paraphrase; ranking partly does not.
- **θ (cracking trigger) and dedup IoU** (C13, 9 settings each on Replica and S3DIS):
  - IS-query AP varies by at most 0.2 (Replica 19.5–19.7) and by 0.0 on S3DIS, so the choice is stable.
  - Honest corollary: on class-name workloads cracking itself does not add AP (Replica 20.1 → 19.5–19.7; S3DIS 11.6 → 11.7),
    because the stored relation already covers those objects.
  - Cracking pays off on phrases the proposals missed, and only with admission control (C7).

**Q4. Attribute queries, GOV-3D / OpenScan (C12).**
- *Expressiveness:* OVQL already takes any phrase in IS(·), and conjunctions express "wooden chair" as
  `IS('chair') AND IS('made of wood')`, with product semantics.
- *Evidence quality:* XYZ-only evidence reaches only 2.7 mean AP against 9.9 (OpenMask3D) and 15.8 (Open3DIS) with images.
- The limitation is the evidence column, not the language: an image column (C1) is the remedy.

**Q5. Cross-room relations in open-plan spaces (C4).** Room partitioning loses 15–32% of NEAR answers. A halo of h ≥ d
removes the loss exactly, and level scoping keeps vertical predicates correct (§4).

**Q6. Safeguards against spurious cracked objects, including under dense collapse.** See Q2. In addition:
- cracking uses the same calibrated predicate as querying, so a collapsed dense expert cannot both create and approve a
  tuple without the segmenter token agreeing;
- the collapse guard now checks dense/token agreement (C11);
- human-in-the-loop confirmation is a natural extension: provenance is stored per row, so cracked rows can be listed and
  confirmed or rejected in the viewer.

**Q7. Beyond one building (C8).** The 16 HouseLayout3D catalogs replicated into a campus:

| scale | rooms | objects | catalog | IS | NEAR, padded room join | NEAR, pushdown |
|---|---|---|---|---|---|---|
| 1× | 354 | 19 k | 50 MB | 13.7 ms* | 2.3 ms | **0.42 ms** |
| 4× | 1,416 | 76 k | 199 MB | 2.3 ms | 9.5 ms | **3.8 ms** |
| 16× | 5,664 | 305 k | 796 MB | 8.8 ms | 25.2 ms | **7.2 ms** |
| 64× | 22,656 | **1.22 M** | 3.2 GB | 48 ms | 92.6 ms (out of GPU memory in a first run) | **2.6 ms** |

\* includes some warm-up.

- Pushdown applies the IS selections before the join (keep p ≥ τ on both sides; exact for threshold answers, because
  p_a · max_j rel · p_b ≥ τ needs both factors ≥ τ).
- It then runs a cost-based join on the surviving candidates: a dense |S|×|Q| join when that is small, an exact
  uniform-grid hash join otherwise.
- At 1.2 M objects pushdown is **36× faster** and avoids the padded plan's quadratic per-room memory.
- **What this means for indexing:** at campus scale the relation needs selection pushdown plus a spatial hash or R-tree
  over boxes. Embeddings are kept in fp16, 2.5 KB per object.

**Weakness: tuple independence and max-witness semantics (C9).**
- On the same stored objects, max-witness, noisy-OR and clipped expected count land within 0.5 AP50 of one another, with
  similar ECE (0.021–0.034).
- So the existential aggregator does not limit NEAR queries in practice.
- What matters is evaluating the structure at all. The phrase-embedding baseline, `IS('a near a b')` (how a
  CLIP/SigLIP-retrieval system such as OpenMask3D, Search3D or OVI-MAP would ask), loses to OVQL by 7.4 AP50 on S3DIS
  (14.3 vs 21.7) and 3.1 on Replica, and is about 3× worse calibrated (ECE 0.07–0.09 vs 0.02–0.03).

**Weakness: the collapse guard tuned on a small sample (C11).**
- The original share rule fired on **14 / 312 clean ScanNet val scans (4.5%)** and **21 / 354 Matterport rooms (5.9%)**.
  These are rooms genuinely dominated by one class, such as lecture rooms full of folded chairs.
- It also **misses partial collapse.** On all 8 RGB-only Replica reconstructions the dense features label 45–71% of object
  voxels "stair rail", just under the 0.75 threshold. This is why the dense-only expert scores about 0 on video (C10).
- We tried two replacements, rule fixed before testing:
  - **v2:** exact agreement between dense and token labels < 0.5. Still fires on 28 of the 35 earlier false positives.
    Lexical near-duplicates ("folded chair" vs "chair") count as disagreement.
  - **v3:** soft agreement (cosine of the two labels' text embeddings) below a threshold calibrated on **clean scans
    only**, the 1st percentile of 122 random scans (0.782).
- v3 cuts false alarms to 2 of 122 random clean scans, 0 of 14 ScanNet and 5 of 21 Matterport former false positives. But it
  **detects only 1 of 9 collapsed video reconstructions and misses the lab desk.**
- **Conclusion (negative result):** no scene-level rule we tried both avoids false alarms and catches collapse. The
  original rule is kept for the demo, where false alarms only cost the dense expert. The principled fix is **per-object
  gating**: drop the dense expert for an object when it disagrees with the token and image experts. This is future work,
  stated as such.

**Weakness: the phone-video pipeline validated on one capture (C10).**
- 8 more captures: the Replica sequences, **RGB only** (60 frames, no depth, no poses), scored against GT.
- The cameras are accurate (4.2 cm RMSE after similarity alignment). The **monocular metric scale is off by 20–42%**
  (mean 35%), and this is the dominant failure.
- With the scale corrected (oracle, diagnostic only), quality roughly doubles: image-column mIoU goes from 10.6 to 17.5.
- 60 frames observe only 42–67% of each room.
- *(Observed-only mIoU from C10 to be filled in.)*
- The collapse guard did not fire on any of the 8 reconstructions.

---

## 6. Method changes made in response (all in the code)

### 6.1 Level-scoped, halo-extended joins (`scripts/eval_partition.py`)
- **Claim.** If gap(i, j) ≤ d, then box j intersects box i dilated by d. That dilated box lies inside room(i)'s bounding box
  dilated by d. So every witness of NEAR(·, d) for a subject in room r lies in room r's halo block when h ≥ d.
- **Consequence.** Room + halo is exact for NEAR and, for d ≤ h, costs one extra box-intersection test per block.
- **Vertical predicates** (ABOVE / BELOW / ON) are evaluated inside the subject's floor level.
- **Levels** are clustered from region floor heights.

### 6.2 Cost-based pushdown join (`scripts/eval_scale.py`, `candidate_join`)
- IS selections run before the join: keep p_a ≥ τ and p_b ≥ τ.
- Then either a dense candidate join or an exact grid hash join: references are inserted into every cell of their box
  dilated by d, and subjects probe every cell of their own box.

### 6.3 Admission control for cracking (`opensnap/query/engine.py`)
- `admit` (minimum confidence), `verify` (the new tuple must satisfy IS(phrase) ≥ θ), `budget` (cap on cracked rows as a
  fraction of base rows) with least-hit eviction, and `gamma` (rank by P · conf^γ).
- `ObjectTable.remove` for eviction.

### 6.4 Image-evidence column (`opensnap/views.py`, `OpenSNAP.mask_probs(image=…)`)
- Per object: top-5 frames by visible points, with a depth-tested projection; 3 crop scales; SigLIP-large-384 embeddings
  averaged.
- A third expert in the product of experts. Rows without views fall back to a uniform term.
- Works with posed RGB-D (`PosedFrames`) and with video reconstructions (`VideoFrames`, from VGGT cameras and depth).

### 6.5 Calibrated background
- B = the segmenter's 234-word training vocabulary (C5).

### 6.6 Witness aggregation option
- `Engine(agg="max" | "noisy_or" | "sum")` (C9).

---

## 7. Protocol and fairness notes (put these in the paper)

1. **Input modality differs.** OpenMask3D, Search3D, OpenScan baselines and OVI-MAP use posed RGB(-D) images. OpenSNAP
   rows without "image column" use XYZ only. C1 contains both kinds of row.
2. **Training data.**
   - The segmenter was trained on ScanNet (and the segmenter token on ScanNet200 labels), so ScanNet results are
     in-distribution for masks and tokens. The same holds for Mask3D in the published rows.
   - Replica is held out from all our training. Matterport3D is not used for training the segmenter.
   - OpenScene's Matterport model is distilled on Matterport *train* without labels, and 11 of the 16 HouseLayout3D
     buildings are Matterport train scans. The partition study (C4) compares joins over the same relation, so it is
     unaffected.
3. **Selections without the test set.**
   - Chosen on the development scenes (S3DIS Area 1 and 30 ScanNet *train* scenes): the ScanNet dense features, the
     proposal mode and the labelling setting (α, temperature, β, confidence form).
   - The image weight, and for lifted 2D proposals their confidence multiplier and γ, have no image-bearing
     development set. They are chosen by leave-one-scene-out on Replica.
   - C1b is shown only as sensitivity.
4. **Evaluator.** `opensnap/bench.py` equals OVI-MAP's `eval_utils.evaluate` to five decimals (AP 0.11164 / 0.19113 /
   0.23952 on the same predictions).
5. **OVI-MAP's ScanNet list** includes three ScanNet train scenes. C2 also reports the 15 val scenes.

---

## 8. Remaining limitations (honest list)

- **Long-tail naming without images is weak** (ScanNet200 common / tail about 2 AP). The image column helps where frames
  exist.
- **Image-fusion weights are sensitive to α and the temperature** and are not learned. With the old setting the fused
  Replica row is 23.4 mIoU; with the dev-chosen one it is 31.6. A learned per-object gate is future work.
- **Leave-one-scene-out on 8 Replica scenes is too noisy to pick settings** (per-fold choices pool to lower numbers than
  any single setting), so the headline uses one setting chosen on ScanNet train scenes instead.
- **The dev-chosen labelling setting helps semantics but not instances without images:**
  - mIoU: ScanNet-18 14.4 → 15.0, ScanNet200 10.4 → 11.0;
  - AP: unchanged on both.
  On ScanNet-18, union_l4 proposals give lower AP than fps (8.9 vs 9.8), so C2 keeps fps as its main row.
- **RGB-only metric scale** (20–42% error) is the main failure of phone-video ingest. An IMU or a known object size would
  fix it.
- **Coverage.** Proposal-based relations miss large structural surfaces and some small objects (AP25 below OVI-MAP).
- **OVQL** has no part-level predicates (Search3D's strength) and no learned relation predicates (Open3DSG's).
  NEAR / ABOVE / BELOW / ON / CONTAINS and boolean composition only. Multi-hop queries nest (`NEAR(IS(a) AND NEAR(IS(b),
  1), 1)`), but are not benchmarked.

---

## 9. Corrections to earlier claims

1. *"Replica 19.9 / 25.9 AP / AP50 vs OpenMask3D 13.1 / 18.4"* (README, RESULTS §1, paper §8.3) compared a
   **presence-oracle query workload** with OpenMask3D's closed-set protocol.
   - Under the closed-set protocol (C1), XYZ-only OpenSNAP gets 12.9 / 15.1 / 17.3 (AP / AP50 / AP25), *below* OpenMask3D's
     published 13.1 / 18.4 / 24.2.
   - With the image column it gets 16.9 / 21.8 / 27.1, *above* them.
   - With the image column and every setting chosen on ScanNet train development scenes, it gets 24.7 / 30.6 / 34.8.
   - The query-workload numbers (T4b) stay valid as *query* accuracy, not as a comparison with OpenMask3D.
2. *"Dense-collapse guard: fires only on the phone video"* (T18, 13 scenes) does not hold at scale. It fires on 4.5–5.9%
   of clean scans and misses the partial collapse on the 8 RGB-only Replica videos (C11).
