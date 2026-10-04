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

## 7. Trained decoder (SNAP+)

Backbone frozen; decoder side trained on ScanNet200 (S3DIS and Matterport kept zero-shot).

* **Semantic keys, geometric values (v2).** Per-voxel 3D CLIP features enter only the attention keys (positional
  branch), not the point embeddings that form masks. *v1 added them to the embeddings; this blurred instances of the
  same class and degraded multi-click IoU (diagnosed with oracle selection).*
* **Role-aware mask tokens.** The segmenter's original mask token is the whole-object / multi-click path; `K=3`
  extra tokens are used only for ambiguous single clicks (min-loss), selected by a per-token IoU head.
* **Native text prompts.** A CLIP text embedding is a prompt token; trained on ScanNet200 head+common classes, the 66
  tail classes are held out for zero-shot evaluation.

## 8. Experimental protocol

* **Interactive segmentation** (S3DIS Area 5 full rooms, zero-shot): XYZ-only input; first click at the object
  centre (farthest from the boundary), next clicks at the centre of the missed region; IoU@{1,3,5,10}, NoC@{80,85,90}.
  Our evaluation of the released checkpoint reproduces the published numbers (53.3/69.1/74.0 vs 53.6/71.1/77.6 IoU@1/3/5).
* **Query workload** (S3DIS Area 5): concept queries `IS(c)` and relational `IS(a) AND NEAR(IS(b), 0.5)` with answer
  sets derived from instance annotations; interpolated AP over IoU 0.50:0.95, AP50, AP25; latency; storage.
* **Open-vocabulary naming**: click-to-name accuracy on Matterport3D-160 (seen/unseen classes).
* **Text prompts**: ScanNet200 val head/common/tail mIoU; S3DIS zero-shot.

## 9. Results so far (from `results/`)

| experiment | result |
|---|---|
| S3DIS interactive, released SNAP-C | IoU@1/3/5/10 = 53.3 / 69.1 / 74.0 / 77.4 |
| S3DIS interactive, decoder fine-tuned (control) | 54.1 / 69.2 / 73.4 / 76.8 (4k steps) |
| S3DIS interactive, SNAP+ v1 | 52.6 / 63.7 / 66.4 / 69.2 (regression, see Sec. 7) |
| S3DIS interactive, SNAP+ v2 | **TBD** |
| S3DIS queries `IS(c)` AP/AP50/AP25 | segmenter token only 9.8/16.7/25.1 → OpenSNAP 11.6/17.8/26.1 → + cracking 11.6/18.2/26.5 |
| S3DIS queries `IS∧NEAR` AP | 12.2 → 14.5 |
| Relation size vs raw XYZ | 2.1 MB vs 13.9 MB per room; 8 ms/query; 11 s ingest |
| Proposal recall@0.5 (S3DIS, 6 rooms) | FPS-400 64.9%, FPS-800 69.2%, coarse-to-fine 68.5% |
| Matterport-160 unseen-class naming (536 clicks) | segmenter token 8.0% → OpenSNAP 15.5% (mAcc 1.9 → 8.9) |

Earlier ScanNet/Matterport semantic-segmentation numbers in the README were produced before the XYZ-only input fix and
must be re-run before reporting.

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
