"""Regenerate results/TABLES.md from the JSON result files (every number traceable to a file)."""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
out = []


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def table(title, src, header, rows, note=""):
    out.append(f"### {title}\n\n_Source: {src}_\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    out.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    if note:
        out.append(f"\n{note}")
    out.append("")


def pending(title, what):
    out.append(f"### {title}\n\n**Pending:** {what}\n")


def semseg():
    names = {"openscene": "OpenScene 3D (point-wise)", "snap_token": "Segmenter text token", "snap_token_relay": "+ vocabulary relay",
             "mask_pool": "Mask-anchored pooling", "mask_pool_dense": "+ dense blend", "opensnap_norelay": "OpenSNAP w/o relay",
             "opensnap": "**OpenSNAP (full)**"}
    sc, mp = load("scannet_val_xyz.json"), load("matterport160_test_xyz.json")
    if sc and mp:
        rows = [[n, pct(sc["results"][k]["mIoU"]), pct(sc["results"][k]["mAcc"]), pct(mp["results"][k]["mIoU"]),
                 pct(mp["results"][k]["mIoU_seen"]), pct(mp["results"][k]["mIoU_unseen"])] for k, n in names.items()]
        table("T1/T2 · Open-vocabulary 3D semantic segmentation and component ablation",
              "scannet_val_xyz.json (312 scenes), matterport160_test_xyz.json (406 regions)",
              ["method", "ScanNet mIoU", "ScanNet mAcc", "MP-160 mIoU", "MP-160 seen", "MP-160 unseen"], rows,
              "Unseen = the 63 Matterport-160 classes absent from the segmenter's training vocabulary. XYZ-only segmenter input.")
    os_ = load("scannet_val_xyz_openseg.json")
    if os_ and sc:
        table("T15 · Dense-feature source (ScanNet val mIoU)", "scannet_val_xyz.json vs scannet_val_xyz_openseg.json",
              ["method", "LSeg (ScanNet-trained)", "OpenSeg (Matterport-trained)"],
              [[n, pct(sc["results"][k]["mIoU"]), pct(os_["results"][k]["mIoU"])] for k, n in names.items() if k in os_["results"]])
    else:
        pending("T15 · Dense-feature source", "scannet_val_xyz_openseg.json")

    nu, nuo = load("nuscenes_val_xyz.json"), load("nuscenes_official_val.json")
    if nu or nuo:
        cell = lambda d, k: [pct(d["results"][k]["mIoU"]), pct(d["results"][k]["mAcc"])] if d and k in d["results"] else ["–", "–"]
        rows = [[n] + cell(nuo, k) + cell(nu, k) for k, n in names.items()]
        rows.append(["OpenScene 3D distill (published)", "36.7", "–", "", ""])
        rows.append(["OpenScene 2D-3D ensemble (published)", "42.1", "–", "", ""])
        table("T2b · Open-vocabulary LiDAR semantic segmentation, nuScenes val (16 classes)", "nuscenes_official_val.json, nuscenes_val_xyz.json",
              ["method", "mIoU (official, 6019 keyframes)", "mAcc", "mIoU (OpenScene-processed, 1003)", "mAcc"], rows,
              "Segmenter in Outdoor mode (XYZ-only, 5 cm grid); dense branch = OpenScene nuScenes OpenSeg 3D model. Official = single "
              "LiDAR keyframes with lidarseg labels (SNAP's 16-class map); OpenScene-processed = every 6th sweep of OpenScene's release. "
              "Published OpenScene numbers use the authors' script.")

    sk = load("semantickitti_val.json")
    if sk:
        rows = [[n, pct(sk["results"][k]["mIoU"]), pct(sk["results"][k]["mAcc"])] for k, n in names.items() if k in sk["results"]]
        table(f"T2c · Open-vocabulary LiDAR semantic segmentation, SemanticKITTI seq 08 (19 classes, {sk['num_scenes']} scans)",
              "semantickitti_val.json", ["method", "mIoU", "mAcc"], rows,
              "Dense branch = OpenScene's nuScenes model applied cross-dataset (OpenScene has no KITTI model), so the dense "
              "features are zero-shot here; the segmenter saw SemanticKITTI train. The weak cross-dataset dense branch (terrain 0.0, "
              "parking 0.1 IoU) costs mIoU on stuff classes; OpenSNAP gains on objects (other vehicle 2.9 -> 11.9) and in mAcc.")


def queries():
    eng = {"snap_native": "Segmenter token only", "opensnap": "**OpenSNAP**", "opensnap_refine": "OpenSNAP + cracking"}
    s3, sn, s3a = load("s3dis_queries.json"), load("scannet_queries.json"), load("s3dis_queries_allclasses.json")
    if s3 and sn:
        rows = []
        for k, n in eng.items():
            a, b = s3["results"], sn["results"]
            rows.append([n, pct(a[f"{k}/IS"]["AP"]), pct(a[f"{k}/IS"]["AP50"]), pct(a[f"{k}/NEAR"]["AP"]),
                         pct(b[f"{k}/IS"]["AP"]), pct(b[f"{k}/IS"]["AP_head"]), pct(b[f"{k}/IS"]["AP_common"]),
                         pct(b[f"{k}/IS"]["AP_tail"]), pct(b[f"{k}/NEAR"]["AP"]), f"{b[f'{k}/IS']['ms']:.1f}"])
        table("T3 · OVQL query workloads (instance AP, IoU 0.5:0.95)", "s3dis_queries.json (68 rooms, object classes), scannet_queries.json (156 val scenes)",
              ["engine", "S3DIS IS AP", "S3DIS IS AP50", "S3DIS NEAR AP", "SN200 IS AP", "head", "common", "tail", "SN200 NEAR AP", "ms/query"], rows,
              "The segmenter's text head was trained on all 200 ScanNet200 classes (in-domain, incl. tail).")
    c2f = load("s3dis_queries_allclasses_c2f.json")
    if s3a:
        rows = [[n, pct(s3a["results"][f"{k}/IS"]["AP"]), pct(s3a["results"][f"{k}/IS"]["AP50"]), pct(s3a["results"][f"{k}/IS"]["AP25"])]
                + ([pct(c2f["results"][f"{k}/IS"]["AP"]), pct(c2f["results"][f"{k}/IS"]["AP50"]), pct(c2f["results"][f"{k}/IS"]["AP25"])] if c2f else ["–"] * 3)
                for k, n in eng.items()]
        rows.append(["SNAP-C@auto (published, Table 4)", "16.1", "23.8", "30.9", "", "", ""])
        table("T4 · Open-vocabulary instance segmentation on S3DIS Area 5 (all 13 classes)",
              "s3dis_queries_allclasses.json (one-shot proposals), s3dis_queries_allclasses_c2f.json (coarse-to-fine)",
              ["engine", "AP (FPS)", "AP50 (FPS)", "AP25 (FPS)", "AP (c2f)", "AP50 (c2f)", "AP25 (c2f)"], rows,
              "Published numbers use the authors' own evaluation; ours use one protocol for all engines.")

    rf, rc = load("replica_queries.json"), load("replica_queries_c2f.json")
    if rf:
        names = {**eng, "opensnap_refine_v4": "OpenSNAP + cracking (v4 refiner)"}
        cell = lambda d, k: [pct(d["results"][f"{k}/IS"][m]) for m in ("AP", "AP50", "AP25")] if d and f"{k}/IS" in d["results"] else ["–"] * 3
        rows = [[n] + cell(rf, k) + cell(rc, k) for k, n in names.items()]
        rows += [["SNAP-C@auto (published, Table 4)", "10.1", "11.7", "13.8", "", "", ""],
                 ["OpenMask3D (published)", "13.1", "18.4", "24.2", "", "", ""]]
        table("T4b · Open-vocabulary instance segmentation on Replica (8 scenes)", "replica_queries.json, replica_queries_c2f.json",
              ["engine", "AP (FPS)", "AP50 (FPS)", "AP25 (FPS)", "AP (c2f)", "AP50 (c2f)", "AP25 (c2f)"], rows,
              "Instances transferred from Replica habitat meshes (exact alignment); all object classes except structure/undefined. "
              "Published numbers use the OpenMask3D 51-class protocol.")

    nq = load("nuscenes_queries.json")
    if nq:
        r = nq["results"]
        rows = [[n, pct(r[f"{k}/IS"]["AP"]), pct(r[f"{k}/IS"]["AP50"]), pct(r[f"{k}/IS"]["AP25"]), pct(r[f"{k}/NEAR"]["AP"]),
                 f"{r[f'{k}/IS']['ms']:.1f}"] for k, n in eng.items() if f"{k}/IS" in r]
        table("T4c · Open-vocabulary instance segmentation on nuScenes LiDAR (10 thing classes)", "nuscenes_queries.json",
              ["engine", "AP", "AP50", "AP25", "NEAR AP", "ms/query"], rows,
              "Instances = lidarseg points inside annotated 3D boxes of the same category (how nuScenes-panoptic is built); "
              "every 4th val keyframe; objects with >= 20 voxels.")

    kq = load("semantickitti_queries.json")
    if kq:
        r = kq["results"]
        rows = [[n, pct(r[f"{k}/IS"]["AP"]), pct(r[f"{k}/IS"]["AP50"]), pct(r[f"{k}/IS"]["AP25"]), pct(r[f"{k}/NEAR"]["AP"]),
                 f"{r[f'{k}/IS']['ms']:.1f}"] for k, n in eng.items() if f"{k}/IS" in r]
        table("T4d · Open-vocabulary instance segmentation on SemanticKITTI seq 08 (8 thing classes)", "semantickitti_queries.json",
              ["engine", "AP", "AP50", "AP25", "NEAR AP", "ms/query"], rows,
              "Panoptic instance labels; every 4th scan; objects with >= 20 voxels.")


def refiner():
    q, sy = load("s3dis_queries_v4.json"), load("systems_s3dis_v4.json")
    if q:
        names = {"opensnap": "OpenSNAP (no cracking)", "opensnap_refine": "cracking, click-oracle refiner",
                 "opensnap_refine_v4": "**cracking, v4 text-decoder refiner (dedup 0.5)**", "opensnap_refine_v4_nodedup": "cracking, v4 refiner, no dedup"}
        r = q["results"]
        rows = [[n, pct(r[f"{k}/IS"]["AP"]), pct(r[f"{k}/IS"]["AP50"]), pct(r[f"{k}/NEAR"]["AP"]), f"{r[f'{k}/IS'].get('refined', 0):.2f}",
                 f"{r[f'{k}/IS']['ms']:.1f}"] for k, n in names.items() if f"{k}/IS" in r]
        table("T12b · Query-time refiner: click oracle vs trained text decoder (S3DIS Area 5 queries)", "s3dis_queries_v4.json",
              ["engine", "IS AP", "IS AP50", "NEAR AP", "refined/room", "IS ms"], rows)
    if sy:
        s = sy["summary"]
        rows = [[m, f"{s[m]['mean_ms']:.1f}", f"{s[m]['second_pass_ms']:.1f}", pct(s[m]["ap50"]), pct(s[m]["synonym_ap50"])]
                for m in ("no_refine", "refine_discard", "cracking")]
        table("T12c · Cracking workload with the v4 text-decoder refiner (S3DIS, 34 rooms)", "systems_s3dis_v4.json",
              ["mode", "mean ms", "2nd-pass ms", "AP50", "synonym AP50"], rows)


def interactive():
    def get(name, sub="results"):
        d = load(name)
        return d[sub] if d else None
    full = get("s3dis_interactive_snap_zeroshot_xyz.json")
    rows = []
    if full:
        rows.append(["SNAP-C (released, our protocol)", "full", pct(full["IoU@1"]), pct(full["IoU@3"]), pct(full["IoU@5"]), pct(full["IoU@10"]), f"{full['NoC@90']:.2f}"])
    rows.append(["SNAP-C (published)", "full", "53.6", "71.1", "77.6", "83.2", "–"])
    for name, label in [("aux/s3dis_inter_snapft_4k.json", "decoder fine-tune on ScanNet (v1 control, 4k)"),
                        ("aux/s3dis_inter_plus_4k.json", "SNAP+ v1 (sem in features, min-loss tokens, 4k)")]:
        d = get(name)
        if d:
            rows.append([label, "full", pct(d["IoU@1"]), pct(d["IoU@3"]), pct(d["IoU@5"]), pct(d["IoU@10"]), f"{d['NoC@90']:.2f}"])
    for name, label in [("aux/diag_snap.json", "SNAP-C"), ("aux/curve_v2_control_4000.json", "v2 control (4k)"),
                        ("aux/curve_v2_plus_4000.json", "SNAP+ v2 (sem keys, role tokens, 4k)"),
                        ("aux/curve_v3_plus_4000_fixed.json", "SNAP+ v3 (frozen segmenter, 4k)")]:
        d = get(name)
        if d:
            rows.append([label, "subset", pct(d["IoU@1"]), pct(d["IoU@3"]), pct(d["IoU@5"]), pct(d["IoU@10"]), f"{d['NoC@90']:.2f}"])
    table("T5 · Interactive segmentation, S3DIS Area 5 (zero-shot)", "s3dis_interactive_*.json, aux/*",
          ["model", "rooms", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
          "Subset = every 4th room (17 rooms). XYZ-only input; first click at object centre, then centre of the missed region.")
    c = get("s3dis_interactive_crops.json")
    if c:
        rows = [["SNAP-C (released, our run)", pct(c["IoU@1"]), pct(c["IoU@3"]), pct(c["IoU@5"]), f"{c['NoC@90']:.2f}", c["objects"]],
                ["SNAP-C (published)", "56.6", "73.8", "80.9", "–", "–"], ["AGILE3D (published)", "58.7", "77.4", "83.6", "–", "–"],
                ["Point-SAM (published)", "45.9", "77.6", "84.6", "–", "–"]]
        table("T5c · Interactive segmentation on S3DIS Area 5 object crops (SNAP Table 2 protocol)", "s3dis_interactive_crops.json",
              ["model", "IoU@1", "IoU@3", "IoU@5", "NoC@90", "objects"], rows,
              "±1.5 m crop around each object. OpenSNAP's click path is SNAP-exact, so its row equals the SNAP-C run.")
    n = get("nuscenes_interactive.json")
    if n:
        table("T5d · Interactive segmentation on nuScenes LiDAR val (Outdoor)", "nuscenes_interactive.json",
              ["model", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@80", "NoC@90", "objects"],
              [["SNAP-C (released, our run) = OpenSNAP click path", pct(n["IoU@1"]), pct(n["IoU@3"]), pct(n["IoU@5"]), pct(n["IoU@10"]),
                f"{n['NoC@80']:.2f}", f"{n['NoC@90']:.2f}", n["objects"]]],
              "Every 10th val keyframe; things = box-derived instances (>= 10 points), stuff = one segment per class (as in nuScenes-panoptic).")
    k = get("semantickitti_interactive.json")
    if k:
        table("T5e · Interactive segmentation on SemanticKITTI seq 08 (Outdoor)", "semantickitti_interactive.json",
              ["model", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@80", "NoC@90", "objects"],
              [["SNAP-C (released, our run) = OpenSNAP click path", pct(k["IoU@1"]), pct(k["IoU@3"]), pct(k["IoU@5"]), pct(k["IoU@10"]),
                f"{k['NoC@80']:.2f}", f"{k['NoC@90']:.2f}", k["objects"]]],
              "Every 10th scan; things = panoptic instances (>= 10 points), stuff = one segment per class.")
    sets = [("scannet_dev", "ScanNet val (dev subset)", None), ("s3dis_crops", "S3DIS crops", "s3dis_interactive_crops.json"), ("s3dis_full", "S3DIS rooms", "s3dis_interactive_snap_zeroshot_xyz.json"),
            ("replica", "Replica", None), ("nuscenes", "nuScenes", "nuscenes_interactive.json"),
            ("kitti", "SemanticKITTI", "semantickitti_interactive.json")]
    methods = [("snap", "SNAP, corrective protocol (negatives ignored)"), ("subtract", "+ negative-mask subtraction"),
               ("ttt", "**OpenSNAP: test-time token tuning**"), ("v5", "OpenSNAP: corrective decoder (ScanNet-trained)"), ("v5m", "OpenSNAP: corrective decoder (multi-domain)"),
               ("v5ttt", "OpenSNAP: ScanNet decoder + token tuning"), ("v5mttt", "OpenSNAP: multi-domain decoder + token tuning")]
    rows = []
    for key, name, pos_file in sets:
        if pos_file and get(pos_file):
            r = get(pos_file)
            rows.append([name, "SNAP, positive clicks only", pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"])
        for m, label in methods:
            r = get(f"aux/pn_{key}_{m}.json") or (get(f"aux/s3dis_crops_pn_{m}.json") if key == "s3dis_crops" and m in ("snap", "subtract") else None)
            if r:
                rows.append([name, label, pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"])
        if key == "s3dis_crops":
            rows += [[name, "AGILE3D (published)", "58.7", "77.4", "83.6", "–", "–"], [name, "Point-SAM (published)", "45.9", "77.6", "84.6", "–", "–"]]
    if rows:
        table("T5f · Corrective clicking with positive and negative clicks", "aux/pn_*.json",
              ["dataset", "method", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Corrective protocol: every click goes to the centre of the larger error region (positive on misses, negative on "
              "false positives). SNAP cannot use negative clicks. Token tuning: 30 Adam steps on the mask token over SNAP's frozen "
              "voxel embeddings, hyper-parameters chosen on ScanNet val only (aux/sweep_ttt_scannet_*.json).")
    row = lambda r: [pct(r.get(k)) for k in ("IoU@1", "IoU@3", "IoU@5", "IoU@10")] + ([f"{r['NoC@90']:.2f}"] if "IoU@10" in r else ["–"])
    pairs = [("S3DIS rooms", "s3dis_interactive_snap_zeroshot_xyz.json", "aux/pn_posonly_s3dis_full_v6.json", "held out"),
             ("S3DIS crops", "s3dis_interactive_crops.json", "aux/pn_posonly_s3dis_crops_v6.json", "held out"),
             ("nuScenes", "nuscenes_interactive.json", "aux/pn_posonly_nuscenes_v6.json", "train split seen"),
             ("SemanticKITTI", "semantickitti_interactive.json", "aux/pn_posonly_kitti_v6.json", "train split seen")]
    rows = []
    for name, base, ad, note in pairs:
        if get(base) and get(ad):
            rows += [[name, note, "SNAP"] + row(get(base)), [name, note, "**+ backbone LoRA adapters**"] + row(get(ad))]
    if rows:
        table("T5i · Backbone adaptation (LoRA in every PTv3 attention layer), positive clicks only", "aux/pn_posonly_*_v6.json",
              ["dataset", "adapter training data", "model", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Adapters (rank 16, zero-init, 6k steps) trained jointly with the corrective decoder on ScanNet + nuScenes-train + "
              "SemanticKITTI-train with feature distillation to the frozen backbone; decoder for positive clicks is SNAP's own.")
    rows = []
    for name, key in [("S3DIS crops", "s3dis_crops"), ("S3DIS rooms", "s3dis_full"), ("Replica", "replica"), ("nuScenes", "nuscenes"),
                      ("SemanticKITTI", "kitti")]:
        for m, label in [("snap", "SNAP (negatives ignored)"), ("v6snap", "SNAP decoder + adapters"), ("ttt", "token tuning"),
                         ("v5mttt", "multi-domain decoder + token tuning"), ("v6ttt", "adapters + decoder + token tuning")]:
            r = get(f"aux/pn_{key}_{m}.json") or (get(f"aux/s3dis_crops_pn_{m}.json") if key == "s3dis_crops" and m == "snap" else None)
            if r:
                rows.append([name, label] + row(r))
    if rows:
        table("T5j · Corrective clicking with the adapted backbone", "aux/pn_*_v6*.json",
              ["dataset", "method", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Adapters help where their training domains are (nuScenes first click +3.6) and are neutral-to-slightly-negative on "
              "held-out indoor data; token tuning alone remains best on indoor rooms.")
    rows = []
    for name, a, b in [("ScanNet val dev (selection set)", "aux/sweep_ttt_scannet_s30_r10_rad0.json", "aux/sweep_tta_scannet_k8_r0.1.json"),
                       ("S3DIS crops (held out)", "aux/pn_s3dis_crops_ttt.json", "aux/pn_s3dis_crops_tttj.json"),
                       ("nuScenes (held out)", "aux/pn_nuscenes_ttt.json", "aux/pn_nuscenes_tttj.json")]:
        if get(a) and get(b):
            ra = dict(get(a)) if "dev" not in name else {k: get(a)[k] for k in ("IoU@1", "IoU@3")}
            rows += [[name, "token tuning"] + row(ra), [name, "**+ click-jitter averaging (8 × 10 cm)**"] + row(get(b))]
    for r in ("0.05", "0.2"):
        x = get(f"aux/sweep_tta_scannet_k8_r{r}.json")
        if x:
            rows.append(["ScanNet val dev (selection set)", f"jitter radius {r} m"] + row(x))
    if rows:
        table("T5k · Click-jitter test-time augmentation for the first click", "aux/sweep_tta_*.json, aux/pn_*_tttj.json",
              ["dataset", "method", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], rows,
              "Dev sweep ran 3 clicks only. Radius chosen on ScanNet; helps indoors (+1.1 / +1.5 IoU@1), no effect on sparse outdoor LiDAR.")
    dc = load("aux/dense_collapse.json")
    if dc:
        bench = [r for r in dc["rows"] if r["source"] == "benchmark scan"]
        video = [r for r in dc["rows"] if r["source"] != "benchmark scan"]
        rows = [[f"{len(bench)} benchmark scans (ScanNet, S3DIS, Replica, Matterport)", f"{min(r['top_object_label_share'] for r in bench):.2f}–{max(r['top_object_label_share'] for r in bench):.2f}", "no"]]
        rows += [[f"{r['scene']} ({r['source']})", f"{r['top_object_label_share']:.2f} ('{r['label']}')", "**yes**"] for r in video]
        table("T18 · Dense-feature collapse guard (out-of-distribution scenes)", "aux/dense_collapse.json",
              ["scene", "top non-structural label share", f"guard fires (> {dc['threshold']})"], rows,
              "When it fires, OpenSNAP drops the dense 3D CLIP features and uses the segmenter's own tokens for relevance and naming.")
    lat = load("aux/click_latency.json")
    if lat:
        names = {"snap": "SNAP (negatives ignored)", "ttt": "token tuning (30 steps)", "v5": "corrective decoder"}
        table(f"T5h · Per-click latency with negative clicks (S3DIS rooms, {lat['mean_voxels'] / 1e3:.0f}k voxels)", "aux/click_latency.json",
              ["method", "median ms", "p90 ms", "clicks"], [[n, f"{lat[k]['median_ms']:.0f}", f"{lat[k]['p90_ms']:.0f}", lat[k]["n"]] for k, n in names.items()],
              "Backbone encoding is done once per scene and excluded; shared MIG 3g.40gb slice.")
    sw = []
    for st, rg, rad in [(30, 0.3, 0), (30, 1, 0), (30, 3, 0), (30, 10, 0), (30, 30, 0), (100, 1, 0), (30, 1, 0.05), (100, 0.3, 0.05)]:
        r = get(f"aux/sweep_ttt_scannet_s{st}_r{rg}_rad{rad}.json")
        if r:
            sw.append([st, rg, rad, pct(r["IoU@1"]), pct(r["IoU@3"]), pct(r["IoU@5"]), pct(r["IoU@10"]), f"{r['NoC@90']:.2f}"])
    if sw:
        table("T5g · Token-tuning hyper-parameters (ScanNet val crops, every 8th scene; selection set)", "aux/sweep_ttt_scannet_*.json",
              ["steps", "reg", "label radius (m)", "IoU@1", "IoU@3", "IoU@5", "IoU@10", "NoC@90"], sw,
              "Selected: 30 steps, reg 10, click-only labels; used unchanged on every other dataset.")
    diag = [("aux/diag_plus_iou.json", "IoU-head selection"), ("aux/diag_plus_oracle.json", "oracle best-of-K"),
            ("aux/diag_plus_first.json", "always token 0"), ("aux/diag_plus_nosem.json", "semantic features removed")]
    rows = [[l, pct(get(n)["IoU@1"]), pct(get(n)["IoU@3"]), pct(get(n)["IoU@5"]), pct(get(n)["IoU@10"])] for n, l in diag if get(n)]
    if rows:
        table("T5b · Diagnosis of SNAP+ v1 regression (S3DIS subset, 4k)", "aux/diag_*.json", ["selection", "IoU@1", "IoU@3", "IoU@5", "IoU@10"], rows)


def naming():
    order = [("snap", "Segmenter text head"), ("opensnap_3d", "OpenSNAP-3D"), ("vlm", "VLM free naming"), ("rav", "Render-Ask-Verify"),
             ("vlm_choose", "VLM choose (alone)"), ("choose", "**3D-propose / VLM-choose**"),
             ("openscene_point", "closed: dense feature at click"), ("opensnap_closed", "closed: OpenSNAP")]
    u, a3, a7 = load("naming_mp160_unseen_xyz.json"), load("naming_mp160_all_xyz_3b.json"), load("naming_mp160_all_xyz_7b.json")
    if u and a3 and a7:
        rows = [[n, pct(u["results"][k]["acc"]), pct(u["results"][k]["macc"]), pct(a3["results"][k]["acc"]), pct(a3["results"][k]["macc"]),
                 pct(a7["results"][k]["acc"]), pct(a7["results"][k]["macc"])] for k, n in order]
        table("T6/T7 · Click-to-name on Matterport-160 (open vocabulary unless marked closed)",
              f"naming_mp160_unseen_xyz.json ({u['clicks']} unseen-class clicks), naming_mp160_all_xyz_3b/7b.json ({a3['clicks']} clicks)",
              ["method", "unseen acc", "unseen mAcc", "all acc (3B)", "all mAcc (3B)", "all acc (7B)", "all mAcc (7B)"], rows,
              "Free-form names mapped to labels by CLIP ViT-L/14 text similarity. Run-to-run noise ≈ ±3 points on 212 clicks.")


def text_prompts():
    rows = []
    for name, label in [("aux/text_scannet_oracleclicks.json", "training-free: language as click oracle"),
                        ("aux/text_scannet_v3_4k_tok0.json", "trained v3 (frozen decoder), 4k"),
                        ("aux/text_v4_4000.json", "trained v4 (own text decoder), 4k"), ("aux/text_v4_8000.json", "trained v4, 8k"),
                        ("aux/text_v4_12000.json", "trained v4, 12k"), ("aux/text_v4_16000.json", "trained v4, 16k")]:
        d = load(name)
        if d:
            r = d["results"]
            rows.append([label, pct(r["mIoU"]), pct(r["mIoU_head"]), pct(r["mIoU_common"]), pct(r["mIoU_tail"])])
    table("T8 · Text-prompted segmentation, ScanNet200 val ('segment the <class>')", "aux/text_*.json (104 scenes)",
          ["method", "mIoU", "head", "common", "tail (unseen as text)"], rows, "Tail classes are held out of text training.")


def text_s3dis():
    a, b = load("text_s3dis_oracle.json"), load("text_s3dis_v4.json")
    if a and b:
        cls = sorted(b["per_class"])
        rows = [[n, pct(d["results"]["mIoU"])] + [pct(d["per_class"].get(c)) for c in cls]
                for n, d in (("training-free: language as click oracle", a), ("**trained v4 text decoder (16k)**", b))]
        table("T8b · Text-prompted segmentation on S3DIS Area 5 (zero-shot dataset, per class IoU)",
              "text_s3dis_oracle.json, text_s3dis_v4.json", ["method", "mIoU"] + cls, rows)


def single_click():
    d = load("aux/single_click_scannet_4k.json")
    if d:
        r = d["results"]
        rows = [["segmenter's own mask", pct(r["snap_token"])], ["oracle best of candidates", pct(r["oracle"])]] + \
               [[f"IoU-head selection, margin {k}", f"{pct(v[0])} (extra used {100 * v[1]:.0f}%)"] for k, v in r["rules"].items()]
        table("T9 · Single-click candidate selection (ScanNet val)", f"aux/single_click_scannet_4k.json ({r['objects']} objects)", ["rule", "mean IoU"], rows)


def proposals():
    d = load("proposals_s3dis_area5.json") or load("aux/proposals_s3dis_6rooms.json")
    if d:
        rows = [[k, pct(v["r50"]), pct(v["r25"]), f"{v['n']:.0f}", f"{v['s']:.1f}"] for k, v in d.items()]
        table("T10 · Proposal recall vs prompting scheme (S3DIS Area 5)",
              "proposals_s3dis_area5.json" if load("proposals_s3dis_area5.json") else "aux/proposals_s3dis_6rooms.json (pilot, 6 rooms)",
              ["scheme", "recall@0.5", "recall@0.25", "proposals/room", "s/room"], rows)


def systems():
    d = load("systems_s3dis.json")
    if d:
        s = d["summary"]
        table("T11 · Systems: ingest, latency, storage (S3DIS Area 5, 34 rooms)", "systems_s3dis.json",
              ["metric", "value"],
              [["encode per room (s)", f"{s['encode_s']:.2f}"], ["materialise per room (s)", f"{s['materialize_s']:.2f}"],
               ["query on stored relation (ms)", f"{s['query_cached_ms']:.1f}"], ["re-process scan per query (ms)", f"{s['query_reprocess_ms']:.0f}"],
               ["speed-up", f"{s['query_reprocess_ms'] / s['query_cached_ms']:.0f}×"], ["relation (MB)", f"{s['relation_MB']:.2f}"],
               ["raw XYZ (MB)", f"{s['raw_MB']:.2f}"], ["compression", f"{s['raw_MB'] / s['relation_MB']:.1f}×"]])
        rows = [[m, f"{s[m]['mean_ms']:.1f}", f"{s[m]['second_pass_ms']:.1f}", pct(s[m]["ap50"]), pct(s[m]["synonym_ap50"])]
                for m in ("no_refine", "refine_discard", "cracking")]
        table("T12/T14 · Semantic cracking over a query workload (class names, synonyms, absent terms; two passes)", "systems_s3dis.json",
              ["mode", "mean ms", "2nd-pass ms", "AP50", "synonym AP50"], rows)


def sweeps():
    rows = []
    for name in ("aux/q_scannet_train_sweep.json", "aux/q_scannet_train_sweep_dedup.json"):
        d = load(name)
        if not d:
            continue
        for k, v in d["results"].items():
            if k.endswith("/IS") and k.split("/")[0] not in [r[0] for r in rows]:
                v2 = d["results"].get(k.replace("/IS", "/NEAR"), {})
                rows.append([k.split("/")[0], pct(v["AP"]), pct(v["AP50"]), f"{v.get('refined', 0):.2f}", pct(v2.get("AP"))])
    if rows:
        rows.sort(key=lambda r: -float(r[1]))
        table("T13 · Cracking calibration ablation (ScanNet train, 61 scenes)", "aux/q_scannet_train_sweep*.json",
              ["setting", "IS AP", "IS AP50", "refined/scene", "NEAR AP"], rows,
              "g = confidence weight γ, t = trigger θ, dedup = IoU above which a refined mask is merged into an existing object.")


def building():
    d, lat = load("building_s3dis_area5.json"), load("building_latency.json")
    if d:
        s = d["summary"]
        rows = [["rooms / objects", f"{s['rooms']} / {s['objects']}"], ["total ingest (s)", f"{s['ingest_s_total']:.0f}"],
                ["raw XYZ (MB)", f"{s['raw_xyz_MB']:.0f}"], ["catalog on disk (MB)", f"{s['catalog_MB']:.0f}"],
                ["catalog load (s)", f"{s['catalog_load_s']:.2f}"], ["building IS AP / AP50", f"{pct(s['IS']['AP'])} / {pct(s['IS']['AP50'])}"],
                ["building NEAR AP / AP50", f"{pct(s['NEAR']['AP'])} / {pct(s['NEAR']['AP50'])}"]]
        if lat:
            rows += [[f"latency {k} (ms)", f"{v:.2f}"] for k, v in lat["latency_ms"].items()]
            rows.append(["partitioned join == per-room", lat.get("partition_check", "")])
        table("T16 · Building-level catalog and queries (S3DIS Area 5 as one database)", "building_s3dis_area5.json, building_latency.json", ["metric", "value"], rows)


if __name__ == "__main__":
    out.append("# OpenSNAP result tables\n\nGenerated by `scripts/make_tables.py` from the JSON files in `results/`. Do not edit by hand.\n")
    for f in (semseg, queries, refiner, interactive, naming, text_prompts, text_s3dis, single_click, proposals, systems, sweeps, building):
        f()
    open(os.path.join(R, "TABLES.md"), "w").write("\n".join(out))
    print("\n".join(out))
