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
