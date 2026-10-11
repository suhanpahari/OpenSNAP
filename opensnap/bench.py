"""Standard closed-set benchmark protocols (ScanNet instance AP, per-vertex mIoU/mAcc), array based.

The instance matcher and AP integration follow the official ScanNet benchmark script, as used by OpenMask3D,
Search3D and OVI-MAP: greedy matching per class at IoU thresholds 0.50:0.95 (+0.25), predictions below
`min_region` vertices dropped, unmatched predictions that mostly cover void/unlabelled vertices not counted as FP.
"""
import numpy as np

OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)


def match_scene(gt_ids, preds, valid_ids, min_region=100):
    """gt_ids: (V,) sem_id*1000+k (0 = none). preds: list of (bool mask (V,), label_id, conf).
    Returns {label_id: dict(gt=[...], pred=[...])} for evaluate()."""
    void = ~np.isin(gt_ids // 1000, valid_ids)
    out = {l: dict(gt=[], pred=[]) for l in valid_ids}
    gts = {}
    for gid in np.unique(gt_ids):
        if gid == 0 or gid // 1000 not in out:
            continue
        m = gt_ids == gid
        g = dict(id=int(gid), count=int(m.sum()), matched=[])
        out[gid // 1000]["gt"].append(g)
        gts[int(gid)] = (g, m)
    for pi, (mask, lab, conf) in enumerate(preds):
        if lab not in out:
            continue
        area = int(mask.sum())
        if area < min_region:
            continue
        p = dict(pid=pi, count=area, conf=float(conf), void=int((void & mask).sum()), matched=[])
        hit = np.unique(gt_ids[mask])
        for gid in hit:
            if gid == 0 or gid // 1000 != lab or int(gid) not in gts:
                continue
            g, gm = gts[int(gid)]
            inter = int((gm & mask).sum())
            g["matched"].append(dict(pid=pi, count=area, conf=float(conf), inter=inter))
            p["matched"].append(dict(id=g["id"], count=g["count"], inter=inter))
        out[lab]["pred"].append(p)
    return out


def evaluate(scenes, valid_ids, min_region=100):
    """scenes: list of match_scene outputs. Returns AP array (C, len(OVERLAPS)) with NaN for absent classes."""
    ap = np.full((len(valid_ids), len(OVERLAPS)), np.nan)
    for oi, th in enumerate(OVERLAPS):
        for li, lab in enumerate(valid_ids):
            y_true, y_score, hard_fn = [], [], 0
            has_gt = has_pred = False
            for si, sc in enumerate(scenes):
                visited = set()
                gts = [g for g in sc[lab]["gt"] if g["count"] >= min_region]
                preds = sc[lab]["pred"]
                has_gt |= bool(gts)
                has_pred |= bool(preds)
                cur_true, cur_score, cur_match = [1.0] * len(gts), [-np.inf] * len(gts), [False] * len(gts)
                for gi, g in enumerate(gts):
                    found = False
                    for p in g["matched"]:
                        if p["pid"] in visited:
                            continue
                        iou = p["inter"] / (g["count"] + p["count"] - p["inter"])
                        if iou > th:
                            if cur_match[gi]:
                                lo, hi = min(cur_score[gi], p["conf"]), max(cur_score[gi], p["conf"])
                                cur_score[gi] = hi
                                cur_true.append(0.0); cur_score.append(lo); cur_match.append(True)
                            else:
                                found = True
                                cur_match[gi], cur_score[gi] = True, p["conf"]
                                visited.add(p["pid"])
                    if not found:
                        hard_fn += 1
                cur_true = [t for t, mm in zip(cur_true, cur_match) if mm]
                cur_score = [s for s, mm in zip(cur_score, cur_match) if mm]
                for p in preds:
                    if any(m["inter"] / (m["count"] + p["count"] - m["inter"]) > th for m in p["matched"]):
                        continue
                    ignore = p["void"] + sum(m["inter"] for m in p["matched"] if m["count"] < min_region)
                    if ignore / p["count"] <= th:
                        cur_true.append(0.0); cur_score.append(p["conf"])
                y_true += cur_true
                y_score += cur_score
            if has_gt and has_pred:
                ap[li, oi] = _ap(np.asarray(y_true), np.asarray(y_score), hard_fn)
            elif has_gt:
                ap[li, oi] = 0.0
    return ap


def _ap(y_true, y_score, hard_fn):
    o = np.argsort(y_score)
    ys, yt = y_score[o], y_true[o]
    cs = np.cumsum(yt)
    _, uniq = np.unique(ys, return_index=True)
    n = len(ys)
    n_true = cs[-1] if len(cs) else 0
    prec, rec = np.zeros(len(uniq) + 1), np.zeros(len(uniq) + 1)
    cs = np.append(cs, 0)
    for k, i in enumerate(uniq):
        c = cs[i - 1]
        tp = n_true - c
        fp = n - i - tp
        fn = c + hard_fn
        prec[k] = tp / (tp + fp)
        rec[k] = tp / (tp + fn)
    prec[-1], rec[-1] = 1.0, 0.0
    r = np.append(np.append(rec[0], rec), 0.0)
    return float(np.dot(prec, np.convolve(r, [-0.5, 0, 0.5], "valid")))


def summarize_ap(ap, names, splits=None):
    o50, o25 = np.isclose(OVERLAPS, 0.5), np.isclose(OVERLAPS, 0.25)
    res = dict(AP=float(np.nanmean(ap[:, ~o25])), AP50=float(np.nanmean(ap[:, o50])), AP25=float(np.nanmean(ap[:, o25])))
    per = {n: float(np.nanmean(ap[i, ~o25])) for i, n in enumerate(names) if not np.isnan(ap[i]).all()}
    res["per_class_AP"] = per
    for sp, members in (splits or {}).items():
        idx = [i for i, n in enumerate(names) if n in members]
        if idx:
            res[f"AP_{sp}"] = float(np.nanmean(ap[idx][:, ~o25]))
            res[f"AP50_{sp}"] = float(np.nanmean(ap[idx][:, o50]))
            res[f"AP25_{sp}"] = float(np.nanmean(ap[idx][:, o25]))
    return res


def semantic_scores(conf):
    """OVI-MAP / OpenNeRF convention: classes without GT vertices skipped; IoU and accuracy averaged over the rest."""
    tp = np.diag(conf).astype(np.float64)
    gt = conf.sum(1).astype(np.float64)
    pred = conf.sum(0).astype(np.float64)
    present = gt > 0
    iou = tp[present] / np.maximum(gt + pred - tp, 1)[present]
    acc = tp[present] / gt[present]
    return dict(mIoU=float(iou.mean()), mAcc=float(acc.mean()), fmIoU=float((iou * gt[present]).sum() / gt[present].sum()))
