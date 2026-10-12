"""Score a test dump once with the dev-chosen labelling setting (results/compare/tuned.json), with head/common/tail."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensnap import bench  # noqa: E402
import eval_standard as E  # noqa: E402
import tune_dev as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="scannet200")
    ap.add_argument("--dump", default="../downloads/runs/dump_scannet200")
    ap.add_argument("--recs", default="results/compare/scannet200_ul4_recs.pt")
    ap.add_argument("--inst", default="198")
    ap.add_argument("--tuned", default="results/compare/tuned.json")
    ap.add_argument("--out", default="results/compare/scannet200_tuned.json")
    a = ap.parse_args()
    labels, ids, _, inst_sets = E.benchmark(a.bench)
    valid = inst_sets[a.inst]
    name_of = dict(zip(ids, labels))
    sp = json.load(open(os.path.join(E.VOCAB, "scannet200_splits.json"))) if a.bench.startswith("scannet") else None
    tuned = json.load(open(a.tuned))
    scenes, D, G, M = T.load_set(a.dump, a.recs, valid)
    res = {}
    for tag in ("default", "chosen"):
        g = tuple(tuned[tag])
        outs = [T.outputs(D[s], g) for s in scenes]
        sc = [bench.match_scene(G[s], [(mk, l, c) for mk, (l, c) in zip(M[s], o[1])], valid) for s, o in zip(scenes, outs)]
        apr = bench.summarize_ap(bench.evaluate(sc, valid), [name_of[i] for i in valid], sp)
        apr.pop("per_class_AP")
        res[tag] = dict(setting=list(g), **T.sem_metrics(sum(o[0] for o in outs)), **apr)
        print(tag, g, {k: round(100 * v, 1) for k, v in res[tag].items() if k != "setting"}, flush=True)
    json.dump(dict(n_scenes=len(scenes), **res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
