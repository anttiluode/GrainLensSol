"""Summarize all registered cells; no best-dose selection."""
import argparse
import json
from collections import defaultdict
from statistics import mean

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report")
    args=p.parse_args()
    report=json.load(open(args.report,encoding="utf-8"))
    rows=report["rows"]
    print("Complete:",report["complete"],"rows:",len(rows))
    groups=defaultdict(list)
    paired=defaultdict(dict)
    for r in rows:
        key=(r["seed"],r["layer"],r["arm"],r["bits"])
        groups[key+(r["grain"],)].append(r)
        paired[key+(r["item"],r["template"])][r["grain"]]=r
    print("seed layer arm bits grain N clean_acc coarse_acc noise_acc clean_margin loss excess_L2matched excess_span half_restore_adv dose noise_dose KL")
    for k,rs in sorted(groups.items()):
        c=lambda r:r["results"]["coarse"]
        n=lambda r:r["results"]["noise"]
        h=lambda r:r["results"]["half_true"]
        rr=lambda r:r["results"]["half_random"]
        span=[r["results"]["noise_in_span"]["margin"]-c(r)["margin"] for r in rs if "noise_in_span" in r["results"]]
        vals=[mean(r["clean"]["choice_correct"] for r in rs),
              mean(c(r)["choice_correct"] for r in rs),mean(n(r)["choice_correct"] for r in rs),
              mean(r["clean"]["margin"] for r in rs),
              mean(r["clean"]["margin"]-c(r)["margin"] for r in rs),
              mean(n(r)["margin"]-c(r)["margin"] for r in rs),
              mean(span) if span else float("nan"),
              mean(h(r)["margin"]-rr(r)["margin"] for r in rs),
              mean(c(r)["relative_rms"] for r in rs),mean(n(r)["relative_rms"] for r in rs),
              mean(c(r)["kl"] for r in rs)]
        print(*k,len(rs),*(f"{v:.4f}" for v in vals))
    print("\nFine-minus-broad excess margin loss, averaged per item before averaging items")
    interactions=defaultdict(lambda:defaultdict(list))
    for k,g in paired.items():
        if 0 not in g or 2 not in g:
            continue
        def excess(r):
            return r["results"]["noise"]["margin"]-r["results"]["coarse"]["margin"]
        interactions[k[:4]][k[4]].append(excess(g[2])-excess(g[0]))
    for k,items in sorted(interactions.items()):
        values=[mean(v) for v in items.values()]
        print(*k,"n_items",len(values),"interaction",round(mean(values),4),"item_values",[round(v,4) for v in values])
    checks=[abs(r["results"]["exact"]["margin"]-r["clean"]["margin"]) for r in rows]
    print("Max exact-restoration margin error:",max(checks,default=None))
    print("Descriptive pilot only. Compare realized dose, clean margin, families and both seeds before interpretation.")

if __name__=="__main__":
    main()
