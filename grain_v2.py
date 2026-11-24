"""Corrected activation-grain pilot. See PROTOCOL_V2.md. V1 remains unchanged."""
import argparse
import json
import sys
from pathlib import Path
import torch
from grain import ITEMS, CAL, quant, rotate, rand_like_norm, compress, prepare, forward, measures, self_test

# Fine alternatives do not contain either item name.
PROPS = [
    ("typically produces acorns", "often has papery peeling bark"),
    ("typically has prickly woody stems", "typically grows from a bulb"),
    ("typically hunts large prey in packs", "typically hunts small rodents alone"),
    ("has a large hooked beak for tearing flesh", "has a small seed-cracking beak"),
    ("drives nails into wood", "cuts boards into pieces"),
    ("typically moves by pedaling", "typically moves using a motor"),
]
# Choose donor with opposite correct answer for each grain. Same question/options.
MID_DONORS = [2,3,0,1,6,7,4,5,10,11,8,9]
BROAD_DONORS = [4,5,6,7,0,1,2,3,4,5,6,7]
GROUPS = {
    "category": "plant animal tree flower bird mammal tool vehicle artifact organism machine instrument".split(),
    "leaf": [i[0] for i in ITEMS],
    "independent": "music cloud river window ocean paper clock mountain circle bridge shadow stone".split(),
}

def prompts(i, t):
    name, mid, broad, nf, mf, bf = ITEMS[i]
    for g, (yes,no,donor) in enumerate(((broad,bf,BROAD_DONORS[i]),(mid,mf,MID_DONORS[i]),
                                      (*((PROPS[i//2]) if i%2==0 else PROPS[i//2][::-1]),i^1))):
        c=(i+g+t)%2
        a,b=(yes,no) if c==0 else (no,yes)
        q = (("Which broad category fits the item?", "Which more specific category fits the item?",
              "Which property is more characteristic of the item?") if t==0 else
             ("What general kind of thing is it?", "What more specific kind is it?",
              "Which description is more typical of the item?"))[g]
        suffix=f".\n{q}\nA. {a}\nB. {b}\nChoose A or B only."
        other=ITEMS[donor][0]
        yield g,c,"Item: "+name+suffix,other,"Item: "+other+suffix

def scaled(d,x,dose):
    n=d.norm(dim=-1,keepdim=True)
    return d*(dose*x.norm(dim=-1,keepdim=True)/n.clamp_min(1e-12))

def j_bases(lm,layers):
    from qimag.lens import compute_jlens
    used={}
    for k,words in GROUPS.items():
        used[k]=[(w,lm.single_token(w)) for w in words]
        used[k]=[(w,t) for w,t in used[k] if t is not None]
        if len({t for w,t in used[k]})<8:
            raise RuntimeError(f"Need >=8 distinct single-token calibration words for {k}")
    ids=list(dict.fromkeys(t for rows in used.values() for w,t in rows))
    J=compute_jlens(lm,ids,CAL,chunk=64,batch=2)
    bases={}
    for l in layers:
        sub={}
        for k,rows in used.items():
            j=J[l,[ids.index(t) for w,t in rows]].float()
            if not torch.isfinite(j).all() or (j.norm(dim=-1)<1e-12).any():
                raise RuntimeError(f"Invalid J rows {k} L{l}")
            j=j/j.norm(dim=-1,keepdim=True)
            _,s,v=torch.linalg.svd(j,full_matrices=False)
            if int((s>s[0]*1e-4).sum())<8:
                raise RuntimeError(f"J rank below 8 for {k} L{l}")
            sub[k]=v[:8].T.contiguous()
        # Allocate 4 directions to each category/leaf source, then orthonormalize.
        mixed=torch.cat((sub["category"][:,:4],sub["leaf"][:,:4]),dim=1)
        u,s,_=torch.linalg.svd(mixed,full_matrices=False)
        if s[-1]<s[0]*1e-4:
            raise RuntimeError("Balanced J span rank deficient")
        sub["balanced"]=u[:,:8].contiguous()
        bases[l]=sub
    return bases,{k:[w for w,t in rows] for k,rows in used.items()}

def summarize(report):
    from collections import defaultdict
    from statistics import mean
    groups=defaultdict(list)
    for r in report["rows"]:
        groups[(r["scope"],r["layer"],r["seed"],r["arm"],r["dose"],r["bits"],r["grain"])].append(r)
    print("scope layer seed arm dose bits grain N clean_acc coarse_acc clean_margin loss excess_vs_noise excess_vs_span actual_dose")
    for k,rs in sorted(groups.items()):
        c=lambda r:r["results"]["coarse"]
        n=lambda r:r["results"]["noise"]
        span=[r["results"]["noise_in_span"]["margin"]-c(r)["margin"] for r in rs if "noise_in_span" in r["results"]]
        vals=[mean(r["clean"]["choice_correct"] for r in rs),mean(c(r)["choice_correct"] for r in rs),
              mean(r["clean"]["margin"] for r in rs),mean(r["clean"]["margin"]-c(r)["margin"] for r in rs),
              mean(n(r)["margin"]-c(r)["margin"] for r in rs),mean(span) if span else float("nan"),
              mean(c(r)["relative_rms"] for r in rs)]
        print(*k,len(rs),*(f"{v:.4f}" for v in vals))
    pairs=defaultdict(list)
    for r in report["transplants"]:
        pairs[(r["scope"],r["layer"],r["grain"])].append(r)
    print("\nDonor gate: clean source AND donor answer correctly; flip rate on eligible pairs")
    for k,rs in sorted(pairs.items()):
        ok=[r for r in rs if r.get("eligible")]
        print(*k,"eligible",len(ok),"/",len(rs),"flip",mean(r["flipped"] for r in ok) if ok else None)
    print("Descriptive pilot; compare clean margins, realized doses and paired items before claims.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--quick",action="store_true")
    p.add_argument("--no-j",action="store_true")
    p.add_argument("--model",default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision")
    p.add_argument("--qimag",default="../Qwenimagination")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",default="runs/v2_gate1.json")
    p.add_argument("--summarize",type=Path)
    args=p.parse_args()
    if args.summarize:
        summarize(json.loads(args.summarize.read_text(encoding="utf-8"))); return
    if args.self_test:
        self_test()
        x=torch.randn(3,16)
        d=scaled(x-quant(x,4),x,.1)
        assert torch.allclose(d.norm(dim=-1),.1*x.norm(dim=-1),atol=1e-6)
        for i in range(len(ITEMS)):
            for g,c,src,other,donor in prompts(i,0):
                assert src.split("\n",1)[1]==donor.split("\n",1)[1]
                assert other!=ITEMS[i][0]
                if g==2:
                    assert all(ITEMS[i][0] not in s.lower().split() for s in src.split("\n")[2:4])
        print("PASS: dose and paired prompt checks (no real model)."); return
    dest=Path(args.out)
    if dest.exists(): p.error("Refusing to overwrite output")
    sys.path.insert(0,str(Path(args.qimag).resolve()))
    from qimag.model import Lm
    from transformers import AutoModelForCausalLM,AutoTokenizer
    dtype=torch.float16 if args.device.startswith("cuda") else torch.float32
    tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,
                                             torch_dtype=dtype).to(args.device)
    model.requires_grad_(False)
    lm=Lm(model,tok,args.device,args.model)
    layers=[lm.n_layers//2] if args.quick else sorted({lm.n_layers//3,lm.n_layers//2,2*lm.n_layers//3})
    seeds=[41] if args.quick else [41,97]
    doses=[.1] if args.quick else [.025,.05,.1,.2]
    bits=[4] if args.quick else [2,4]
    indices=[0,2,4,8] if args.quick else range(len(ITEMS))
    templates=[0] if args.quick else [0,1]
    jb,used=({}, {}) if args.no_j else j_bases(lm,layers)
    rot={s:rotate(lm.d,s,lm.device) for s in seeds}
    report=dict(config=dict(version=2,model=args.model,revision=args.revision,
                resolved_revision=getattr(model.config,"_commit_hash",None),dtype=str(dtype),
                device=args.device,torch=torch.__version__,quick=args.quick,calibration_words=used,
                layers=layers,seeds=seeds,doses=doses,bits=bits),rows=[],transplants=[],complete=False)
    dest.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,allow_nan=False),encoding="utf-8")
        tmp.replace(dest)
    for i in indices:
        name=ITEMS[i][0]
        for t in templates:
            for g,c,prompt,other,dp in prompts(i,t):
                ids,itempos,ans,txt=prepare(lm,name,prompt)
                dids,dpos,dans,dtxt=prepare(lm,other,dp)
                for l in layers:
                    for scope,pos in (("item",itempos),("suffix",list(range(min(itempos),ids.shape[1])))):
                        clean,cap=forward(lm,ids,l,pos)
                        x=cap["x"]
                        base=measures(clean,clean,ans,c)
                        row=dict(item=name,template=t,grain=g,layer=l,scope=scope,prompt=txt,
                                 positions=pos,answer_ids=ans,correct=c,clean=base)
                        identity,_=forward(lm,ids,l,pos,x)
                        if float((identity-clean).abs().max())>.05:
                            raise RuntimeError("Identity hook changed logits")
                        donorpos=dpos if scope=="item" else list(range(min(dpos),dids.shape[1]))
                        tr=dict(item=name,donor=other,template=t,grain=g,layer=l,scope=scope)
                        # Align token positions; never force incompatible lengths.
                        if pos==donorpos and ans==dans and ids.shape==dids.shape:
                            dl,dc=forward(lm,dids,l,donorpos)
                            db=measures(dl,dl,dans,1-c)
                            patched,_=forward(lm,ids,l,pos,dc["x"])
                            pm=measures(patched,clean,ans,c)
                            tr.update(eligible=bool(base["choice_correct"] and db["choice_correct"]),
                                      flipped=pm["margin"]<0,source_margin=base["margin"],
                                      donor_margin=db["margin"],patched_margin=pm["margin"])
                        else:
                            tr.update(eligible=False,reason="unaligned token positions")
                        report["transplants"].append(tr)
                        for s in seeds:
                            arms={"native":None,"rotated":rot[s]}
                            if l in jb:
                                arms.update({f"j_{k}":v for k,v in jb[l].items()})
                                arms["random_rank8"]=rot[s][:,:8]
                                arms["j_balanced_rotated"]=jb[l]["balanced"]@rotate(8,s+77,lm.device)
                            for a,basis in arms.items():
                                for bit in bits:
                                    raw=x-compress(x,bit,basis)
                                    for dose in doses:
                                        d=scaled(raw,x,dose)
                                        r=rand_like_norm(d,s+1000)
                                        candidates={"coarse":x-d,"noise":x-r}
                                        if basis is not None and basis.shape[1]<lm.d:
                                            z=rand_like_norm(d@basis,s+2000)@basis.T
                                            candidates["noise_in_span"]=x-scaled(z,x,dose)*(d.norm(dim=-1,keepdim=True)>0)
                                        result={}
                                        for mode,v in candidates.items():
                                            logits,cp=forward(lm,ids,l,pos,v)
                                            delta=cp["actual"]-x
                                            m=measures(logits,clean,ans,c)
                                            m.update(delta_l2=float(delta.norm()),relative_rms=float(delta.norm()/x.norm().clamp_min(1e-12)))
                                            result[mode]=m
                                        report["rows"].append(dict(**row,arm=a,seed=s,bits=bit,dose=dose,
                                                                  raw_relative_rms=float(raw.norm()/x.norm().clamp_min(1e-12)),results=result))
                        save()
                print(name,t,g,len(report["rows"]),flush=True)
    report["complete"]=True
    save()
    summarize(report)

if __name__=="__main__":
    main()
