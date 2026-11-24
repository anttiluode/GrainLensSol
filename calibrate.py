"""Clean answer calibration before another grain intervention. No J-lens or perturbation."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import torch
from grain import ITEMS
from grain_v2 import prompts

def swapped(prompt):
    lines=prompt.splitlines()
    a,b=[i for i,s in enumerate(lines) if s.startswith(("A. ","B. "))]
    assert lines[a].startswith("A. ") and lines[b].startswith("B. ")
    lines[a],lines[b]="A. "+lines[b][3:],"B. "+lines[a][3:]
    return "\n".join(lines)

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["grain"],r["variant"])].append(r)
    print("grain variant N spaced_correct alias_correct alias_mass_mean")
    for k,rs in sorted(groups.items()):
        n=len(rs)
        print(*k,n,sum(r["spaced_margin"]>0 for r in rs),
              sum(r["alias_margin"]>0 for r in rs),
              round(sum(r["alias_mass"] for r in rs)/n,4))
    pairs=defaultdict(list)
    for r in rows:
        pairs[(r["source"],r["template"],r["grain"],r["role"])].append(r)
    print("grain pairs both orders correct (alias score)")
    for g in range(3):
        ps=[v for k,v in pairs.items() if k[2]==g and len(v)==2]
        print(g,sum(all(r["alias_margin"]>0 for r in v) for v in ps),"/",len(ps))
    print("Failed original or swapped source questions:")
    for k,v in pairs.items():
        if k[3]=="source" and not all(r["alias_margin"]>0 for r in v):
            print(k,[(r["correct"],round(r["spaced_margin"],2),round(r["alias_margin"],2),r["top"][:3]) for r in v])

@torch.no_grad()
def measure(lm,prompt,correct):
    txt=lm.chat_text([("user",prompt)],"Answer:")
    ids=lm.encode(txt)
    x=torch.tensor([ids],device=lm.device)
    logp=lm.model(input_ids=x,use_cache=False).logits[0,-1].float().log_softmax(-1)
    spaced=[]
    aliases=[]
    for letter in ("A","B"):
        group=set()
        for prefix in (" ","","\n"):
            full=lm.encode(txt+prefix+letter)
            if len(full)==len(ids)+1 and full[:-1]==ids:
                group.add(full[-1])
                if prefix==" ": spaced.append(full[-1])
        if not group: raise RuntimeError("No single-token answer aliases")
        aliases.append(sorted(group))
    if len(spaced)!=2 or set(aliases[0])&set(aliases[1]):
        raise RuntimeError("Ambiguous answer tokenization")
    scores=[float(torch.logsumexp(logp[g],0)) for g in aliases]
    other=1-correct
    top=logp.topk(8)
    return dict(spaced_margin=float(logp[spaced[correct]]-logp[spaced[other]]),
                alias_margin=scores[correct]-scores[other],alias_logp=scores,
                alias_mass=float(logp[aliases[0]+aliases[1]].exp().sum()),
                spaced_ids=spaced,aliases=aliases,
                top=[(lm.decode([int(i)]),round(float(v.exp()),6)) for v,i in zip(top.values,top.indices)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision",default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--qimag",default="../Qwenimagination")
    p.add_argument("--out",default="runs/calibration.json")
    p.add_argument("--quick",action="store_true")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        for i in range(len(ITEMS)):
            for g,c,src,other,dp in prompts(i,0):
                assert swapped(swapped(src))==src
                assert swapped(src).splitlines()[0]==src.splitlines()[0]
        print("PASS: option permutation round trip (no model).");return
    dest=Path(a.out)
    if dest.exists():p.error("Refusing to overwrite output")
    sys.path.insert(0,str(Path(a.qimag).resolve()))
    from qimag.model import Lm
    from transformers import AutoTokenizer,AutoModelForCausalLM
    dtype=torch.float16 if a.device.startswith("cuda") else torch.float32
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=dtype).to(a.device)
    lm=Lm(model,tok,a.device,a.model)
    rows=[]
    for i in ([0,2,4,8] if a.quick else range(len(ITEMS))):
        for t in ([0] if a.quick else [0,1]):
            for g,c,src,other,dp in prompts(i,t):
                for role,name,question,correct in (("source",ITEMS[i][0],src,c),("donor",other,dp,1-c)):
                    for variant,pr,answer in (("original",question,correct),("swapped",swapped(question),1-correct)):
                        rows.append(dict(source=ITEMS[i][0],item=name,template=t,grain=g,role=role,
                                         variant=variant,correct=answer,prompt=pr,**measure(lm,pr,answer)))
        print(ITEMS[i][0],len(rows),flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(config=dict(model=a.model,revision=a.revision,device=a.device,
                     dtype=str(dtype),quick=a.quick),rows=rows),indent=1,allow_nan=False),encoding="utf-8")
    summarize(rows)
    print("Saved",dest)

if __name__=="__main__":
    main()
