"""Frozen-Qwen activation granularity pilot. See PROTOCOL.md."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

# (item, middle, broad, fine foil, middle foil, broad foil)
ITEMS = [
    ("oak", "tree", "plant", "birch", "flower", "animal"),
    ("birch", "tree", "plant", "oak", "flower", "animal"),
    ("rose", "flower", "plant", "tulip", "tree", "animal"),
    ("tulip", "flower", "plant", "rose", "tree", "animal"),
    ("wolf", "mammal", "animal", "fox", "bird", "plant"),
    ("fox", "mammal", "animal", "wolf", "bird", "plant"),
    ("eagle", "bird", "animal", "sparrow", "mammal", "plant"),
    ("sparrow", "bird", "animal", "eagle", "mammal", "plant"),
    ("hammer", "hand tool", "artifact", "saw", "vehicle", "animal"),
    ("saw", "hand tool", "artifact", "hammer", "vehicle", "animal"),
    ("bicycle", "vehicle", "artifact", "car", "hand tool", "animal"),
    ("car", "vehicle", "artifact", "bicycle", "hand tool", "animal"),
]
WORDS = ("plant animal tree flower bird mammal tool vehicle artifact living object "
         "food water person place metal wood large small color movement").split()
CAL = [
    "A quiet room contains several ordinary objects. Some can move and others remain still.",
    "The morning light reaches a garden. Different shapes appear against the wall.",
    "A person travels through the city and notices changes in the environment.",
    "A wooden surface feels different from a metallic surface when touched.",
    "Some organisms live underwater, while others live on dry land.",
    "An object can be described by its material, function, location and size.",
    "The observer compares two examples and considers the properties they share.",
    "A name can refer to a particular example or to a larger collection of related things.",
]


def questions(item, i, template):
    name, mid, broad, nf, mf, bf = item
    for level, (yes, no) in enumerate(((broad,bf),(mid,mf),(name,nf))):
        correct = (i + level + template) % 2
        options = [yes,no] if correct == 0 else [no,yes]
        if template == 0:
            q = ("Which broad category fits the item?", "Which specific category fits the item?",
                 "Which item was named?")[level]
        else:
            q = ("What general kind of thing is it?", "What more specific kind is it?",
                 "What was the exact item?")[level]
        yield level, correct, f"Item: {name}.\n{q}\nA. {options[0]}\nB. {options[1]}\nChoose A or B only."


def quant(x, bits):
    if bits < 2:
        raise ValueError("signed symmetric rounding requires >=2 bits")
    m = 2 ** (bits - 1) - 1
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / m
    return (x / scale).round().clamp(-m,m) * scale


def rotate(d, seed, device):
    # Dense random orthogonal basis; QR on CPU for reproducibility.
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d,d,generator=g)
    q,r = torch.linalg.qr(a)
    q = q * torch.where(r.diag() < 0,-1.,1.)
    return q.to(device)


def rand_like_norm(delta, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.randn(delta.shape,generator=g).to(delta.device)
    return r * (delta.norm(dim=-1,keepdim=True) / r.norm(dim=-1,keepdim=True).clamp_min(1e-12))


def compress(x, bits, basis=None):
    if basis is None:
        return quant(x,bits)
    c = x @ basis
    # Preserve orthogonal complement (including when basis is rectangular).
    return x + (quant(c,bits)-c) @ basis.T


def hs(out):
    return out[0] if isinstance(out,tuple) else out


def replace(out, v):
    return (v,)+out[1:] if isinstance(out,tuple) else v


def self_test():
    x=torch.randn(3,16,generator=torch.Generator().manual_seed(4))
    q=rotate(16,41,"cpu")
    assert torch.allclose(q.T@q,torch.eye(16),atol=2e-6)
    b=q[:,:5]
    for basis in (None,q,b):
        c=compress(x,3,basis)
        d=x-c
        assert torch.allclose(c+d,x,atol=2e-6)
        assert torch.allclose(rand_like_norm(d,42).norm(dim=-1),d.norm(dim=-1),atol=2e-6)
        if basis is not None:
            assert torch.allclose(d,(d@basis)@basis.T,atol=2e-6)
    assert torch.isfinite(quant(torch.zeros_like(x),2)).all()
    layer=torch.nn.Identity()
    clean=layer(x)
    h=layer.register_forward_hook(lambda m,i,o: replace(o,hs(o)+1))
    try:
        assert torch.equal(layer(x),x+1)
    finally:
        h.remove()
    assert torch.equal(layer(x),clean)
    print("PASS: numerical controls, subspace complement, residue, hook cleanup (no model).")


def prepare(lm, name, prompt):
    text=lm.chat_text([("user",prompt)],"Answer:")
    enc=lm.tok(text,add_special_tokens=False,return_offsets_mapping=True)
    start=text.index("Item: "+name)+len("Item: ")
    end=start+len(name)
    pos=[i for i,(a,b) in enumerate(enc["offset_mapping"]) if a<end and b>start]
    ids=enc["input_ids"]
    if not pos:
        raise RuntimeError("No item tokens found")
    answer=[]
    for suffix in (" A"," B"):
        extended=lm.encode(text+suffix)
        if extended[:-1]!=ids:
            raise RuntimeError("Answer token not a clean single-token continuation")
        answer.append(extended[-1])
    if answer[0]==answer[1]:
        raise RuntimeError("Identical answer tokens")
    return torch.tensor([ids],device=lm.device),pos,answer,text


@torch.no_grad()
def forward(lm, ids, layer, pos, replacement=None):
    captured={}
    def hook(m,i,o):
        h=hs(o)
        captured["x"]=h[0,pos].detach().float().clone()
        if replacement is not None:
            new=h.clone()
            new[0,pos]=replacement.to(h.dtype)
            captured["actual"]=new[0,pos].detach().float().clone()
            return replace(o,new)
    handle=lm.layers[layer].register_forward_hook(hook)
    try:
        logits=lm.model(input_ids=ids,use_cache=False).logits[0,-1].float()
    finally:
        handle.remove()
    return logits,captured


def measures(logits, clean, answer, correct):
    a,b=answer[correct],answer[1-correct]
    logp=clean.log_softmax(-1)
    lp=logits.log_softmax(-1)
    return dict(margin=float(logits[a]-logits[b]),
                choice_correct=bool(logits[a]>logits[b]),
                kl=float((logp.exp()*(logp-lp)).sum().clamp_min(0)))


def build_j(lm, layers):
    from qimag.lens import compute_jlens
    ids=list(dict.fromkeys(t for w in WORDS if (t:=lm.single_token(w)) is not None))
    if len(ids)<4:
        raise RuntimeError("Insufficient calibration vocabulary")
    J=compute_jlens(lm,ids,CAL,chunk=64,batch=2)
    bases={}
    for l in layers:
        j=J[l].float()
        if not torch.isfinite(j).all() or (j.norm(dim=-1)<1e-12).any():
            raise RuntimeError(f"Degenerate J at layer {l}")
        j=j/j.norm(dim=-1,keepdim=True)
        _,s,vh=torch.linalg.svd(j,full_matrices=False)
        rank=int((s>s[0]*1e-4).sum())
        if rank<2:
            raise RuntimeError("J rank too small")
        bases[l]=vh[:rank].T.contiguous()
    return bases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--quick",action="store_true")
    p.add_argument("--no-j",action="store_true")
    p.add_argument("--model",default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision",default=None)
    p.add_argument("--qimag",default="../Qwenimagination")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",default="runs/gate1.json")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    dest=Path(args.out)
    if dest.exists():
        p.error("Refusing to overwrite output")
    sys.path.insert(0,str(Path(args.qimag).resolve()))
    from qimag.model import Lm
    from transformers import AutoModelForCausalLM,AutoTokenizer
    dtype=torch.float16 if args.device.startswith("cuda") else torch.float32
    tok=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,
                                              torch_dtype=dtype).to(args.device)
    lm=Lm(model,tok,args.device,args.model)
    model.requires_grad_(False)
    layers=[lm.n_layers//2] if args.quick else sorted({lm.n_layers//3,lm.n_layers//2,2*lm.n_layers//3})
    bits=[2,4] if args.quick else [2,3,4,6,8]
    seeds=[41] if args.quick else [41,97]
    indices=[0,2,4,8] if args.quick else list(range(len(ITEMS)))
    templates=[0] if args.quick else [0,1]
    bases={} if args.no_j else build_j(lm,layers)
    rotations={s:rotate(lm.d,s,lm.device) for s in seeds}
    jrot={(l,s):bases[l]@rotate(bases[l].shape[1],s,lm.device) for l in bases for s in seeds}
    report=dict(config=dict(model=args.model,revision=args.revision,
                resolved_revision=getattr(model.config,"_commit_hash",None),dtype=str(dtype),
                layers=layers,bits=bits,seeds=seeds,quick=args.quick,j_ranks={l:b.shape[1] for l,b in bases.items()},
                torch=torch.__version__),rows=[],complete=False)
    dest.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(report,allow_nan=False,indent=1),encoding="utf-8")
        tmp.replace(dest)
    for i in indices:
        item=ITEMS[i]
        for template in templates:
            prefix=None
            for level,correct,prompt in questions(item,i,template):
                ids,pos,answer,text=prepare(lm,item[0],prompt)
                this=ids[0,:max(pos)+1].tolist()
                if prefix is not None and this!=prefix:
                    raise RuntimeError("Item prefix changed with question")
                prefix=this
                for l in layers:
                    clean,cap=forward(lm,ids,l,pos)
                    x=cap["x"]
                    base=measures(clean,clean,answer,correct)
                    clean_row=dict(item=item[0],family=item[2],template=template,grain=level,
                                   layer=l,positions=pos,prompt=text,answer_ids=answer,
                                   correct=correct,clean=base)
                    # Exact replacement is a hook/no-op control, independent of rounding.
                    check,_=forward(lm,ids,l,pos,x)
                    err=float((check-clean).abs().max())
                    if err>0.05:
                        raise RuntimeError(f"Identity hook changed logits: {err}")
                    for s in seeds:
                        arms={"native":None,"rotated":rotations[s]}
                        if l in bases:
                            arms.update(j=bases[l],j_rotated=jrot[l,s])
                        for arm,basis in arms.items():
                            for b in bits:
                                coarse=compress(x,b,basis).to(dtype).float()
                                d=x-coarse
                                random=rand_like_norm(d,s+1000)
                                candidates={"coarse":coarse,"noise":x-random,
                                            "half_true":coarse+0.5*d,
                                            "half_random":coarse+0.5*rand_like_norm(d,s+2000),
                                            "exact":coarse+d}
                                if basis is not None and basis.shape[1] < x.shape[-1]:
                                    sub_d=d@basis
                                    in_span=rand_like_norm(sub_d,s+1000)@basis.T
                                    in_span=in_span*(d.norm(dim=-1,keepdim=True)/in_span.norm(dim=-1,keepdim=True).clamp_min(1e-12))
                                    candidates["noise_in_span"]=x-in_span
                                result={}
                                for mode,v in candidates.items():
                                    logits,c=forward(lm,ids,l,pos,v)
                                    actual=c["actual"]
                                    m=measures(logits,clean,answer,correct)
                                    m["delta_l2"]=float((actual-x).norm())
                                    m["relative_rms"]=float((actual-x).norm()/x.norm().clamp_min(1e-12))
                                    m["restoration_l2"]=float((actual-coarse).norm())
                                    result[mode]=m
                                report["rows"].append(dict(**clean_row,seed=s,arm=arm,bits=b,
                                                          identity_max_error=err,results=result))
                    save()
                print(item[0],template,level,len(report["rows"]),flush=True)
    report["complete"]=True
    save()
    print("Saved",dest)


if __name__=="__main__":
    main()
