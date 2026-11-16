"""Run the five local conditions on a llama.cpp server with the materials-science LoRA adapter loaded."""
import argparse
import hashlib
import random
import re
from pathlib import Path

from replay import ROOT,select
from replay_io import read,write,digest
from live_client import api,turn,completion
from protocol import messages,grade
from voting import vote

def file_hash(p):
    with Path(p).open("rb") as fp:return hashlib.file_digest(fp,"sha256").hexdigest()

def evaluate(r,folder,sig,adapter,first):
    from worksheets import verify
    from extract_materials import analyze
    from materials_role_binding import check
    n=r["n"];blind=[];answers={}
    for k in ("direct","cot","finetuned","sc"):
        is_sc=k=="sc";samples=[]
        for i in range(1,6 if is_sc else 2):
            raw=turn(folder/(f"{k}-{i}.json"),messages(r,"direct" if k=="direct" else "cot"),16384,.6 if is_sc else 0,
                     6020000+n*10+i if is_sc else 6020,[dict(id=adapter,scale=.25)] if k=="finetuned" else [],sig)
            samples.append(dict(text=raw["content"],finish="length" if raw["stop"]=="limit" else "stop"))
        if is_sc:
            # Numeric extraction uses the submitted FINAL line. Scoring follows selection.
            predictions=[submitted(x) for x in samples];v=vote(predictions);slot=v["selected_slot"]
            answers[k]=dict(samples=samples,vote=v,p=predictions[slot-1] if slot else None)
        else: answers[k]=dict(**samples[0],p=submitted(samples[0]))
    prompts=read(ROOT/"src/judge_prompts.json")
    arms=("cot","finetuned")
    for i,a in enumerate((first,next(s for s in arms if s!=first)),1):
        b=next(s for s in arms if s!=a)
        for arm in (a,b):
            if re.search(r"\bQwen\b|\bLoRA\b|\bfine[- ]?tuned model\b|<\|im_(?:start|end)\|>|</?candidate_[AB]>",answers[arm]["text"],re.I):
                raise ValueError("Candidate contains identifying text or a delimiter")
        body=f"Problem\n{r['question']}\n\nRequested answer unit: {r['unit'] or '(none specified)'}\n\n<candidate_A>\n{answers[a]['text']}\n</candidate_A>\n\n<candidate_B>\n{answers[b]['text']}\n</candidate_B>"
        raw=completion(folder/f"blind-{i}.json",[dict(role="system",content=prompts["blind"]),dict(role="user",content=body)],prompts["blind_grammar"],768,sig)
        m=re.fullmatch(r"Check: [^\r\n]+\nVERDICT = ([AB])",raw["content"].strip())
        blind.append(dict(first=a,pick=({"A":a,"B":b}[m[1]] if m and raw["stop"]!="limit" else None),review=raw["content"]))
    checks=[];q,u=r["question"],r["unit"]
    ws=verify(folder/"worksheet",n,q,u,sig)
    if ws["status"]=="approved":checks.append(dict(route="worksheet",values=ws["calculator"]["values"]))
    for name,e in (("material",analyze(q,u,folder/"material",sig)),("binding",check(q,u))):
        if e["status"]=="calculated":checks.append(dict(route=name,values={"Value":e["calculation"]["value"]}))
    chosen,route,_=select({k:answers[k]["p"] for k in arms},blind,checks)
    scores={}
    for k in ("direct","cot","finetuned","sc"):
        x=answers[k]
        if k=="sc":
            idx=x["vote"]["selected_slot"]
            if idx is None:scores[k]=False;continue
            x=x["samples"][idx-1]
        scores[k]=grade(x["text"],str(r["gold"]),x["finish"])[1]
    return dict(signature=sig,n=n,qid=r["qid"],answers=answers,blind=blind,checks=checks,selected=chosen,route=route,scores=dict(scores,final=scores[chosen]))

def submitted(sample):
    from protocol import FINAL
    lines=sample["text"].rstrip().splitlines()
    m=FINAL.fullmatch(lines[-1].strip()) if sample["finish"]=="stop" and lines else None
    return m[1] if m else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--question",type=int,choices=range(1,121),metavar="1..120")
    ap.add_argument("--out",type=Path,default=ROOT/"runs/evaluation")
    args=ap.parse_args()
    recorded=read(ROOT/"results/protocol.json");weights=read(ROOT/"models/materials-lora/weights.json")
    props=api("/props")
    if file_hash(props["model_path"])!=recorded["base"]:raise ValueError("Base GGUF differs from the recorded BF16 export")
    matches=[a for a in api("/lora-adapters") if file_hash(a["path"])==weights["files"]["adapter-f32.gguf"]]
    if len(matches)!=1:raise ValueError("Load the packaged LoRA GGUF using --lora and --lora-init-without-apply")
    sig=digest([recorded,weights,props.get("build_info"),{p.name:file_hash(p) for p in sorted((ROOT/"src").glob("*.py"))},file_hash(ROOT/"src/judge_prompts.json")])
    positions=["cot"]*60+["finetuned"]*60;random.Random(6020).shuffle(positions)
    split=ROOT/"data/test120.json"
    if hashlib.sha256(split.read_text(encoding="utf8").encode()).hexdigest()!=recorded["test_split"]:
        raise ValueError("Evaluation split differs from the frozen benchmark")
    rows=[dict(r,n=n,gold=r["answer"]) for n,r in enumerate(read(split)["rows"],1) if args.question in (None,n)]
    totals={}
    for r in rows:
        n=r["n"];folder=args.out/f"{n:03}";file=folder/"result.json"
        if file.is_file():
            saved=read(file)
            if saved["signature"]!=sig or saved["qid"]!=r["qid"]:raise ValueError("Saved protocol mismatch")
        else:
            saved=evaluate(r,folder,sig,matches[0]["id"],positions[n-1]);write(file,saved)
        for k,ok in saved["scores"].items():totals[k]=totals.get(k,0)+int(ok)
        print(n,saved["scores"],flush=True)
    print(f"{len(rows)} questions",totals)
if __name__=="__main__":main()

