"""Replay the frozen selector from archived, label-free evidence; score after selection."""
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from protocol import grade
from voting import vote

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("direct", "cot", "sc", "d3", "final")

def read_rows():
    archive = ROOT / "results/questions.jsonl.gz"
    protocol = json.loads((ROOT / "results/protocol.json").read_bytes())
    with archive.open("rb") as raw:
        if hashlib.file_digest(raw, "sha256").hexdigest() != protocol["archive_sha256"]:
            raise ValueError("Answer archive differs from the frozen protocol")
    with gzip.open(archive, "rt", encoding="utf8") as fp:
        return [json.loads(line) for line in fp]

def classify(value, point):
    def margin(x, factor): return factor*abs(x) if x else 1e-12
    lo,hi,ref=(point.get(k) for k in ("Lower","Upper","Value"))
    if lo is not None and value<lo-margin(lo,.05) or hi is not None and value>hi+margin(hi,.05): return "fails"
    inside=(lo is None or value>=lo) and (hi is None or value<=hi)
    if ref is None: return "passes" if inside else "uncertain"
    error=abs(value-ref)
    if error>margin(ref,.10): return "fails"
    return "passes" if inside and error<=margin(ref,.05) else "uncertain"

def select(predictions,blind,checks):
    # Only candidates' submitted numbers, blind verdicts and physical evidence enter selection.
    both=[r["pick"] for r in blind]
    same=both[0] and both[0]==both[1]
    picked=both[0] if same else "cot"
    route="blind" if same else "default"
    if predictions["cot"] is None and predictions["d3"] is not None and picked!="d3":
        picked,route="d3","format"
    chosen_check=None
    if all(predictions[k] is not None for k in ("cot","d3")):
        for i,c in enumerate(checks):
            statuses={k:classify(float(predictions[k]),c["values"]) for k in ("cot","d3")}
            if set(statuses.values())=={"passes","fails"}:
                arm=next(k for k,status in statuses.items() if status=="passes")
                if arm!=picked: picked,route,chosen_check=arm,c["route"],i
    return picked,route,chosen_check

def audit(rows):
    assert len(rows)==len({r["qid"] for r in rows})==120
    totals=Counter()
    for n,r in enumerate(rows,1):
        assert n==r["n"]
        arms=r["answers"]
        for k in ("direct","cot","d3"):
            a=arms[k]
            assert grade(a["text"],str(r["gold"]),a["finish"])==(a["p"],a["ok"],a["status"])
        sc=arms["sc"]; samples=sc["samples"]
        assert len(samples)==5
        votes=[grade(a["text"],str(r["gold"]),a["finish"])[0] for a in samples]
        assert votes==[a["p"] for a in samples] and vote(votes)==sc["vote"]
        slot=sc["vote"]["selected_slot"]
        assert sc["p"]==(votes[slot-1] if slot else None)
        assert sc["ok"]==(grade(samples[slot-1]["text"],str(r["gold"]),samples[slot-1]["finish"])[1] if slot else False)
        selected,route,check=select({k:arms[k]["p"] for k in ("cot","d3")},r["blind"],r["checks"])
        assert (selected,route,check)==(r["selected"],r["route"],r["check"])
        assert r["final_ok"]==arms[selected]["ok"]
        for k in METHODS: totals[k]+=r["final_ok"] if k=="final" else arms[k]["ok"]
    assert dict(totals)==dict(direct=16,cot=69,sc=77,d3=66,final=85)
    return totals

if __name__=="__main__":
    print(json.dumps(audit(read_rows()),indent=2))

