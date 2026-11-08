"""Rebuild the standalone four-group replay from the shareable result records."""
import json
from pathlib import Path

from compare import ARMS, ROOT, records
from protocol import grade
from voting import vote


def build():
    rows = json.loads((ROOT/"data/test.json").read_bytes())["rows"]
    items = records(ROOT/"results/questions.jsonl")
    summary = json.loads((ROOT/"results/summary.json").read_bytes())
    if len(rows) != 120 or len({row["qid"] for row in rows}) != 120 or len(items) != len(rows):
        raise ValueError("Expected 120 unique frozen test items")
    for row,item in zip(rows,items):
        if any(row[k] != item[k] for k in ("qid","question","unit","answer","difficulty_level")):
            raise ValueError(f"Frozen item mismatch: {row['qid']}")
        for method in ARMS[:3]:
            rec = item[method]
            pred, ok, status = grade(rec["responses"][-1], str(row["answer"]), "stop")
            if (pred, ok, status) != (rec["prediction"], rec["correct"], rec["grade_status"]):
                raise ValueError(f"Recorded {method} grade differs at {row['qid']}")
        sc = item["sc5"]
        votes = [grade(answer, str(row["answer"]), "stop") for answer in sc["responses"]]
        values = [p for p,_,_ in votes]
        selected = vote(values)
        slot = selected["selected_slot"]
        if (len(votes) != 5 or values != sc["samples"] or selected != sc["vote"]
                or sc["prediction"] != (values[slot-1] if slot else None)
                or sc["correct"] != (votes[slot-1][1] if slot else False)):
            raise ValueError(f"Recorded five-vote result differs at {row['qid']}")
    for key in ARMS:
        if sum(r[key]["correct"] for r in items) != summary["methods"][key]["correct"]:
            raise ValueError(f"Summary mismatch: {key}")
    template = (ROOT/"demo/template.html").read_text(encoding="utf-8")
    data = json.dumps(items, ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
    if template.count("/*__RECORDS__*/") != 1:
        raise ValueError("Expected one data insertion point")
    (ROOT/"demo/index.html").write_text(template.replace("/*__RECORDS__*/",data), encoding="utf-8", newline="\n")
    print("Verified all four groups. Open demo/index.html")


if __name__ == "__main__":
    build()
