"""Score a candidate's saved responses against the same frozen questions, offline."""
import argparse
import json
from pathlib import Path

from protocol import grade


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("direct", "cot", "gated", "sc5")


def records(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score(path, split):
    rows = json.loads((ROOT/"data"/f"{split}.json").read_bytes())["rows"]
    gold = {r["qid"]: str(r["answer"]) for r in rows}
    predictions = {}
    for r in records(path):
        qid = r["qid"]
        if qid not in gold or qid in predictions:
            raise ValueError(f"Unknown or duplicate {split} qid: {qid}")
        if not isinstance(r["response"], str) or not isinstance(r["finish_reason"], str):
            raise ValueError(f"Expected response and finish_reason strings for {qid}")
        _, ok, _ = grade(r["response"], gold[qid], r["finish_reason"])
        predictions[qid] = ok
    if split == "dev":
        ref = {r["qid"]: grade(r["response"], gold[r["qid"]], r["finish_reason"])[1]
               for r in records(ROOT/"results/dev_cot.jsonl")}
        controls = {"cot":ref}
    else:
        saved = records(ROOT/"results/questions.jsonl")
        controls = {k:{r["qid"]:r[k]["correct"] for r in saved} for k in ARMS}
    if any(ref.keys() != gold.keys() for ref in controls.values()):
        raise ValueError("Incomplete baseline")
    correct = sum(predictions.values())
    result = {"split":split, "answered":len(predictions), "planned":len(rows), "correct":correct,
              "accuracy":correct/len(rows) if len(predictions)==len(rows) else None,
              "missing":[qid for qid in gold if qid not in predictions], "paired":{}}
    for method, ref in controls.items():
        result["paired"][method] = {
            "gains":[qid for qid, ok in predictions.items() if ok and not ref[qid]],
            "losses":[qid for qid, ok in predictions.items() if not ok and ref[qid]]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("--split", choices=("dev","test"), default="dev")
    args = parser.parse_args()
    print(json.dumps(score(args.responses,args.split), ensure_ascii=False, indent=2))
