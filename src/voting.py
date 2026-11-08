"""Five independent trajectories; voting sees numerical predictions only."""
from itertools import combinations
from decimal import Decimal

TOLERANCE = Decimal("0.01")

SLOTS = (1, 2, 3, 4, 5)

RULE = ("Use the largest cluster where every pair of valid FINAL values has symmetric relative gap "
        "<= 1% (denominator=max absolute value, two zeros agree). Among equal-size clusters prefer "
        "smaller relative span, then lexicographically earlier slot indices. Select the observed lower "
        "median, breaking duplicate-value ties by earliest slot in the cluster. With no agreeing pair "
        "select first valid slot and flag no consensus; all invalid means incorrect. No gold used.")

def vote(predictions):
    valid = {s: Decimal(v) for s, v in enumerate(predictions, 1) if v is not None}
    for size in range(len(valid), 1, -1):
        candidates = []
        for slots in combinations(valid, size):
            values = sorted(valid[s] for s in slots)
            lo, hi = values[0], values[-1]
            scale = max(abs(lo), abs(hi))
            span = (hi - lo) / scale if scale else Decimal(0)
            if span <= TOLERANCE:
                candidates.append((span, slots, values[(size-1)//2]))
        if candidates:
            _, slots, median = min(candidates)
            chosen = next(s for s in slots if valid[s] == median)
            return {"cluster": list(slots), "selected_slot": chosen, "support": size,
                    "state": "unanimous" if size == 5 else "majority" if size >= 3 else "plurality"}
    first = next(iter(valid), None)
    return {"cluster": [first] if first else [], "selected_slot": first, "support": int(first is not None),
            "state": "no_consensus" if first else "no_valid_output"}
