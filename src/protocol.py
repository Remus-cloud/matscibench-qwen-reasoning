"""Frozen classroom prompts and numerical grading."""
import re
from decimal import Decimal, InvalidOperation

SYSTEM = ("You solve materials science problems. End with a separate line: FINAL = <number>. "
          "Use one signed decimal or e-notation number only on that line, without units or LaTeX.")

DIRECT_SYSTEM = ("You are a numeric answer-only API for materials science. Your entire visible reply must "
                 "be exactly one line matching FINAL = <signed decimal or e-notation number>. "
                 "Start immediately with FINAL = and stop after the number. Do calculations silently. "
                 "Do not output an explanation, a heading, a unit, or intermediate calculations.")

METHODS = {
    "direct": "Give only the final answer line.",
    "cot": "Before the final answer line, give a concise worked solution with the governing "
           "equation, substitution with units, and the calculation."
}

FINAL = re.compile(r"FINAL\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*", re.ASCII | re.I)

UNIT_PROMPT = "Express the answer in this unit: "

def messages(row, method):
    question = row["question"]
    if row["unit"].strip():
        question += "\n" + UNIT_PROMPT + row["unit"]
    return [{"role": "system", "content": DIRECT_SYSTEM if method == "direct" else SYSTEM},
            {"role": "user", "content": question + "\n\n" + METHODS[method]}]

def grade(response, gold, reason):
    if reason != "stop":
        return None, False, "truncated" if reason == "length" else reason
    lines = response.strip().splitlines()
    match = FINAL.fullmatch(lines[-1].strip()) if lines else None
    if not match:
        return None, False, "format"
    value = match[1]
    try:
        predicted, target = Decimal(value), Decimal(gold)
        tolerance = abs(target) * Decimal(".05") if target else Decimal("1e-12")
        correct = abs(predicted - target) <= tolerance
    except (InvalidOperation, ArithmeticError):
        correct = False
    return value, correct, "correct" if correct else "numeric"
