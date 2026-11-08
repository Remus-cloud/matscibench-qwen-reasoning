"""Gate the local calculator on a closed formula, traceable inputs and numerical work."""
import ast
import json
import math
import re
import unicodedata

import pint
from protocol import messages as make_messages
from quantity_calculator import UNITS, as_unit, calculate
AFTER_TOOL = ("Continue your worked solution using the calculator result if appropriate. If it returned "
              "an error, check the relation and units and finish the calculation yourself. "
              "Conclude with FINAL = <number> in the requested unit.")


def worksheet_result(worksheet, target_unit, *, allow_assignment=False):
    trace = {"worksheet": worksheet, "target_unit": target_unit}
    try:
        if allow_assignment:
            statements = ast.parse(worksheet.replace("^", "**")).body
            last = statements[-1] if statements else None
            if isinstance(last, ast.Assign) and len(last.targets) == 1 and isinstance(last.targets[0], ast.Name):
                worksheet += "\n" + last.targets[0].id
                trace["returned_variable"] = last.targets[0].id
        trace["value"] = calculate(worksheet, target_unit)
    except (ValueError, SyntaxError, RecursionError) as exc:
        trace["error"] = str(exc)
    return trace


OFFER = """Solve normally and give a worked solution and a standalone FINAL = <number>. Only if a
single scientific relation is established, all numerical inputs are explicit, and evaluating it
needs multi-step arithmetic, a nonlinear function or a real unit conversion, you may request one
calculator check. If data or material constants are missing, a convention is uncertain, or the
arithmetic is simple, finish normally. Always give your own answer, even if you request a check.
Immediately BEFORE the FINAL line you may insert one fenced ```request JSON object:
{"formula":"distance/time","inputs":{"distance":"3 km","time":"5 min"}}
Use a single arithmetic expression for the requested final quantity, variables and quoted
quantities copied directly from the question, including units when present. Use *, /, **, sqrt,
ln, log10, exp, abs. Only simple structural coefficients and these named constants are available:
pi, eps0, elem_charge, NA, Rgas, kB. No guessed constants or intermediate values as inputs.
Use fractions for percent targets; the calculator converts the result to the requested unit."""

CONSTANTS = {"eps0": (8.8541878128e-12, "F/m"), "elem_charge": (1.602176634e-19, "C"),
             "NA": (6.02214076e23, "1/mol"), "Rgas": (8.31446261815324, "J/(mol*K)"),
             "kB": (1.380649e-23, "J/K")}
FUNCTIONS = {"sqrt", "abs", "ln", "log10", "exp"}
COEFFICIENTS = {0, 0.5, 1, 2, 3, 4}
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
FENCE = re.compile(r"```([^\n`]*?)\r?\n(.*?)\r?\n```", re.S)
UNIT_TOKEN = r"[A-Za-zµμΩ%]+(?:\^[+-]?\d+)?"
PHYSICAL_QUANTITY = re.compile(
    rf"(?<![\w.^])({NUMBER})\s*({UNIT_TOKEN}(?:[*/]{UNIT_TOKEN})*)(?![\w/^*])")


def normalize(text):
    text = unicodedata.normalize("NFKC", text.replace("²", "^2").replace("³", "^3").replace("−", "-"))
    for a, b in [(r"\mathrm", ""), (r"\text", ""), (r"\left", ""), (r"\right", ""),
                 (r"\times", "*"), ("×", "*"), (r"\cdot", "*"), (r"\circ", "°"),
                 (r"\AA", "angstrom"), (r"\Omega", "ohm"), (r"\%", "%"), (r"\,", " "),
                 ("~", " "), ("$", ""), ("{", ""), ("}", "")]:
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\s*([*/^])\s*", r"\1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\*10\^([+-]?\d+)", r"\1e\2", text)
    text = re.sub(r"(?<![\w.])10\^([+-]?\d+)", r"1e\1", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=[A-Za-z])", " ", text)
    text = re.sub(r"\^?°\s*C", "degC", text)
    text = re.sub(r"\b(wt|mol)\s*%", lambda m: "wt%" if m[1] == "wt" else "mol %", text)
    return " ".join(text.split())


def read_quantity(quote):
    m = re.fullmatch(rf"({NUMBER})\s*(.*)", quote)
    if not m:
        raise ValueError("input needs one quoted numerical quantity")
    number, unit_text = m.groups()
    magnitude = float(number)
    if not math.isfinite(magnitude):
        raise ValueError("non-finite input")
    return number, unit_text, magnitude, as_unit(unit_text)


def quantity_continues(rest):
    rest = rest.lstrip()
    if rest.startswith(("/", "^", "*", "%")) or re.match(r"[.,]\d", rest):
        return True
    word = re.match(r"[A-Za-zµμΩ]+", rest)
    if word:
        try:
            as_unit(word[0])
            return True
        except (ValueError, pint.errors.PintError):
            pass
    return False


def quoted_quantity(source, question):
    number, unit_text, magnitude, unit = read_quantity(normalize(source))
    # Let a space between a value and its unit vary, but never match a partial value or unit.
    pattern = r"(?<![\w.])" + re.escape(number) + r"\s*" + re.escape(unit_text) + r"(?![\w/^*])"
    for hit in re.finditer(pattern, question):
        if not quantity_continues(question[hit.end():]):
            return magnitude, unit
    # A unit conversion can be checked without trusting a converted value supplied by the model.
    if unit_text:
        for match in PHYSICAL_QUANTITY.finditer(question):
            if quantity_continues(question[match.end():]):
                continue
            try:
                original = UNITS.Quantity(float(match[1]), as_unit(match[2]))
                if math.isclose(original.to(unit).magnitude, magnitude, rel_tol=1e-10):
                    return magnitude, unit
            except (ValueError, pint.errors.PintError):
                continue
    raise ValueError("quantity or its unit not supported by the question")


def infer_unit(node, units):
    one = UNITS.dimensionless
    if isinstance(node, ast.Name) and node.id in units:
        return units[node.id]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float) and node.value in COEFFICIENTS:
        return one
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return infer_unit(node.operand, units)
    if isinstance(node, ast.BinOp):
        left, right = infer_unit(node.left, units), infer_unit(node.right, units)
        if isinstance(node.op, (ast.Add, ast.Sub)) and left.dimensionality == right.dimensionality:
            return left
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and right.dimensionless:
            if left.dimensionless:
                return one
            if isinstance(node.right, ast.Constant):
                return left ** node.right.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords:
        unit = infer_unit(node.args[0], units)
        if node.func.id == "sqrt":
            return unit ** 0.5
        if node.func.id == "abs":
            return unit
        if node.func.id in FUNCTIONS and unit.dimensionless:
            return one
    raise ValueError("unbound value, unsupported formula, or incompatible dimensions")


def decide(text, question, target):
    blocks = [body for lang,body in FENCE.findall(text)
              if lang.strip().lower() == "request" or
              (lang.strip().lower() in ("json","") and '"formula"' in body and '"inputs"' in body)]
    if not blocks:
        reason = "malformed_request" if "```request" in text else "no_request"
        return {"accepted": False, "reason": reason}
    try:
        if len(blocks) != 1 or len(blocks[0]) > 3500:
            raise ValueError("expected one bounded request")
        request = json.loads(blocks[0])
        formula, inputs = request["formula"], request["inputs"]
        if not isinstance(formula, str) or not isinstance(inputs, dict) or not 1 <= len(inputs) <= 15:
            raise ValueError("expected a formula and named inputs")
        expr = ast.parse(formula.strip().replace("^", "**"), mode="eval").body
        nodes = list(ast.walk(expr))
        if len(nodes) > 160:
            raise ValueError("formula too long")
        names = {n.id for n in nodes if isinstance(n, ast.Name)}
        if any(not re.fullmatch("[a-zA-Z][a-zA-Z0-9_]*", k) or k in FUNCTIONS or k == "pi"
               for k in inputs) or not inputs.keys() <= names:
            raise ValueError("invalid or unused input")
        normalized = normalize(question)
        bindings = {k: quoted_quantity(v, normalized) for k,v in inputs.items() if k not in CONSTANTS}
        if not bindings:
            raise ValueError("no numerical input from the question")
        for k in inputs.keys() & CONSTANTS.keys():
            _, _, magnitude, unit = read_quantity(normalize(inputs[k]))
            expected, canonical_unit = CONSTANTS[k]
            converted = UNITS.Quantity(magnitude, unit).to(canonical_unit).magnitude
            if not math.isclose(converted, expected, rel_tol=1e-4):
                raise ValueError("named constant disagrees with its canonical value")
        bindings.update({k: (v, as_unit(u)) for k,(v,u) in CONSTANTS.items() if k in names})
        natural = infer_unit(expr, {"pi": UNITS.dimensionless, **{k:u for k,(_,u) in bindings.items()}})
        destination = as_unit(target)
        if natural.dimensionality != destination.dimensionality:
            raise ValueError("formula output dimension differs from target")
        ops = sum(isinstance(n, ast.BinOp) for n in nodes)
        nonlinear = any((isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)
                         and not (isinstance(n.right, ast.Constant) and n.right.value in (0,1)))
                        or (isinstance(n, ast.Call) and n.func.id != "abs") for n in nodes)
        conversion = natural != destination and any(
            not math.isclose(UNITS.Quantity(v, natural).to(destination).magnitude, v, abs_tol=1e-12)
            for v in (0, 1))
        if not (conversion or nonlinear or (ops >= 2 and len(names & bindings.keys()) >= 2)):
            raise ValueError("only simple arithmetic in the requested unit")
        lines = [f"{k} = Q({v!r}, '{u}')" for k,(v,u) in bindings.items()]
        reason = "conversion" if conversion else "nonlinear" if nonlinear else "multi_step"
        return {"accepted": True, "reason": reason, "formula": formula, "inputs": inputs,
                "worksheet": "\n".join([*lines, formula])}
    except (ValueError, TypeError, KeyError, AttributeError, SyntaxError, RecursionError, pint.errors.PintError) as exc:
        return {"accepted": False, "reason": "rejected", "detail": str(exc)[:250], "proposal": blocks[0]}


def solve(question, ask, total_tokens):
    messages = make_messages(question, "cot")
    messages[0]["content"] += "\n" + OFFER
    first = ask(messages, total_tokens, 1)
    turns = [first]
    decision = {"accepted": False, "reason": "first_turn_" + first["finish_reason"]}
    trace = None
    final = first
    remaining = total_tokens - first["usage"]["completion_tokens"]
    if first["finish_reason"] == "stop" and remaining > 0:
        decision = decide(first["response"], question["question"], question["unit"])
    if decision["accepted"]:
        trace = worksheet_result(decision["worksheet"], question["unit"])
        feedback = {k:v for k,v in trace.items() if k in ("value", "target_unit", "error")}
        messages += [{"role": "assistant", "content": first["response"]},
                     {"role": "user", "content": "Local calculator returned " +
                      json.dumps(feedback, ensure_ascii=False) + "\n" + AFTER_TOOL}]
        final = ask(messages, remaining, 2)
        turns.append(final)
    return {"turns": turns, "tool": trace, "gate": decision, "response": final["response"],
            "finish_reason": final["finish_reason"]}
