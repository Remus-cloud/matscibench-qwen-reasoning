"""Evaluate a short arithmetic worksheet and convert its result to the requested unit."""
import ast
import math
import operator
import re

import pint


UNITS = pint.UnitRegistry()
UNIT_ALIASES = {
    "": "dimensionless", "%": "percent", "wt%": "percent", "mol %": "percent",
    "per cm³": "1/cm**3", r"$\Omega \cdot \mathrm{cm}$": "ohm*cm",
    r"${}^{\circ}\mathrm{C}$": "degC",
}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow}
TRANSCENDENTAL = {"ln": math.log, "log": math.log, "log10": math.log10, "exp": math.exp,
                  "sin": math.sin, "cos": math.cos, "tan": math.tan}


def as_unit(text):
    name = UNIT_ALIASES.get(text, text).replace("^", "**")
    if len(name) > 100 or not re.fullmatch(r"[\wµμÅ°*/(). +%-]+", name):
        raise ValueError("Invalid unit syntax")
    if any(abs(int(n)) > 64 for n in re.findall(r"\d+", name)):
        raise ValueError("Unit exponent out of range")
    return UNITS.parse_units(name)


def checked_power(base, exponent):
    power = float(exponent)
    if abs(power) > 64:
        raise ValueError("Exponent exceeds 64")
    return base ** power


def value(node, variables):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name) and not node.id.startswith("_"):
        if node.id in variables:
            return variables[node.id]
        return UNITS.Quantity(1, as_unit(node.id))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        arg = value(node.operand, variables)
        return -arg if isinstance(node.op, ast.USub) else arg
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = value(node.left, variables), value(node.right, variables)
        if isinstance(node.op, ast.Pow):
            return checked_power(left, right)
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name == "Q" and len(node.args) == 2:
            unit = node.args[1]
            if isinstance(unit, ast.Constant) and isinstance(unit.value, str):
                return UNITS.Quantity(float(value(node.args[0], variables)), as_unit(unit.value))
        if len(node.args) == 1:
            arg = value(node.args[0], variables)
            if name == "sqrt":
                return checked_power(arg, 0.5)
            if name == "abs":
                return abs(arg)
            if name in TRANSCENDENTAL:
                return TRANSCENDENTAL[name](float(arg))
    raise ValueError("Unsupported expression: " + type(node).__name__)


def calculate(worksheet, target_unit):
    """Assignments followed by one expression. Only arithmetic and named functions are allowed."""
    if len(worksheet) > 5000:
        raise ValueError("Worksheet too long")
    try:
        tree = ast.parse(worksheet.strip().replace("^", "**"))
        if sum(1 for _ in ast.walk(tree)) > 500 or not tree.body or not isinstance(tree.body[-1], ast.Expr):
            raise ValueError("Expected a bounded worksheet ending in an expression")
        variables = {"pi": math.pi}
        for stmt in tree.body[:-1]:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("Only plain variable assignments are allowed")
            name = stmt.targets[0].id
            if name.startswith("_"):
                raise ValueError("Invalid variable name")
            variables[name] = value(stmt.value, variables)
        result = UNITS.Quantity(value(tree.body[-1].value, variables)).to(as_unit(target_unit))
        number = float(result.magnitude)
        if not math.isfinite(number):
            raise ValueError("Non-finite result")
        return number
    except (pint.errors.PintError, SyntaxError, TypeError, ArithmeticError, RecursionError) as exc:
        raise ValueError(f"{type(exc).__name__}: {exc}") from None
