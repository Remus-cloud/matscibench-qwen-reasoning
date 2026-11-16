"""Question-only physical constraints and two-stage checked arithmetic."""
import ast
import json
import math
from pint.errors import PintError
from physical_units import unit_for
from quantity_calculator import as_unit, calculate
from live_client import completion
from replay_io import write
from compat_calculator import calculate as extended_calculate

PLAN_CAP, REVIEW_CAP = 1536, 512
CLOSE, FAR = .05, .10
PLAN = """Independently check a materials-science numerical problem using the question and standard
physical identities. Identify what is being counted/measured, units, relevant regime and physical constraints.
Never invent missing material data, graph readings or unspecified conventions. If a unique relation and its
inputs are justified, give a worksheet calculating the requested value in mode EXACT. If only a necessary
upper or lower bound follows, use BOUNDS with available bound(s). Use ABSTAIN for insufficient information.
A plausible estimate is not a necessary bound. List givens, units and assumptions before the worksheet.
Setup is a single line of plain variable assignments separated by semicolons, or NONE. Value, Lower and Upper
are single arithmetic expressions using those variables. Write NONE for unused fields.
Supported: + - * / **, pi, sqrt, abs, exp, ln, log10, sin, cos, tan. Represent physical inputs as Q(value,"unit")
in Pint syntax; e.g. Q(2,"m") / Q(4,"s"). Use kelvin for absolute temperature ratios; trig accepts radians.
Use plain numbers for counts, fractions and geometric coefficients. Do not use imports, functions, lists,
chained assignments or explanatory prose in the worksheet. Give a real calculation, not a guessed number
tagged with units. Use the supplied calculator target unit if present. If no unit is supplied, infer the
intended numeric output unit from the question itself; abstain if ambiguous."""
LABELS = ("Target","Constraints","Givens","Relation","Mode","Unit","Setup","Value","Lower","Upper")
PLAN_GRAMMAR = r'''root ::= "Target: " line "\nConstraints: " line "\nGivens: " line "\nRelation: " line "\nMode: " ("EXACT" | "BOUNDS" | "ABSTAIN") "\nUnit: " line "\nSetup: " line "\nValue: " line "\nLower: " line "\nUpper: " line
line ::= [^\r\n]+'''
REVIEW = """Audit the proposed independent physics worksheet against the problem alone. The calculator checks
arithmetic and unit conversion, not the choice of physical model. Identify a concrete check on the governing
relation and interpretation of the givens. Check the requested quantity, unit/scale, geometry, assumptions,
directions, missing empirical data and bounds. Reject an unjustified relation, wrong counting convention,
unit/target mismatch or a guessed bound. Approve only if the calculation and its mode are physically justified.
Reply with one concise Check: line and then DECISION = APPROVE or DECISION = REJECT."""
REVIEW_GRAMMAR = r'''root ::= "Check: " [^\r\n]+ "\nDECISION = " ("APPROVE" | "REJECT")'''

def unpack(output):
    lines=output["content"].strip().splitlines()
    if output["stop"]=="limit" or len(lines)!=len(LABELS):
        return None
    if not all(s.startswith(k+": ") for s,k in zip(lines,LABELS)):
        return None
    return {k:s[len(k)+2:].strip() for k,s in zip(LABELS,lines)}

def is_derived(worksheet):
    # Follow assignments so an unrelated calculation cannot justify a literal final number.
    body=ast.parse(worksheet.replace("^","**")).body
    assigned={s.targets[0].id:s.value for s in body[:-1] if isinstance(s,ast.Assign)
              and len(s.targets)==1 and isinstance(s.targets[0],ast.Name)}
    def has_operation(node,visited):
        if isinstance(node,ast.BinOp) or (isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                                         and node.func.id!="Q"):
            return True
        if isinstance(node,ast.Name) and node.id in assigned and node.id not in visited:
            return has_operation(assigned[node.id],visited|{node.id})
        return any(has_operation(c,visited) for c in ast.iter_child_nodes(node))
    return bool(body and isinstance(body[-1],ast.Expr) and has_operation(body[-1].value,set()))

def execute(plan,unit,calculator=calculate):
    if plan is None:
        return dict(status="no_complete_plan")
    if plan["Mode"]=="ABSTAIN":
        return dict(status="abstained")
    values={}
    try:
        target=unit or plan["Unit"]
        if as_unit(plan["Unit"])!=as_unit(target):
            raise ValueError("Declared unit differs from the requested calculator unit")
        setup="" if plan["Setup"].upper()=="NONE" else plan["Setup"]+"\n"
        for key in ("Value","Lower","Upper"):
            if plan[key].upper()=="NONE":
                continue
            worksheet=setup+plan[key]
            value=calculator(worksheet,target)
            if key=="Value" and not is_derived(worksheet):
                raise ValueError("Standalone value not derived from a calculation")
            values[key]=value
        if (plan["Mode"]=="EXACT")!=("Value" in values) or not values:
            raise ValueError("Missing or contradictory evidence for declared mode")
        lo,hi=values.get("Lower",-math.inf),values.get("Upper",math.inf)
        if lo>hi or ("Value" in values and not lo<=values["Value"]<=hi):
            raise ValueError("Calculated point conflicts with bounds")
        return dict(status="calculated",unit=target,values=values)
    except (ValueError,TypeError,SyntaxError,RecursionError,ArithmeticError,PintError) as error:
        return dict(status="calculation_error",error=str(error),partial_values=values)

def verify(folder,n,question,raw_unit,sig):
    folder.mkdir(parents=True,exist_ok=True)
    unit=unit_for(raw_unit)
    user=f"Problem\n{question}\n\nRequested unit as printed: {raw_unit or '(unspecified)'}"
    if unit:
        user+=f"\nCalculator output unit: {unit}"
    chat=[dict(role="system",content=PLAN),dict(role="user",content=user)]
    generated=completion(folder/"plan.json",chat,PLAN_GRAMMAR,PLAN_CAP,sig)
    plan=unpack(generated)
    calc=execute(plan,unit) if unit is not None else dict(status="unclear_requested_unit")
    repaired=False
    if calc["status"]=="calculation_error":
        instruction=("The local calculator returned this error: "+calc["error"]+
                     ". Repair the syntax, unit representation or worksheet consistency once. Preserve "
                     "the physical model and input data. Use ABSTAIN if unjustified. Return all fields.")
        repair=chat+[dict(role="assistant",content=generated["content"]),dict(role="user",content=instruction)]
        generated=completion(folder/"repair.json",repair,PLAN_GRAMMAR,PLAN_CAP,sig)
        plan=unpack(generated)
        initial=calc
        calc=execute(plan,unit)
        calc["initial_attempt"]=initial
        repaired=True
    if calc["status"]=="calculation_error":
        calc=execute(plan,unit,extended_calculate)
    status=calc["status"]
    review=None
    if status=="calculated":
        audit=[dict(role="system",content=REVIEW),dict(role="user",content=user+
               "\n\nProposed worksheet\n"+generated["content"]+"\n\nLocal calculator: "+
               json.dumps({"unit":calc["unit"],"values":calc["values"]}))]
        review=completion(folder/"audit.json",audit,REVIEW_GRAMMAR,REVIEW_CAP,sig)
        status="rejected"
        if review["stop"]=="limit":
            status="no_complete_audit"
        elif review["content"].strip().splitlines()[-1]=="DECISION = APPROVE":
            status="approved"
    evidence=dict(protocol=sig,number=n,question=question,requested_unit=raw_unit,plan=plan,
                  calculator=calc,audit=review["content"] if review else None,repaired=repaired,status=status)
    write(folder/"evidence.json",evidence)
    return evidence
