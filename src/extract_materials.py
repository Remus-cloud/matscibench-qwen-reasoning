"""Typed, question-only materials definition retrieval and field extraction."""
import json
import re
import materials_tools as mt
import materials_givens as givens
from replay_io import read as load, write
from live_client import completion

SYSTEM="""Read the problem alone. Choose a retrieved material definition only when its solver calculates exactly the quantity asked, with adequate physical assumptions and complete givens. Otherwise return SKIP. The computer will do the arithmetic; do not calculate, restate the answer or invent a number. Return the solver name and the fields in the listed order separated by vertical bars. For each numeric field copy the exact printed literal and select its unit (one for dimensionless, percent for percentages). Respect common units in headers and lists. Use - for absent optional fields. For an explicit scalar unit conversion enter its SOURCE unit. For a mixture f1 must be the stated volume fraction of phase E1, label phases accordingly. rho1/rho2 only for specific modulus, use one for specific gravities. For stress-strain points, distinguish engineering from true stresses AND strains; an ordinary reported permanent elongation is engineering strain, unless stated true. For electronic displacement, explicit charge means the magnitude of one elementary charge when provided; omit if unstated. For Schottky defects determine whether pairs or individual vacancies are asked for. If the problem simply asks for number of Schottky defects, use pair count. For ferrites require a saturation/ordered sublattice moment with an unambiguous basis; no paramagnetic effective moments. Use - for standard molar mass, charge and neutral-atom electron count when not explicitly supplied. Do not silently assume other missing numerical inputs."""
GUARD = {
    "spinel":r"partially|canting|noncollinear|effective moment|paramagnetic|measured saturation",
    "linear_count":r"ratio of.*densit|planar density",
    "flow":r"after necking|post.neck|at the neck|cyclic loading|compressive",
    "composite":r"unknown loading|loading (?:are |is )?unknown|morphology and loading are unknown",
}
def literal(x): return json.dumps(x,ensure_ascii=False)

def schema(names,question):
    # Fixed layout and closed, input-derived vocabulary, no executable text.
    grounded=givens.options(question)
    def alts(xs): return " | ".join(map(literal,xs))
    grammar=['root ::= '+ ' | '.join([literal("SKIP")]+[n.replace("_","-") for n in names])]
    docs=[]
    for n in names:
        c=mt.CARDS[n];parts=[literal(n)]
        for field in c.fields.split():
            key,typ=field.split(":")
            parts.extend([literal("|"),"t-"+typ.replace("_","-")])
        grammar.append(n.replace("_","-")+' ::= '+" ".join(parts))
        docs.append(n+" | "+c.fields+"\n"+c.definition)
    kinds=sorted({field.split(":")[1] for n in names for field in mt.CARDS[n].fields.split()})
    for typ in kinds:
        production=alts(mt.OPTIONS[typ].split() if typ in mt.ENUM else grounded[typ])
        grammar.append("t-"+typ.replace("_","-")+" ::= "+literal("-")+((" | "+production) if production else ""))
    docs.append("Allowed descriptors: "+", ".join(k+"={"+mt.OPTIONS[k]+"}" for k in kinds))
    return "\n".join(grammar),"\n".join(docs)

def output_unit(raw,name,q,fields):
    if raw and raw.strip(): return raw
    # Only directly indicated units, or a pressure unit common to all supplied flow points.
    txt=mt.clean(q)
    if name=="flow" and fields["s1"]!="-" and fields["s2"]!="-" and fields["s1"].split()[1]==fields["s2"].split()[1]:
        return fields["s1"].split()[1]
    if name=="spinel" and re.search(r"Bohr magneton|mu_?B",txt,re.I): return "BM"
    if name=="polarization" and re.search(r"C\s*/\s*m\s*(?:\^\s*)?2",txt): return "C/m2"
    raise ValueError("no unambiguous requested unit")

def analyze(q,u,folder,sig):
    path=folder/"evidence.json"
    if path.exists():
        prev=load(path)
        if (prev["question"],prev["requested_unit"],prev["signature"])!=(q,u,sig): raise RuntimeError("cache mismatch")
        return prev
    hits=mt.retrieve(q)
    record=dict(signature=sig,question=q,requested_unit=u,status="outside_coverage",
                retrieved=[dict(name=n,definition=mt.CARDS[n].definition,sources=mt.CARDS[n].source) for n in hits])
    if hits:
        grammar,docs=schema(hits,q)
        messages=[dict(role="system",content=SYSTEM),dict(role="user",content="Problem: "+mt.clean(q)+"\nRequested unit: "+str(u or "(unspecified)")+"\nDefinition cards and field order:\n"+docs)]
        folder.mkdir(parents=True,exist_ok=True)
        reply=completion(folder/"extraction.json",messages,grammar,256,sig)
        line=reply["content"].strip();record["extraction"]=line;record["status"]="abstained"
        if reply["stop"]!="limit" and line!="SKIP":
            parts=line.split("|");name=parts[0]
            if name not in hits: raise RuntimeError("grammar violated")
            keys=[f.split(":")[0] for f in mt.CARDS[name].fields.split()]
            if len(parts)!=len(keys)+1: raise RuntimeError("malformed fields")
            fields=dict(zip(keys,parts[1:]));record.update(tool=name,extracted_fields=fields)
            try:
                if name in GUARD and re.search(GUARD[name],mt.clean(q),re.I): raise ValueError("outside applicability")
                fields=givens.bind(name,q,u,fields)
                record["fields"]=fields
                target=output_unit(u,name,q,fields)
                calc=mt.solve(name,q,target,fields)
                record.update(status="calculated",calculation=calc)
            except (ValueError,ArithmeticError,TypeError,mt.verifier.PintError) as exc:
                record.update(status="abstained",detail=str(exc))
    write(path,record)
    return record
