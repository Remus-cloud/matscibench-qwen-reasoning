"""Question-only field binding. Each accepted calculation retains values and their source roles."""
from dataclasses import dataclass
import math
import re

import materials_tools as mt
import materials_givens as givens

class Unresolved(ValueError):
    pass

def require(condition,message):
    if not condition: raise Unresolved(message)

def one(values,reason):
    distinct=list(dict.fromkeys(values))
    require(len(distinct)==1,reason)
    return distinct[0]

def norm(s):
    return mt.clean(s).strip()

@dataclass
class Table:
    columns: list
    rows: list
    prose: str

def table_from(q):
    rows=[]; prose=[]
    for line in q.splitlines():
        if line.count("|")>=2:
            cells=[norm(x) for x in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?",x) for x in cells): rows.append(cells)
        elif line.strip(): prose.append(line)
    require(len(rows)>=3 and len(rows[0])>=2,"need one labelled data table")
    require(all(len(r)==len(rows[0]) for r in rows),"inconsistent table widths")
    return Table(rows[0],rows[1:]," ".join(prose))

def column(tab,pattern):
    return one([i for i,h in enumerate(tab.columns) if re.search(pattern,h,re.I)],"missing or ambiguous column: "+pattern)

def scalar(tab,idx,i,dimension,default=None):
    head,cell=tab.columns[idx],tab.rows[i][idx]
    m=re.fullmatch("("+mt.NUM+r")\s*(.*)",cell)
    require(m,"non-scalar table cell")
    val,inline=m.groups()
    if inline: inline=givens.at_start(inline) or "unknown"
    allowed=mt.OPTIONS[dimension].split()
    units=[u for u in allowed if u in givens.PATTERNS and re.search(r"(?<![A-Za-z])(?:"+givens.PATTERNS[u]+r")(?![A-Za-z0-9/])",head,re.I)]
    # Atomic units or length units that are substrings of compound density are not accepted here.
    header=one(units,"ambiguous header unit") if units else None
    require(not (header and inline and header!=inline),"conflicting cell/header units")
    unit=inline or header or default
    require(unit in allowed,"missing or incompatible column unit")
    return f"{val} {unit}"

def volume_fractions(tab,materials):
    columns=[i for i,h in enumerate(tab.columns) if re.search(r"volume (?:fraction|percent)|vol\.?\s*%",h,re.I)]
    if columns:
        idx=one(columns,"multiple phase-fraction columns")
        return [scalar(tab,idx,i,"strain","one") for i in range(2)]
    prose=norm(tab.prose).lower()
    # Match the order explicitly given in "fractions of A and B are x and y, respectively".
    mat=r"("+ "|".join(re.escape(m.lower()) for m in materials)+r")"
    delim=r"\s*(?:,?\s+and|,)\s*"
    num="("+mt.NUM+r")\s*(%|percent)?"
    pairs=re.findall(r"\bvolume (fractions|percentages) (?:of|for)\s+"+mat+delim+mat+r"\s+(?:are|were|of|=)\s*"+num+delim+num,prose,re.I)
    require(len(pairs)==1,"volume fractions need explicit phase labels or a fraction column")
    kind,a,b,x,ux,y,uy=pairs[0]
    require(a!=b and set([a,b])==set(m.lower() for m in materials),"phase labels disagree")
    unit="percent" if ux or uy or kind=="percentages" else "one"
    mapping={a:x+" "+unit,b:y+" "+unit}
    return [mapping[m.lower()] for m in materials]

def composite(q,u):
    t=table_from(q);require(len(t.rows)==2,"requires two material rows")
    msg=norm(t.prose).lower()
    require(re.search(r"(?:specific )?(?:stiffness|modulus)",msg),"unknown requested mixture property")
    require(not re.search(r"strength|unknown loading|mass fractions?|weight fractions?|poisson|porosity",msg),"unsupported composition or property")
    rule=[]
    if re.search(r"\bvoigt\b|upper (?:limit|bound)|iso[- ]?strain|longitudinal modulus",msg): rule.append("voigt")
    if re.search(r"\breuss\b|lower (?:limit|bound)|iso[- ]?stress",msg): rule.append("reuss")
    rule=one(rule,"loading model not unique")
    labels=[i for i,h in enumerate(t.columns) if not h or re.fullmatch("material|phase|constituent|component",h,re.I)]
    col=one(labels,"need a material-name column"); materials=[r[col] for r in t.rows]
    require(all(materials) and len(set(m.lower() for m in materials))==2,"need two distinct materials")
    ei=column(t,r"young|elastic modulus|modulus of elasticity|young.s modulus|^E\b")
    moduli=[scalar(t,ei,i,"stress") for i in range(2)]
    fracs=volume_fractions(t,materials)
    vals=[float(mt.Q(float(x.split()[0]),mt.UNIT.get(x.split()[1],x.split()[1])).to("").magnitude) for x in fracs]
    require(all(0<v<1 for v in vals) and math.isclose(sum(vals),1,abs_tol=1e-6),"volume fractions do not close")
    target="specific" if re.search(r"specific (?:elastic )?(?:stiffness|modulus)",msg) else "modulus"
    densities=["-","-"]
    if target=="specific":
        ri=column(t,r"specific gravit|relative densit|\bdensity\b")
        relative=bool(re.search(r"specific gravit|relative densit",t.columns[ri],re.I))
        densities=[scalar(t,ri,i,"density","one" if relative else None) for i in range(2)]
    fields=dict(E1=moduli[0],E2=moduli[1],f1=fracs[0],rho1=densities[0],rho2=densities[1],rule=rule,target=target)
    trace=[dict(phase=m,E=E,volume_fraction=f,density=d) for m,E,f,d in zip(materials,moduli,fracs,densities)]
    return fields,dict(headers=t.columns,material_rows=trace,model=rule)

CONV=r"\b(true|logarithmic|Cauchy|nominal|engineering)\s+(?:(?:axial|tensile|plastic|flow)\s+)*(stresses|stress|strains|strain|elongation)\b"

def conventions(text):
    return [("true" if c.lower() in ("true","logarithmic","cauchy") else "engineering",
             "stress" if t.startswith("stress") else "strain") for c,t in re.findall(CONV,text.lower(),re.I)]

def role(text,kind):
    return one([c for c,t in conventions(text) if t==kind],f"missing or ambiguous {kind} convention")

def flow(q,u):
    t=table_from(q);require(len(t.rows)==2,"requires two flow points")
    msg=norm(t.prose)
    require(re.search(r"plastic|hardening|hollomon|prior to necking|before necking",msg,re.I),"uniform plastic regime not established")
    require(not re.search(r"after necking|post[- ]neck|beyond necking|compressive|elastic correction|yield offset",msg,re.I),"unsupported loading regime")
    s=column(t,r"\bstress(?:es)?\b");e=column(t,r"\bstrain(?:s)?\b|\belongation\b")
    ask=re.split(r"\b(?:find|calculate|compute|predict|estimate|determine|what|report)\b",msg,flags=re.I)
    require(len(ask)>=2,"requested quantity not explicit")
    target=next((part for part in reversed(ask[1:]) if conventions(part)),"")
    sw,ew=role(t.columns[s],"stress"),role(t.columns[e],"strain")
    out_s,out_e=role(target,"stress"),role(target,"strain")
    expr=CONV+r"\s*(?:of|=|is|:|reaches|at)?\s*("+mt.NUM+r")\s*(%|percent)?"
    amounts=[v+" "+("percent" if unit else "one") for _,k,v,unit in re.findall(expr,target,re.I) if not k.lower().startswith("stress")]
    if not amounts:
        expr=r"("+mt.NUM+r")\s*(%|percent)?\s*"+CONV
        amounts=[v+" "+("percent" if unit else "one") for v,unit,_,k in re.findall(expr,target,re.I) if not k.lower().startswith("stress")]
    target_strain=one(amounts,"cannot uniquely bind requested strain value")
    fields=dict(s1=scalar(t,s,0,"stress"),s2=scalar(t,s,1,"stress"),e1=scalar(t,e,0,"strain","one"),e2=scalar(t,e,1,"strain","one"),target_strain=target_strain,
                input_stress=sw,input_strain=ew,output_stress=out_s,target_strain_type=out_e)
    return fields,dict(headers=t.columns,target_clause=target,flow_model="two-point Hollomon, homogeneous plastic flow; neglect elastic strain")

def lattice(q,u):
    text=norm(q)
    require(not re.search(r"ratio|surface density|planar density|interstitial|vacanc|alloy|pore radius|ionic radius",text,re.I),"unsupported geometry")
    name=r"(?:silicon|germanium|diamond|copper|nickel|silver|gold|alumini?um|iron|metal|atom|Si|Ge|Al|Cu|Fe|Ag|Au|Ni)"
    nouns={"radius":r"(?:atomic|hard[- ]sphere|contact|metallic|covalent) radius|\bradius",
           "edge":r"lattice (?:constant|parameter)|cell (?:edge|parameter)|edge (?:length )?of (?:a |the )?(?:\w+ )?(?:unit )?cell"}
    roles={}
    for kind,descriptor in nouns.items():
        symbol="r" if kind=="radius" else "a"
        pattern=(r"(?:(?:"+descriptor+r")(?:\s+(?:of|for)\s+(?:an? |the )?"+name+r")?(?:\s*[,(:]?\s*\b"+symbol+r"\b\s*\)?)?|"
                 r"(?<!\w)"+symbol+r"(?:[_ ]+"+name+r")?\s*(?==))\s*(?:=|:|is|of|measures)?\s*("+mt.NUM+r")\s*([A-Za-zμµ]+)")
        vals=[]
        for v,unit in re.findall(pattern,text,re.I):
            actual=givens.at_start(unit)
            if actual in mt.OPTIONS["length"].split(): vals.append(v+" "+actual)
        if vals: roles[kind]=one(vals,"conflicting sizes")
    require(bool(roles),"no labelled atomic radius or lattice parameter")
    struct=mt.crystal(text)
    if len(roles)==2:
        sizes={k:mt.Q(float(v.split()[0]),v.split()[1]).to("m").magnitude for k,v in roles.items()}
        multiple=dict(sc=2,bcc=4/math.sqrt(3),fcc=math.sqrt(8),diamond=8/math.sqrt(3))[struct]
        require(math.isclose(sizes["edge"],multiple*sizes["radius"],rel_tol=.02),"inconsistent contact radius and lattice edge")
    kind="edge" if "edge" in roles else "radius"
    return dict(size=roles[kind],size_kind=kind),dict(roles=roles,crystal=struct,chosen_size=kind,convention="occupied row through the origin")

SPINEL_NI="https://doi.org/10.1103/PhysRevB.96.094427"
def spinel(q,u):
    text=norm(q).lower(); condensed=re.sub(r"[\s_\\{}$]","",text)
    require(re.search(r"\bmoment\b",text),"target is not a magnetic moment")
    require(not re.search(r"measur|nanoparticle|thin film|partial|disorder|deficien|doped|canting|canted|noncollinear|paramagnet|effective moment|normal spinel|room temperature|at\s+\d|experimental|orbital contribution|antiphase|primitive|tetragonal",text),"sample-specific state or unsupported cell setting")
    groups={"manganese":("mn",7),"cobalt":("co",9),"nickel":("ni",10),"copper":("cu",11)}
    ion=one([element for element,(symbol,_) in groups.items() if element+"ferrite" in condensed or symbol+"fe2o4" in condensed],"requires a single known stoichiometric ferrite")
    if ion not in ("nickel","copper"):
        require(re.search(r"\binverse\b",text) and re.search(r"high[- ]spin",text) and re.search(r"spin[- ]only",text),"needs explicit high-spin, spin-only inverse model")
    basis=[]
    if re.search(r"\b(?:unit cell|conventional cell)\b",text): basis.append("cell")
    if re.search(r"\bformula unit\b",text): basis.append("formula")
    basis=one(basis,"basis unspecified or contradictory")
    require(mt.Q(1,mt.convert_unit(u)).is_compatible_with("bohr_magneton"),"output must be a magnetic moment")
    d_electrons=groups[ion][1]-2
    unpaired=min(d_electrons,10-d_electrons)
    return dict(basis=basis),dict(material=ion+" ferrite",basis=basis,d_electrons_M2=d_electrons,unpaired_M2=unpaired,
        formula_units=8 if basis=="cell" else 1,assumption="Ideal stoichiometric collinear spin-only ordered moment. Fe3+ moments on opposite sublattices cancel. A generic unit cell means the conventional spinel cell.",sources=[mt.SPINEL,SPINEL_NI,mt.ELEMENTS])

BINDERS={"composite":composite,"linear_count":lattice,"flow":flow,"spinel":spinel}
def check(question,unit):
    text=norm(question)
    names=set(mt.retrieve(question))
    if re.search(r"\b(?:Cu|Ni|Co|Mn)\s*Fe\s*_?\s*2\s*O\s*_?\s*4\b",text,re.I): names.add("spinel")
    attempts=[]; successes=[]
    for name,binder in BINDERS.items():
        if name not in names: continue
        try:
            fields,bound=binder(question,unit)
            if name=="spinel":
                target=mt.convert_unit(unit)
                moment=mt.Q(bound["unpaired_M2"]*bound["formula_units"],"bohr_magneton").to(target)
                calc=dict(value=float(moment.magnitude),unit=target,trace=bound)
            else:
                calc=mt.solve(name,question,unit,fields)
            successes.append((name,calc))
            attempts.append(dict(tool=name,status="calculated",fields=fields,binding=bound,calculation=calc))
        except (ValueError,ArithmeticError,mt.verifier.PintError) as ex:
            attempts.append(dict(tool=name,status="abstained",reason=str(ex)))
    if len(successes)==1:
        tool,calc=successes[0]
        return dict(status="calculated",tool=tool,calculation=calc,attempts=attempts)
    return dict(status="ambiguous" if successes else "abstained" if attempts else "outside_coverage",attempts=attempts)

