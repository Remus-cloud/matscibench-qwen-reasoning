"""Enumerate unit-bearing numerical givens before the language model assigns their roles."""
import re
import materials_tools as mt

# Longest unit wins. A pressure must not be read as kelvin, or mass density as a length.
PATTERNS={
    "kg/m3":r"kg\s*/\s*m\s*\^?\s*3", "g/cm3":r"g\s*/\s*cm\s*\^?\s*3",
    "kJ/mol":r"kJ\s*/\s*mol", "J/mol":r"J\s*/\s*mol", "kg/mol":r"kg\s*/\s*mol", "g/mol":r"g\s*/\s*mol",
    "mm2":r"mm\s*\^?\s*2","cm2":r"cm\s*\^?\s*2","m2":r"m\s*\^?\s*2",
    "degC":r"(?:degrees?\s+Celsius|°\s*C|degrees?\s+C)\b", "K":r"(?:K|kelvins?)",
    "percent":r"(?:vol(?:ume)?\.?\s*|weight\s*)?(?:percent(?:age)?|per cent|%)",
    "angstrom":r"(?:angstroms?|ångströms?)", "um":r"(?:[µμu]m|micro(?:meter|metre)s?)",
    "nm":r"(?:nm|nano(?:meter|metre)s?)", "pm":r"(?:pm|pico(?:meter|metre)s?)",
    "mm":r"(?:mm|milli(?:meter|metre)s?)", "cm":r"(?:cm|centi(?:meter|metre)s?)",
    "m":r"(?:m|metres?|meters?)", "C":r"(?:C|coulombs?)",
    "GPa":r"(?:GPa|gigapascals?)","MPa":r"(?:MPa|megapascals?)","kPa":r"(?:kPa|kilopascals?)","Pa":r"(?:Pa|pascals?)",
    "ksi":r"ksi","psi":r"psi","eV":r"(?:eV|electron[- ]?volts?)","J":r"(?:J|joules?)",
}
TAG=re.compile("|".join("(?P<u"+str(i)+">"+s+r")(?![A-Za-z0-9/])" for i,s in enumerate(PATTERNS.values())),re.I)
UNITS=list(PATTERNS)
LIST=re.compile(r"^(?:\s*(?:,|;|\band\b|\bor\b)\s*(?:and\s+)?"+mt.NUM+r"){1,4}\s*",re.I)

def at_start(tail):
    match=TAG.match(tail.lstrip())
    return UNITS[int(match.lastgroup[1:])] if match else None

def options(q):
    s=mt.clean(q); found={k:set() for k in mt.OPTIONS if k not in mt.ENUM}
    for m in re.finditer(mt.NUM,s):
        value=m[0]; tail=s[m.end():].lstrip()
        u=at_start(tail)
        if u is None:
            group=LIST.match(tail)
            if group: u=at_start(tail[group.end():])
        if u is None and re.search(r"(?:specific gravities|relative densities).{0,35}$",s[:m.start()],re.I):
            u="one"
        for typ,choices in mt.OPTIONS.items():
            if typ not in found: continue
            if u and u in choices.split():
                found[typ].add(value+" "+u)
            if u is None and "one" in choices.split():
                found[typ].add(value+" one")
        # Scientific tables often print the unit once in a header.
        if u is None and ("tabular" in q or "&" in q or "|" in q or "\t" in q):
            for typ in ("stress","length"):
                available={UNITS[int(n.lastgroup[1:])] for n in TAG.finditer(s)}
                for unit in available.intersection(mt.OPTIONS[typ].split()):
                    found[typ].add(value+" "+unit)
    return {k:sorted(v) for k,v in found.items()}

def bind(name,q,raw,fields):
    """Resolve deterministic source bindings, e.g. a signed tensor and definition of a count."""
    fields=fields.copy();found=options(q);msg=mt.clean(q)
    if name=="flow":
        # "true Hollomon law" names a constitutive law; descriptors attached to the
        # observed stresses and strains determine the required data transformations.
        conv=lambda label: "engineering" if label.lower() in ("nominal","engineering") else "true"
        for field,noun in (("input_stress","stress(?:es)?"),("input_strain","(?:plastic )?(?:strains?|elongations?)")):
            m=re.search(r"\b(nominal|engineering|true)\s+"+noun+r"\s+(?:of|at|values|are|were|in|=)",msg,re.I)
            if m: fields[field]=conv(m[1])
        m=re.search(r"\b(?:find|estimate|calculate|determine|predict)\s+(?:the\s+)?(nominal|engineering|true)\s+(?:flow )?stress",msg,re.I)
        if m: fields["output_stress"]=conv(m[1])
        if re.search(r"\b(?:nominal|engineering) elongations?\b",msg,re.I) and re.search(r"\bat\s+[\d.]+\s*(?:percent|%)\s+elongation",msg,re.I):
            fields["target_strain_type"]="engineering"
    if name=="mean_stress":
        # Require an identified group of exactly three normal components and their common unit.
        clause=re.search(r"(?:normal (?:stresses|components)|diagonal components)([^.;]*?)(?:\.\s|;|$)",msg,re.I)
        if clause is None: raise ValueError("three normal components not identified")
        clause=re.split(r"(?:\band\b|\bwith\b)?\s*(?:there is|shear)",clause[1],flags=re.I)[0]
        vals=re.findall(mt.NUM,clause)
        actual={s.split()[1] for s in options(clause)["stress"]}
        if len(vals)!=3 or len(actual)!=1: raise ValueError("ambiguous normal tensor entries")
        unit=next(iter(actual))
        for k,x in zip(("x","y","z"),vals): fields[k]=x+" "+unit
    if name=="polarization":
        if not found["charge"]: fields["charge"]="-"
        if not re.search(r"atomic number|number of electrons|electrons? per atom|(?<=\d) electrons|\bZ\s*=",msg,re.I):
            fields["electrons"]="-"
    if name=="schottky":
        for key,typ in (("energy","energy"),("temperature","temperature"),("density","density"),("molar_mass","massmol")):
            items=[x for x in found[typ] if not x.endswith(" one")]
            if len(items)<=1: fields[key]=items[0] if items else "-"
        if re.search(r"(fraction|percentage) of .*?(?:sites|vacan)|(?:vacancy|defect) fraction",msg,re.I):
            fields["quantity"]="fraction_one_sublattice"
        if re.search(r"(?:combined|total).{0,35}(?:vacant sites|vacancies)|vacan(?:t sites|cies) of both",msg,re.I):
            fields["quantity"]="all_vacancies_per_volume"
    if name=="composite":
        if re.search(r"specific (?:stiffness|modulus)|modulus divided by (?:relative |mass )?density",msg,re.I):
            fields["target"]="specific"
        if re.search(r"\bReuss\b",msg,re.I): fields["rule"]="reuss"
        elif re.search(r"\bVoigt\b",msg,re.I): fields["rule"]="voigt"
    if name=="composite" and re.search("specific gravit|relative densit",msg,re.I):
        for key in ("rho1","rho2"):
            if fields[key]!="-": fields[key]=fields[key].split()[0]+" one"
        props=re.search(r"\bmoduli (?:are|of)\s+("+mt.NUM+r")\s*(?:GPa|MPa|Pa|ksi)?\s*(?:and|,)\s*("+mt.NUM+r")",msg,re.I)
        rho=re.search(r"(?:specific gravities|relative densities) (?:are|of)\s+("+mt.NUM+r")\s*(?:and|,)\s*("+mt.NUM+r")",msg,re.I)
        if props and rho and fields["E1"]!="-" and fields["E2"]!="-":
            order=[float(fields[k].split()[0]) for k in ("E1","E2")]
            listed=list(map(float,props.groups()))
            if order==listed or order==listed[::-1]:
                densities=rho.groups() if order==listed else rho.groups()[::-1]
                fields["rho1"],fields["rho2"]=[x+" one" for x in densities]
    return fields

