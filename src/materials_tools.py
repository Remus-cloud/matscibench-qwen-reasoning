"""Question-only definition retrieval, grounded field extraction and numerical checks."""
import math
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

import physical_units as verifier
from quantity_calculator import UNITS

MECH = "https://web.mit.edu/course/3/3.11/www/modules/ss.pdf"
COMPOSITE = "https://eng.libretexts.org/Bookshelves/Materials_Science/TLP_Library_II/6%3A_Mechanics_of_Fibre-Reinforced_Composites/6.2%3A_Stiffness_of_Long_Fiber_Composites"
CRYSTAL = "https://lampz.tugraz.at/~hadley/ss1/crystalstructure/structures/diamond/diamond.php"
DEFECT = "https://www.tf.uni-kiel.de/matwis/amat/def_en/kap_2/backbone/r2_1_3.html"
SPINEL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5768695/"
CONSTANTS = "https://physics.nist.gov/cuu/Constants/"
ELEMENTS = "https://periodic-table.rsc.org/"
DIPOLE = "https://122.physics.ucdavis.edu/course/cosmology/sites/default/files/files/Ferro%20Electricity/Kittel_c16_Dielectrics_Ferroelectrics.pdf"

@dataclass(frozen=True)
class Card:
    keywords: str
    definition: str
    fields: str
    source: tuple

CARDS = {
    "flow": Card(r"plastic strain|strain.harden|flow stress|true strain|true stress|plastic deformation|plastically deform|hollomon|nominal stress|engineering stress",
        "Uniform uniaxial plastic flow. Infer the Hollomon power law only for two flow points on one material and the requested stress at another plastic strain. Use true stress and true plastic strain. Nominal inputs require pre-necking volume conservation. Engineering plastic elongation approximates total elongation only in plastic-dominated strains. Exclude explicit necking, nonzero yield-offset laws, significant elastic correction or unknown data.",
        "s1:stress e1:strain s2:stress e2:strain target_strain:strain input_stress:convention input_strain:convention output_stress:convention target_strain_type:convention", (MECH,)),
    "area": Card(r"area|cross.section",
        "Under homogeneous volume-conserving uniaxial elongation, A_initial=A_final*(1+engineering elongation); invert for final area. Only for an explicit known area, known elongation and requested area.",
        "given_area:area elongation:strain wanted_area:initial_final", (MECH,)),
    "mean_stress": Card(r"mean normal|hydrostatic|mean stress",
        "Mean tensile-positive normal stress is the trace of the Cauchy stress tensor divided by three. Pressure reverses the sign. Shear does not contribute.",
        "x:stress y:stress z:stress sign:stress_sign", (MECH,)),
    "linear_count": Card(r"linear densit|line density|atoms per (unit )?length",
        "Number of atomic centres per length along the occupied row through a lattice point (conventional cell origin). Use the cubic basis and the stated direct-space direction; count a periodic half-open repeat. In hard-sphere geometry nearest-neighbor distance is 2r. Si and Ge normally have diamond-cubic structure. Use a lattice parameter or an atomic contact radius.",
        "size:length size_kind:edge_radius", (CRYSTAL,)),
    "polarization": Card(r"polari[sz]ation|dipole moment per unit volume",
        "Magnitude of electronic dipole moment per volume for a rigid displacement of the electron cloud. Cell density times the displaced charge per atom times displacement. Z electrons for the entire cloud of a neutral atom. For an explicitly specified subset use its count. Exclude orientational, ionic, spontaneous or susceptibility questions.",
        "edge:length displacement:length electrons:count charge:charge", (DIPOLE,CRYSTAL,ELEMENTS,CONSTANTS)),
    "schottky": Card(r"schottky",
        "Dilute equilibrium in a stoichiometric 1:1 ionic solid with equal sublattice populations and unit degeneracy prefactor. Pair density = formula-unit density times exp(-pair formation energy/(2 k_B T)). Total number of vacant sites is twice the number of pairs. Number of vacancies of one species equals the pair count. Distinguish eV per pair and energy per mole of pairs.",
        "energy:energy temperature:temperature density:density molar_mass:massmol quantity:defect_target", (DEFECT,ELEMENTS,CONSTANTS)),
    "composite": Card(r"composite|specific (?:modulus|stiffness)|rule of mixtures",
        "Binary iso-strain composite or upper modulus bound: volume-fraction weighted moduli. Iso-stress or Reuss lower bound: reciprocal mixture. Specific modulus divides by mass density. If constituent densities are specific gravities and an output pressure unit is stipulated, use E/relative density in that convention. Require the loading rule and association between phases. Exclude strength, force, stress or modulus of an individual phase.",
        "E1:stress f1:strain E2:stress rho1:density rho2:density rule:mix target:mix_target", (COMPOSITE,)),
    "spinel": Card(r"ferrit|spinel",
        "Ideal stoichiometric collinear spin-only saturation of an inverse ferrite MFe2O4: Fe3+ occurs on both antiparallel sublattices and its moments cancel. Net is the ordered moment of M2+ per formula unit. Conventional spinel cell has 8 formula units. The saturation value gS differs from the paramagnetic effective moment. Bulk CuFe2O4 typically inverse; real samples may be partially inverted. Exclude measured moments, thermal corrections, or unspecified site occupancy for materials without an inverse assignment.",
        "basis:moment_basis", (SPINEL,)),
    "conversion": Card(r"\bconvert\b|express .* (?:in|as) |unit conversion",
        "Convert one given scalar physical quantity to the requested compatible unit. Apply only when the requested result is solely that change of units. Exclude requests that first require a physical model.",
        "given:convert", (CONSTANTS,)),
}

OPTIONS = {
    "stress":"Pa kPa MPa GPa psi ksi", "strain":"one percent", "length":"m cm mm um nm pm angstrom",
    "count":"one", "charge":"C", "energy":"eV J kJ/mol J/mol", "temperature":"K degC", "density":"one g/cm3 kg/m3",
    "massmol":"g/mol kg/mol", "area":"m2 cm2 mm2", "convention":"true engineering",
    "initial_final":"initial final", "stress_sign":"tensile pressure", "edge_radius":"edge radius",
    "defect_target":"pairs_per_volume all_vacancies_per_volume fraction_one_sublattice",
    "mix":"voigt reuss", "mix_target":"modulus specific", "moment_basis":"cell formula",
}
OPTIONS["convert"]="Pa kPa MPa GPa psi ksi m cm mm um nm pm angstrom m2 cm2 mm2 K degC J eV kg/m3 g/cm3"
ENUM = {"convention","initial_final","stress_sign","edge_radius","defect_target","mix","mix_target","moment_basis"}
NUM = r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
Q = UNITS.Quantity
UNIT = {"one":"","m2":"m^2","cm2":"cm^2","mm2":"mm^2","kg/m3":"kg/m^3","g/cm3":"g/cm^3",
        "microC/m2":"microcoulomb/m^2","C/m2":"C/m^2","C/m²":"C/m^2","atoms/nm":"1/nm",
        "m2/s2":"m^2/s^2","defects/m³":"1/m^3","pairs/m3":"1/m^3","BM":"bohr_magneton"}

def clean(s):
    s=unicodedata.normalize("NFKC",s).replace("−","-").replace("–","-")
    s=re.sub(r"\\(?:mathrm|text|operatorname|mathbf)\s*\{([^{}]*)\}",r"\1",s)
    s=s.replace(r"\AA","angstrom").replace("Å"," angstrom ").replace(r"\circ","°")
    s=re.sub(r"\\(?:times|cdot)|×","*",s)
    s=re.sub(r"(\d+(?:\.\d*)?)\s*\*\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",r"\1e\2",s)
    s=re.sub(r"(?<![\w.])10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",r"1e\1",s)
    s=re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))","",s)
    s=re.sub(r"\\(?:left|right)|\\[,;!]|[\${}]"," ",s)
    return " ".join(s.split())

def retrieve(question):
    q=clean(question).lower()
    return [name for name,c in CARDS.items() if re.search(c.keywords,q,re.I)]

def numeric_literals(question):
    return re.findall(NUM,clean(question))

def crystal(q):
    if re.search(r"diamond|silicon\b|germanium\b",q,re.I): return "diamond"
    if re.search(r"body.center|body.centr|\bbcc\b",q,re.I): return "bcc"
    if re.search(r"face.center|face.centr|\bfcc\b",q,re.I): return "fcc"
    if re.search(r"simple cubic|primitive cubic",q,re.I): return "sc"
    if re.search(r"\b(copper|alumin[iu]+m|silver|gold|nickel)\b",q,re.I): return "fcc"
    raise ValueError("unspecified cubic structure")

FCC = [(0,0,0),(0,2,2),(2,0,2),(2,2,0)]
BASIS = {"sc":FCC[:1],"bcc":[(0,0,0),(2,2,2)],"fcc":FCC,
         "diamond":FCC+[(x+1,y+1,z+1) for x,y,z in FCC]}
ELEMENT = {"copper":29,"aluminium":13,"aluminum":13,"nickel":28,"silver":47,"gold":79,"silicon":14,"germanium":32}
MASS = {"Na":22.990,"K":39.098,"Li":6.94,"Cs":132.905,"Ag":107.868,"Mg":24.305,"Ca":40.078,
        "Cl":35.45,"F":18.998,"Br":79.904,"I":126.904,"O":15.999,"S":32.06}
SALT = {"sodium":"Na","potassium":"K","lithium":"Li","cesium":"Cs","caesium":"Cs","silver":"Ag","magnesium":"Mg","calcium":"Ca",
        "chloride":"Cl","fluoride":"F","bromide":"Br","iodide":"I","oxide":"O","sulfide":"S"}

def salt_mass(q):
    for m in re.finditer(r"\b([A-Z][a-z]?)([A-Z][a-z]?)\b",q):
        a,b=m.groups()
        if a in MASS and b in MASS: return MASS[a]+MASS[b]
    words=re.findall(r"[a-z]+",q.lower())
    for a,b in zip(words,words[1:]):
        if a in SALT and b in SALT and b.endswith(("ide","oxide")): return MASS[SALT[a]]+MASS[SALT[b]]
    raise ValueError("cannot identify 1:1 formula")

def convert_unit(raw):
    if raw in UNIT: return UNIT[raw]
    if not raw: raise ValueError("unspecified output unit")
    unit=verifier.unit_for(raw)
    if not unit: raise ValueError("unsupported requested unit")
    return unit

def count_on_row(q,structure):
    text=re.sub(r"\\bar\s*(\d)",r"-\1",clean(q))
    matches=re.findall(r"[\[<]\s*(-?[0-9])\s*[, ]*\s*(-?[0-9])\s*[, ]*\s*(-?[0-9])\s*[\]>]",text)
    if len(set(matches))!=1: raise ValueError("need one stated direct-space direction")
    h=tuple(map(int,matches[0])); g=math.gcd(*h)
    if g==0: raise ValueError("zero direction")
    h=tuple(i//g for i in h)
    # For a primitive integer direction all occupied sites at t*h have 4*t integral.
    times=[Fraction(i,4) for i in range(4) if tuple((i*j)%4 for j in h) in BASIS[structure]]
    return len(times),math.sqrt(sum(i*i for i in h)),str(times)

def solve(name,q,raw,fields):
    """fields only contain copied numerical literals and a restricted unit or category."""
    trace={}
    def num(k,dest,optional=False):
        if fields.get(k) in (None,"-"):
            if optional: return None
            raise ValueError("missing "+k)
        v,u=fields[k].split()
        if v not in numeric_literals(q): raise ValueError("ungrounded literal "+k)
        u=UNIT.get(u,u)
        value=float(Q(float(v),u).to(dest).magnitude)
        if not math.isfinite(value): raise ValueError("nonfinite "+k)
        trace[k]=[fields[k],value,dest]
        return value
    def enum(k):
        v=fields[k]
        if v=="-": raise ValueError("missing "+k)
        return v
    if name=="flow":
        a,b=num("s1","Pa"),num("s2","Pa")
        e,f,t=(num(k,"") for k in ("e1","e2","target_strain"))
        if min(a,b,e,f,t)<=0: raise ValueError("requires positive tensile plastic points")
        if re.search(r"(after|beyond|post)[- ]?neck|localized neck|compressi",q,re.I): raise ValueError("unsupported localization/loading")
        ew,sw=enum("input_strain"),enum("input_stress")
        x,y=(math.log1p(e),math.log1p(f)) if ew=="engineering" else (e,f)
        if sw=="engineering": a,b=a*math.exp(x),b*math.exp(y)
        t=math.log1p(t) if enum("target_strain_type")=="engineering" else t
        n=math.log(b/a)/math.log(y/x)
        if not 0<n<1: raise ValueError("outside power-law hardening range")
        s=a*(t/x)**n
        if enum("output_stress")=="engineering": s/=math.exp(t)
        trace.update(n=n,true_target=t,method="two-point Hollomon fit, uniform plastic tension")
        result=Q(s,"Pa")
    elif name=="area":
        a,e=num("given_area","m^2"),num("elongation","")
        if a<=0 or e<=-1: raise ValueError("invalid geometry")
        result=Q(a*(1+e) if enum("wanted_area")=="initial" else a/(1+e),"m^2")
    elif name=="mean_stress":
        result=Q(sum(num(k,"Pa") for k in ("x","y","z"))/3*(1 if enum("sign")=="tensile" else -1),"Pa")
    elif name=="linear_count":
        struct=crystal(q); a=num("size","m")
        if enum("size_kind")=="radius": a*=dict(sc=2,bcc=4/math.sqrt(3),fcc=math.sqrt(8),diamond=8/math.sqrt(3))[struct]
        if a<=0: raise ValueError("invalid length")
        count,length,points=count_on_row(q,struct)
        trace.update(structure=struct,sites=count,repeat_in_cell_edges=length,fractional_positions=points,cell_edge_m=a)
        result=Q(count/(a*length),"1/m")
    elif name=="polarization":
        struct=crystal(q); a,d=num("edge","m"),abs(num("displacement","m"))
        z=num("electrons","",True)
        if z is None:
            z=next((v for word,v in ELEMENT.items() if re.search(r"\b"+word+r"\b",q,re.I)),None)
        if not z or z!=int(z) or a<=0: raise ValueError("unknown displaced charge")
        charge=num("charge","C",True) or 1.602176634e-19
        result=Q(len(BASIS[struct])*z*charge*d/a**3,"C/m^2")
        trace.update(structure=struct,atoms_per_cell=len(BASIS[struct]),electrons_per_atom=z,charge_C=charge)
    elif name=="schottky":
        if re.search(r"\b(?:CaF2|MgCl2|ZrO2|UO2|fluorite)\b|nonstoichiomet",q,re.I): raise ValueError("requires 1:1 stoichiometry")
        inferred_mass=salt_mass(q)  # also enforces a supported 1:1 compound
        T=num("temperature","K"); energy=fields["energy"]
        if energy=="-": raise ValueError("missing formation energy")
        E=num("energy","J/mol")/6.02214076e23 if "/mol" in energy else num("energy","J")
        if min(T,E)<=0: raise ValueError("nonpositive energy/temperature")
        frac=math.exp(-E/(2*1.380649e-23*T)); wanted=enum("quantity")
        trace.update(absolute_T=T,vacancy_fraction=frac)
        if wanted=="fraction_one_sublattice": result=Q(frac,"")
        else:
            rho=num("density","kg/m^3"); M=num("molar_mass","kg/mol",True) or inferred_mass/1000
            if min(rho,M)<=0: raise ValueError("invalid formula-unit density")
            sites=rho/M*6.02214076e23; trace.update(formula_units_per_m3=sites,molar_mass_kg_per_mol=M)
            result=Q(sites*frac*(2 if wanted=="all_vacancies_per_volume" else 1),"1/m^3")
    elif name=="composite":
        a,b,f=num("E1","Pa"),num("E2","Pa"),num("f1","")
        if min(a,b)<=0 or not 0<f<1: raise ValueError("invalid two-phase mixture")
        E=a*f+b*(1-f) if enum("rule")=="voigt" else 1/(f/a+(1-f)/b)
        result=Q(E,"Pa");trace["mixture_modulus_Pa"]=E
        if enum("target")=="specific":
            sg="specific gravit" in q.lower() or "relative densit" in q.lower()
            as_pressure=Q(1,convert_unit(raw)).check("[pressure]")
            if as_pressure and not sg: raise ValueError("pressure convention needs relative density")
            unit="" if sg else "kg/m^3"
            rho=f*num("rho1",unit)+(1-f)*num("rho2",unit)
            if rho<=0: raise ValueError("invalid density")
            trace["mixture_density"]=rho
            result=result/rho if as_pressure else result/Q(rho*(1000 if sg else 1),"kg/m^3")
    elif name=="spinel":
        s=re.sub(r"[\s_\\{}$]","",q.lower())
        if re.search(r"partially|canting|noncollinear|effective moment|paramagnetic|normal spinel",q,re.I) and not re.search(r"inverse spinel",q,re.I):
            raise ValueError("unsupported magnetic state")
        ions={"copper":("cu",1),"nickel":("ni",2),"cobalt":("co",3),"manganese":("mn",5),"iron":("fe",4)}
        ion=next(((sym,u) for word,(sym,u) in ions.items() if sym+"fe2o4" in s or word+"ferrite" in s),None)
        if ion is None or (ion[0] not in {"cu","ni"} and "inverse" not in s): raise ValueError("inverse occupancy not justified")
        multiplicity=8 if enum("basis")=="cell" else 1
        trace.update(ion=ion[0],saturated_muB_per_formula=ion[1],formula_units=multiplicity,model="ideal inverse spin-only")
        result=Q(ion[1]*multiplicity,"bohr_magneton")
    elif name=="conversion":
        if fields["given"]=="-": raise ValueError("missing source unit")
        canonical=UNIT.get(fields["given"].split()[1],fields["given"].split()[1])
        result=Q(num("given",canonical),canonical)
    else: raise ValueError("unknown relation")
    dest=convert_unit(raw)
    value=float(result.to(dest).magnitude)
    if not math.isfinite(value): raise ValueError("nonfinite result")
    return dict(value=value,unit=dest,trace=trace)

