import re
import unicodedata
from pint.errors import PintError
from quantity_calculator import as_unit

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

UNITS = {"kg/m^2s":"kg/(m**2*s)", "J/m^2*s":"J/(m**2*s)", "W/(m·K)":"W/(m*K)",
    r"$(\min )^{-1}$":"1/min", "$"+r"{}^\circ$":"degree", "BM":"bohr_magneton", r"\mu_B":"bohr_magneton",
    "defects/m³":"1/m**3","defects/m^3":"1/m**3","atoms/nm^2":"1/nm**2","atoms/unit cell":"dimensionless",
    r"$\frac{\text{atoms}}{\mathrm{nm}}$":"1/nm", "%":"percent", "wt%":"percent"}

def unit_for(raw):
    if not raw:
        return ""
    if raw.strip().isnumeric():
        return None
    name=UNITS.get(raw,normalize(raw))
    try:
        as_unit(name)
    except (ValueError,TypeError,PintError):
        return None
    return name

