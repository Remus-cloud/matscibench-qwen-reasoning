"""Build the standalone classroom HTML from the reviewed result archive."""
import json
from pathlib import Path
from replay import ROOT, read_rows, audit

def main():
    rows=read_rows();audit(rows)
    payload=json.dumps(rows,ensure_ascii=False,separators=(",",":")).replace("&",r"\u0026").replace("<",r"\u003c").replace(">",r"\u003e")
    template=(ROOT/"src/classroom.html").read_text(encoding="utf8")
    assert template.count("__REVIEWED_RESULTS__")==1
    dest=ROOT/"demo/index.html"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(template.replace("__REVIEWED_RESULTS__",payload),encoding="utf8",newline="\n")
    print(f"{len(rows)} questions, five groups: {dest}")
if __name__=="__main__":main()

