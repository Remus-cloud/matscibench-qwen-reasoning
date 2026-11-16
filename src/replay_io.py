import hashlib
import json
import os
import tempfile
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_bytes())

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False,encoding="utf8") as fp:
        json.dump(obj,fp,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(fp.name,path)

