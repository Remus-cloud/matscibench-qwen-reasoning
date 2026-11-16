import json
import os
from urllib.request import Request, ProxyHandler, build_opener
from replay_io import digest,read,write

URL=os.environ.get("LLAMA_SERVER_URL","http://127.0.0.1:8080").rstrip("/")
SAMPLER=dict(top_k=0,top_p=1.0,min_p=0.0,repeat_penalty=1.0,frequency_penalty=0.0,presence_penalty=0.0,samplers=["temperature"])

def api(route,data=None):
    body=None if data is None else json.dumps(data).encode()
    req=Request(URL+route,data=body,headers={"Content-Type":"application/json"})
    with build_opener(ProxyHandler({})).open(req,timeout=7200) as r:return json.load(r)

def turn(path,chat,cap,temperature,seed,adapters,signature,grammar=None):
    identity=digest([chat,cap,temperature,seed,adapters,signature,grammar])
    if path.is_file():
        obj=read(path)
        if obj["identity"]!=identity: raise ValueError(f"Different run already exists in {path}")
        return obj
    prompt=api("/apply-template",{"messages":chat,"chat_template_kwargs":{"enable_thinking":False}})["prompt"]
    if not prompt.endswith("<think>\n\n</think>\n\n"):raise ValueError("Qwen3.5 non-thinking chat template required")
    payload=dict(prompt=prompt,lora=adapters,n_predict=cap,temperature=temperature,seed=seed,cache_prompt=False,**SAMPLER)
    if grammar:payload["grammar"]=grammar
    raw=api("/completion",payload);settings=raw["generation_settings"]
    if raw["truncated"] or raw["stop_type"] not in ("eos","word","limit") or not raw["tokens_predicted"]:
        raise RuntimeError("Incomplete response or context overflow")
    if [a for a in settings.get("lora",[]) if a["scale"]]!=adapters or settings["n_predict"]!=cap or settings["temperature"]!=temperature and abs(settings["temperature"]-temperature)>1e-6:
        raise RuntimeError("Requested model configuration was not applied")
    result=dict(identity=identity,content=raw["content"],stop=raw["stop_type"],output_tokens=raw["tokens_predicted"])
    write(path,result);return result

def completion(path,chat,grammar,cap,protocol):
    return turn(path,chat,cap,0,6020,[],protocol,grammar)

