"""One local text-only QLoRA epoch for the frozen materials SFT release."""
import hashlib
import json
import math
import os
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent
BASE = None
DATA = ROOT / "data"
OUT = ROOT / "runs"
sys.path.insert(0, str(ROOT.parent / "src"))
from protocol import messages

SEED = 6020
ACCUM = 8
STEPS = 134
CHECKPOINTS = (67, 134)
LR = 5e-5
RANK = 16
ALPHA = 32
TARGETS = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"


def digest(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def save_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def inputs(tokenizer, chat, max_length=512):
    def encode(ms, generation):
        text = tokenizer.apply_chat_template(ms, tokenize=False, add_generation_prompt=generation, enable_thinking=False)
        return tokenizer.encode(text, add_special_tokens=False)
    prefix = encode(chat[:-1], True)
    full = encode(chat, False)
    if full[:len(prefix)] != prefix or full[-2] != tokenizer.eos_token_id or len(full) > max_length:
        raise ValueError("Unexpected assistant boundary, EOS or sequence length")
    if tokenizer.decode(full[len(prefix):-2]) != chat[-1]["content"]:
        raise ValueError("Supervised tokens do not reconstruct the target")
    return torch.tensor(full[:-1], device="cuda")[None, :], len(prefix)


def forward_loss(model, sample):
    ids, start = sample
    # Score the first visible answer token through EOS; discard prompt logits.
    positions = torch.arange(start-1, ids.shape[1]-1, device=ids.device)
    logits = model(input_ids=ids, logits_to_keep=positions, use_cache=False).logits
    return F.cross_entropy(logits[0].float(), ids[0, start:], reduction="sum")


@torch.inference_mode()
def validate(model, samples):
    model.eval()
    loss_sum = 0.0
    n_tokens = 0
    for sample in samples:
        loss_sum += forward_loss(model, sample).item()
        n_tokens += sample[0].numel() - sample[1]
    if not math.isfinite(loss_sum):
        raise ValueError("Nonfinite validation loss")
    return loss_sum / n_tokens


def load_base():
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["visual", "lm_head"])
    return Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, quantization_config=quant, device_map={"": 0},
        attn_implementation="sdpa", local_files_only=True)


def next_lr(step):
    if step <= 2:
        return LR * step / 2
    # Keep the final update at the preceding small positive rate.
    decay_step = min(step, STEPS - 1)
    return LR * 0.5 * (1+math.cos(math.pi*(decay_step-2)/(STEPS-2)))


def train():
    OUT.mkdir(exist_ok=True, parents=True)
    lock = (OUT / "running.lock").open("a+b")
    import msvcrt
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (OUT / "summary.json").exists():
        print("Completed run already exists.", flush=True)
        return
    torch.manual_seed(SEED)
    assert torch.cuda.is_bf16_supported()
    torch.cuda.manual_seed_all(SEED)
    cfg = json.loads((ROOT / "config.json").read_bytes())
    def load_rows(name, key):
        path = DATA / name
        if digest(path) != cfg[key]: raise ValueError(f"Dataset differs: {name}")
        return [json.loads(line) for line in path.read_bytes().splitlines()]
    rows = load_rows("train.jsonl", "train_sha256")
    validation = load_rows("validation.jsonl", "val_sha256")
    assert len(rows) == len({r["id"] for r in rows}) == 1072 and len(validation) == 100
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    max_length = 1024
    train_samples = [inputs(tokenizer, rows[i]["messages"], max_length) for i in order]
    weights = [rows[i]["curriculum"]["weight"] for i in order]
    assert all(math.isfinite(w) and w > 0 for w in weights)
    # The external file has scalar keys only. Validate the gold number and EOS
    # conditional on "FINAL =" under the direct-answer prompt, never as SFT input.
    numeric_val = []
    for row in validation:
        prompt = messages(row, "direct")
        response = "FINAL = " + row["answer"]
        ids, offset = inputs(tokenizer, prompt + [{"role": "assistant", "content": response}])
        prefix = tokenizer.encode("FINAL =", add_special_tokens=False)
        if ids[0, offset:offset+len(prefix)].tolist() != prefix:
            raise ValueError("Unexpected answer prefix")
        numeric_val.append((ids, offset+len(prefix)))
    meta = dict(model="Qwen/Qwen3.5-4B", base_path=str(BASE), seed=SEED, lr=LR, rank=RANK, alpha=ALPHA,
                dropout=.05, target_modules=TARGETS, quantization="NF4 double, bf16 compute",
                template_thinking=False, optimizer="AdamW", weight_decay=0.0, warmup_steps=2,
                schedule="cosine; final step repeats penultimate rate", max_length=max_length, microbatch=1, accumulation=ACCUM,
                epochs=1, checkpoints=list(CHECKPOINTS), train_sha256=cfg["train_sha256"],
                val_sha256=cfg["val_sha256"], order=[rows[i]["id"] for i in order],
                base_index_sha256=digest(BASE / "model.safetensors.index.json"),
                gpu=torch.cuda.get_device_name(), packages={k:version(k) for k in
                    ["torch","transformers","bitsandbytes","accelerate","peft","safetensors"]})
    if meta["order"] != cfg["order"] or meta["base_index_sha256"] != cfg["base_index_sha256"]:
        raise ValueError("Unexpected training order or base weight index")
    if (OUT / "protocol.json").exists() and json.loads((OUT / "protocol.json").read_bytes()) != meta:
        raise ValueError("Saved run protocol differs")
    save_json(OUT / "protocol.json",meta)
    print("Loading local 4-bit base",flush=True)
    base = load_base()
    base.requires_grad_(False)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    config = LoraConfig(task_type="CAUSAL_LM",r=RANK,lora_alpha=ALPHA,lora_dropout=.05,
                        bias="none",target_modules=TARGETS)
    saved = [s for s in CHECKPOINTS if (OUT / f"checkpoint-{s}" / "resume.pt").is_file()]
    start = max(saved,default=0)
    model = (PeftModel.from_pretrained(base, OUT / f"checkpoint-{start}", is_trainable=True)
             if start else get_peft_model(base,config))
    trainables = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert trainables and all("lora_" in n and "language_model.layers." in n for n,p in trainables)
    assert len({n.split("language_model.layers.")[1].split(".")[0] for n,p in trainables})==32
    opt = torch.optim.AdamW([p for _,p in trainables],lr=LR,weight_decay=0.0)
    valfile = OUT / "validation.json"
    vals = json.loads(valfile.read_bytes()) if valfile.exists() else {}
    if start:
        state = torch.load(OUT / f"checkpoint-{start}/resume.pt", map_location="cpu",weights_only=True)
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state
    print("Trainable parameters",sum(p.numel() for n,p in trainables),"resume step",start,flush=True)
    def measure(step):
        key = str(step)
        if key not in vals:
            print("Checking held-out numeric token loss at step",step,flush=True)
            vals[key]=validate(model,numeric_val)
            save_json(valfile,vals)
        print("External numeric-token NLL",step,vals[key],flush=True)
    measure(start)
    for step in range(start+1, STEPS+1):
        model.train()
        t = time.perf_counter()
        group=train_samples[(step-1)*ACCUM:step*ACCUM]
        targets=sum(ids.numel()-offset for ids,offset in group)
        opt.zero_grad(set_to_none=True)
        for param_group in opt.param_groups:
            param_group["lr"]=next_lr(step)
        summed=torch.zeros((),device="cuda")
        batch_weights=weights[(step-1)*ACCUM:step*ACCUM]
        weight_mass=sum((ids.numel()-offset)*weight for (ids,offset),weight in zip(group,batch_weights))
        for sample, weight in zip(group,batch_weights):
            loss=forward_loss(model,sample)
            loss=loss*weight
            (loss/weight_mass).backward()
            summed+=loss.detach()
        norm=torch.nn.utils.clip_grad_norm_([p for n,p in trainables],1.0,error_if_nonfinite=True)
        opt.step()
        record=dict(step=step, examples=min(step*ACCUM,len(rows)), lr=next_lr(step),
                    loss=summed.item()/weight_mass, prior_clip_grad_norm=norm.item(),target_tokens=targets,
                    seconds=round(time.perf_counter()-t,3),
                    peak_gpu_GiB=round(torch.cuda.max_memory_allocated()/2**30,3))
        with (OUT / "progress.jsonl").open("a",encoding="utf8") as fp:
            fp.write(json.dumps(record)+"\n")
        print(record,flush=True)
        if step in CHECKPOINTS:
            dest=OUT / f"checkpoint-{step}"
            dest.mkdir(exist_ok=True)
            model.save_pretrained(dest,save_embedding_layers=False)
            torch.save(dict(optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state()),dest / "partial.pt")
            (dest / "partial.pt").replace(dest / "resume.pt")
            measure(step)
    model.eval()
    sample=train_samples[0]
    def signature(m):
        with torch.inference_mode():
            return m(input_ids=sample[0],logits_to_keep=1,use_cache=False).logits[0,-1].float().cpu()
    before=signature(model)
    # Reload from the actual files, keeping the exact same quantized base for comparison.
    del opt,trainables
    base=model.unload()
    del model
    torch.cuda.empty_cache()
    hashes={}
    for step in CHECKPOINTS:
        dest=OUT / f"checkpoint-{step}"
        weights=load_file(str(dest/"adapter_model.safetensors"))
        assert weights and all(torch.isfinite(v).all() for v in weights.values())
        assert any("lora_B" in n and torch.count_nonzero(v) for n,v in weights.items())
        del weights
        hashes[str(step)]=digest(dest/"adapter_model.safetensors")
    assert len(set(hashes.values())) == len(CHECKPOINTS)
    reload=PeftModel.from_pretrained(base,OUT/f"checkpoint-{STEPS}").eval()
    diff=(before-signature(reload)).abs().max().item()
    if diff > 0.05 or not math.isfinite(diff):
        raise ValueError(f"Adapter reload logits differ: {diff}")
    save_json(OUT/"summary.json",dict(completed_updates=STEPS,unique_examples=len(rows),epochs=1,
             optimizer_steps_examples=[min(step*ACCUM,len(rows)) for step in CHECKPOINTS],adapter_sha256=hashes,reload_max_logit_diff=diff,
             trainable_parameters=sum(p.numel() for n,p in reload.named_parameters() if "lora_" in n),
             numeric_validation_nll=vals))
    print("Verified both saved adapters and final reload.",flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    BASE,OUT=args.base.resolve(),args.out.resolve()
    train()
