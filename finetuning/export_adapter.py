"""Export Qwen3.5-4B's training adapters and verify all GGUF factors against their PEFT originals."""
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base",required=True,type=Path)
parser.add_argument("--llama-cpp",required=True,type=Path)
parser.add_argument("--checkpoint",required=True,type=Path)
parser.add_argument("--out",required=True,type=Path)
args=parser.parse_args()
LLAMA,MODEL=args.llama_cpp.resolve(),args.base.resolve()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py")]
import gguf
import numpy as np
import torch
from safetensors.numpy import load_file
from conversion.qwen import _LinearAttentionVReorderBase as Reorder

original_reorder = Reorder._reorder_v_heads


def reorder(weight, dim, key_heads, ratio, width):
    if not hasattr(weight, "get_lora_A_B"):
        return original_reorder(weight, dim, key_heads, ratio, width)
    a, b = weight.get_lora_A_B()
    # The GGUF head permutation on BA acts only on B for rows, only on A for columns.
    indices = torch.arange(key_heads * ratio * width).reshape(key_heads, ratio, width).transpose(0, 1).reshape(-1)
    if dim == 0:
        return type(weight)(a, b.index_select(0, indices))
    if dim == 1:
        return type(weight)(a.index_select(1, indices), b)
    raise ValueError(dim)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(path, hf, config):
    reader = gguf.GGUFReader(path)
    tensors = {t.name: t.data for t in reader.tensors}
    assert reader.get_field("adapter.lora.alpha").contents() == config["lora_alpha"]
    model = json.loads((MODEL / "config.json").read_bytes())["text_config"]
    nh, nv, d = model["linear_num_key_heads"], model["linear_num_value_heads"], model["linear_value_head_dim"]
    # Derive the source V row index for each destination head, independently of torch's reshape.
    vi = [k * (nv // nh) * d + repeat * d + channel for repeat in range(nv // nh)
          for k in range(nh) for channel in range(d)]
    qk = 2 * nh * model["linear_key_head_dim"]
    modules = {"q_proj":"attn_q", "k_proj":"attn_k", "v_proj":"attn_v", "o_proj":"attn_output",
               "in_proj_qkv":"attn_qkv", "out_proj":"ssm_out", "down_proj":"ffn_down",
               "up_proj":"ffn_up", "gate_proj":"ffn_gate"}
    verified = set()
    for name, array in load_file(hf).items():
        layer, _, projection, factor, _ = name.split(".layers.")[1].split(".")
        field = f"blk.{layer}.{modules[projection]}.weight.{factor.lower()}"
        if projection == "out_proj" and factor == "lora_A":
            array = array[:, vi]
        if projection == "in_proj_qkv" and factor == "lora_B":
            array = np.concatenate((array[:qk], array[qk:][vi]))
        if not np.array_equal(tensors[field], array):
            raise ValueError(f"Adapter factor mismatch: {field}")
        verified.add(field)
    if verified != set(tensors):
        raise ValueError("Unexpected GGUF factors")
    return len(verified)



if __name__=="__main__":
    Reorder._reorder_v_heads=staticmethod(reorder)
    f=args.checkpoint.resolve()
    out=args.out.resolve()
    out.parent.mkdir(exist_ok=True,parents=True)
    temporary=out.with_suffix(".partial.gguf")
    if not out.is_file():
        sys.argv=[str(LLAMA/"convert_lora_to_gguf.py"),"--base",str(MODEL),"--no-lazy","--outtype","f32","--outfile",str(temporary),str(f)]
        runpy.run_path(sys.argv[0],run_name="__main__")
        count=verify(temporary,f/"adapter_model.safetensors",json.loads((f/"adapter_config.json").read_bytes()))
        temporary.replace(out)
    else:
        count=verify(out,f/"adapter_model.safetensors",json.loads((f/"adapter_config.json").read_bytes()))
    print(f"Verified {count} adapter tensors in {out}")
