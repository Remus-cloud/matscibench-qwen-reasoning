"""Extend A's local runner with auditable, resumable base/V3 CoT evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from compare import ROOT, records
from protocol import messages


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def append(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def finish_reason(tokens, eos_ids, budget):
    if tokens and tokens[-1] in eos_ids:
        return 'stop'
    return 'length' if len(tokens) >= budget else 'unknown'


def prepare_output(path, config, qids, resume):
    manifest = path.with_suffix('.manifest.json')
    if path.exists() or manifest.exists():
        if not resume:
            raise ValueError('Output exists; use --resume only with the identical configuration.')
        if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise ValueError('Resume configuration mismatch or missing manifest')
        saved = records(path) if path.exists() else []
        if [r['qid'] for r in saved] != qids[:len(saved)]:
            raise ValueError('Duplicate, out-of-order or unexpected saved qid')
        if any(r.get('config_id') != fingerprint(config) for r in saved):
            raise ValueError('Saved result configuration mismatch')
        if any(not isinstance(r.get('response'), str) or
               r.get('finish_reason') not in ('stop', 'length', 'unknown') for r in saved):
            raise ValueError('Invalid saved response or finish reason')
        return len(saved)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path.touch(exist_ok=False)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke', action='store_true', help='Fixed five varied development questions')
    parser.add_argument('--max-new-tokens', type=int, default=16384)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--materials', type=Path,
                        default=ROOT / 'provenance/b_eval/provenance/materials_verified.json',
                        help='Verified model and dataset inventory')
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('max-new-tokens must be positive')
    data_path = ROOT / 'data' / f'{args.split}.json'
    rows = json.loads(data_path.read_bytes())['rows']
    if args.smoke:
        if args.split != 'dev' or args.limit is not None:
            parser.error('--smoke requires dev and no --limit')
        rows = [rows[i] for i in (0, 2, 9, 15, 19)]
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            parser.error('Invalid --limit')
        rows = rows[:args.limit]
    materials_path = args.materials
    materials = json.loads(materials_path.read_text(encoding='utf-8'))
    if materials['status'] != 'verified' or sha(data_path) != materials['data'][args.split][1]:
        raise ValueError('Unverified materials or changed dataset')
    base_stats = {}
    for item in materials['base_files']:
        path = args.model / item['path']
        stat = path.stat()
        if stat.st_size != item['bytes']:
            raise ValueError(f'Changed model file: {path}')
        base_stats[item['path']] = [stat.st_size, stat.st_mtime_ns]
        if not item['path'].endswith('.safetensors') and sha(path) != item['sha256']:
            raise ValueError(f'Changed model configuration: {path}')
    adapter = None
    if args.adapter:
        adapter = {'path': str(args.adapter.resolve()),
                   'weights_sha256': sha(args.adapter / 'adapter_model.safetensors'),
                   'config_sha256': sha(args.adapter / 'adapter_config.json')}
        if adapter['weights_sha256'] != materials['adapter_sha256']:
            raise ValueError('Adapter differs from verified V3')

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA and BF16 required')
    set_seed(6020)
    config = {
        'arm': 'lora_cot' if adapter else 'base_cot', 'split': args.split,
        'qids': [r['qid'] for r in rows], 'data_sha256': sha(data_path),
        'protocol_sha256': sha(ROOT / 'src/protocol.py'), 'runner_sha256': sha(Path(__file__)),
        'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'materials_sha256': sha(materials_path), 'base_revision': materials['base_revision'],
        'base_path': str(args.model.resolve()), 'base_file_stats': base_stats, 'adapter': adapter,
        'seed': 6020, 'do_sample': False, 'max_new_tokens': args.max_new_tokens,
        'enable_thinking': False, 'quantization': 'NF4 double quantization, BF16 compute',
        'attention': 'sdpa', 'batch_size': 1, 'warmup_tokens': 64,
        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
        'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
        'cuda': torch.version.cuda, 'tf32': torch.backends.cuda.matmul.allow_tf32,
        'optional_kernels': {k: installed_version(k) for k in
                             ('flash-linear-attention', 'fla-core', 'causal-conv1d', 'triton')},
        'packages': {k: importlib.metadata.version(k) for k in
                     ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate', 'safetensors')},
    }
    config['materials_upstream_commit'] = materials['upstream_commit']
    start_index = prepare_output(args.output, config, config['qids'], args.resume)
    if start_index == len(rows):
        print('All requested questions already saved; no regeneration.', flush=True)
        return
    config_id = fingerprint(config)
    events = args.output.with_suffix('.events.jsonl')
    session = datetime.now(timezone.utc).isoformat()
    session_start = perf_counter()
    append(events, {'event': 'session_start', 'session': session, 'config_id': config_id,
                    'resumed_after': start_index})
    qid = None
    try:
        load_start = perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16))
        matched = 0
        if args.adapter:
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
            actual = get_peft_model_state_dict(model)
            expected = load_file(str(args.adapter / 'adapter_model.safetensors'), device='cpu')
            if actual.keys() != expected.keys():
                raise ValueError('Loaded adapter tensor keys do not match V3')
            for key, value in expected.items():
                observed = actual[key].detach().cpu()
                if not torch.equal(observed, value.to(observed.dtype)):
                    raise ValueError(f'Loaded adapter tensor differs: {key}')
            matched = len(expected)
            del actual, expected
            if model.active_adapters != ['default']:
                raise ValueError('Unexpected active adapter')
        elif any(hasattr(m, 'lora_A') for m in model.modules()):
            raise ValueError('Base arm unexpectedly contains LoRA modules')
        model.eval()
        torch.cuda.synchronize()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        runtime = {'event': 'model_loaded', 'session': session, 'config_id': config_id,
                   'load_seconds': perf_counter()-load_start, 'adapter_tensors_matched': matched,
                   'active_adapters': model.active_adapters if args.adapter else [],
                   'model_class': type(model).__name__, 'eos_ids': eos_ids,
                   'generation_config': model.generation_config.to_dict(),
                   'chat_template_sha256': fingerprint(tokenizer.chat_template),
                   'model_memory_bytes': model.get_memory_footprint()}
        append(events, runtime)
        print('MODEL VERIFIED', config['arm'], 'matched LoRA tensors:', matched, flush=True)

        def generate(row, budget):
            prompt = tokenizer.apply_chat_template(messages(row, 'cot'), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to('cuda:0')
            torch.cuda.synchronize()
            start = perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=budget)
            tokens = output[0, inputs.input_ids.shape[1]:].tolist()
            elapsed = perf_counter()-start
            return {'qid': row['qid'], 'response': tokenizer.decode(tokens, skip_special_tokens=True),
                    'finish_reason': finish_reason(tokens, eos_ids, budget),
                    'generated_tokens': len(tokens), 'generated_token_ids': tokens,
                    'input_tokens': inputs.input_ids.shape[1], 'prompt': prompt,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'generation_seconds': elapsed, 'session': session, 'config_id': config_id}

        warmup = json.loads((ROOT/'data/dev.json').read_bytes())['rows'][0]
        append(events, {'event': 'warmup', **generate(warmup, 64)})
        for row in rows[start_index:]:
            qid = row['qid']
            result = generate(row, args.max_new_tokens)
            append(args.output, result)
            print(f'{qid}: {result["finish_reason"]}, {result["generated_tokens"]} tokens, '
                  f'{result["generation_seconds"]:.1f}s', flush=True)
        append(events, {'event': 'session_end', 'session': session, 'status': 'completed',
                        'elapsed_seconds': perf_counter()-session_start, 'config_id': config_id})
        print('RUN COMPLETE', args.output, flush=True)
    except BaseException as error:
        append(events, {'event': 'session_end', 'session': session, 'status': 'failed',
                        'qid': qid, 'error_type': type(error).__name__, 'error': str(error),
                        'elapsed_seconds': perf_counter()-session_start, 'config_id': config_id})
        raise


if __name__ == '__main__':
    main()
