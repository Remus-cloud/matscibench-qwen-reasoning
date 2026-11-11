"""Compare the new local base/V3 pair using the original project's grading rules."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from compare import ROOT, records
from protocol import grade
from run_27b import fingerprint, sha


def validate_pair(base, lora):
    if base['arm'] != 'base_cot' or lora['arm'] != 'lora_cot':
        raise ValueError('Expected new base_cot and lora_cot arms, in that order')
    if base['adapter'] is not None or not lora['adapter']:
        raise ValueError('Adapter assignment is invalid')
    common = lambda c: {k: v for k, v in c.items() if k not in ('arm', 'adapter')}
    if common(base) != common(lora):
        raise ValueError('The two arms have different experimental conditions')


def score_rows(rows, saved, config_id):
    gold = {r['qid']: r for r in rows}
    result = {}
    for r in saved:
        qid = r['qid']
        if qid not in gold or qid in result:
            raise ValueError('Unknown or duplicate question')
        if r.get('config_id') != config_id or not isinstance(r['response'], str):
            raise ValueError('Wrong configuration or missing raw response')
        if r['finish_reason'] not in ('stop', 'length', 'unknown'):
            raise ValueError('Invalid finish reason')
        pred, ok, category = grade(r['response'], str(gold[qid]['answer']), r['finish_reason'])
        result[qid] = {**r, 'prediction': pred, 'correct': ok, 'error_category': category}
    return result


def summarize(rows, scored):
    n = len(rows)
    tokens = [r['generated_tokens'] for r in scored.values() if r.get('generated_tokens') is not None]
    times = [r['generation_seconds'] for r in scored.values() if r.get('generation_seconds') is not None]
    unresolved = sum(r['finish_reason'] == 'unknown' for r in scored.values())
    complete = len(scored) == n and not unresolved
    correct = sum(r['correct'] for r in scored.values())
    return {'planned': n, 'answered': len(scored), 'coverage': len(scored)/n,
            'complete': complete, 'correct': correct, 'accuracy': correct/n if complete else None,
            'missing': [r['qid'] for r in rows if r['qid'] not in scored],
            'unknown_finish_count': unresolved,
            'categories': {c: sum(r['error_category'] == c for r in scored.values()) for c in
                           ('correct', 'numeric', 'format', 'truncated', 'unknown')},
            'generated_tokens': sum(tokens) if tokens else None, 'token_coverage': len(tokens),
            'mean_tokens': statistics.mean(tokens) if tokens else None,
            'generation_seconds': sum(times) if times else None, 'time_coverage': len(times),
            'mean_seconds': statistics.mean(times) if times else None,
            'median_seconds': statistics.median(times) if times else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('lora', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = (args.base, args.lora)
    configs = [json.loads(p.with_suffix('.manifest.json').read_text(encoding='utf-8')) for p in paths]
    validate_pair(*configs)
    config = configs[0]
    data = ROOT/'data'/f'{config["split"]}.json'
    if sha(data) != config['data_sha256'] or sha(ROOT/'src/protocol.py') != config['protocol_sha256']:
        raise ValueError('Dataset or grading protocol changed')
    all_rows = json.loads(data.read_bytes())['rows']
    if config['split'] == 'test' and config['qids'] != [r['qid'] for r in all_rows]:
        raise ValueError('Formal evaluation must cover the complete fixed test set')
    row_map = {r['qid']: r for r in all_rows}
    rows = [row_map[qid] for qid in config['qids']]
    scored = [score_rows(rows, records(p), fingerprint(c)) for p, c in zip(paths, configs)]
    summaries = [summarize(rows, s) for s in scored]
    runtimes = []
    for p, c, summary in zip(paths, configs, summaries):
        events = records(p.with_suffix('.events.jsonl'))
        if any(e['config_id'] != fingerprint(c) for e in events):
            raise ValueError('Runtime event configuration mismatch')
        loaded = [e for e in events if e['event'] == 'model_loaded']
        if not loaded or (c['arm'] == 'lora_cot' and any(e['adapter_tensors_matched'] == 0 for e in loaded)):
            raise ValueError('Missing actual model/adapter loading proof')
        runtimes.append(loaded[-1])
        starts = {e['session'] for e in events if e['event'] == 'session_start'}
        ends = [e for e in events if e['event'] == 'session_end']
        summary['unfinished_sessions'] = sorted(starts - {e['session'] for e in ends})
        summary['failed_sessions'] = sum(e['status'] == 'failed' for e in ends)
        summary['session_elapsed_seconds'] = sum(e['elapsed_seconds'] for e in ends)
        summary['load_seconds'] = sum(e['load_seconds'] for e in loaded)
        summary['warmup_tokens'] = sum(e['generated_tokens'] for e in events if e['event'] == 'warmup')
        summary['complete'] &= not summary['unfinished_sessions']
        if not summary['complete']:
            summary['accuracy'] = None
    for key in ('generation_config', 'chat_template_sha256', 'eos_ids'):
        if runtimes[0][key] != runtimes[1][key]:
            raise ValueError(f'Runtime conditions differ: {key}')
    paired, groups = [], {k: [] for k in ('gains', 'losses', 'both_correct', 'both_wrong')}
    for row in rows:
        qid = row['qid']
        a, b = (s.get(qid) for s in scored)
        entry = {'qid': qid, 'question': row['question'], 'unit': row['unit'], 'gold': row['answer']}
        if a is not None and b is not None:
            category = ('both_correct' if a['correct'] else 'gains') if b['correct'] else (
                        'losses' if a['correct'] else 'both_wrong')
            groups[category].append(qid)
            entry['change'] = category
        else:
            entry['change'] = 'missing'
        for arm, result in zip(('base', 'lora'), (a, b)):
            for key in ('prediction', 'correct', 'error_category', 'finish_reason',
                        'generated_tokens', 'generation_seconds', 'response'):
                entry[f'{arm}_{key}'] = result.get(key) if result else None
        paired.append(entry)
    complete = all(s['complete'] for s in summaries)
    report = {'split': config['split'], 'complete': complete, 'base_cot': summaries[0],
              'lora_cot': summaries[1], 'paired': groups,
              'accuracy_difference_percentage_points':
                  100*(summaries[1]['accuracy']-summaries[0]['accuracy']) if complete else None,
              'source_files': [str(p.resolve()) for p in paths],
              'source_sha256': [sha(p) for p in paths], 'config_ids': [fingerprint(c) for c in configs]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (args.output_dir/'paired_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps({k: v for k, v in report.items() if k != 'paired'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
