"""Verified local dataset snapshots and deterministic, disjoint experiment splits."""
import hashlib
import json
from pathlib import Path
import random


def read_snapshot(directory, label, cfg):
    directory = Path(directory)
    meta = json.loads((directory / 'manifest.json').read_text())['datasets'][label]
    if meta['revision'] != cfg[f'{label}_revision']:
        raise ValueError(f'{label} snapshot revision does not match config')
    blob = (directory / meta['file']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta['sha256']:
        raise ValueError(f'{label} snapshot checksum mismatch')
    rows = [json.loads(line) for line in blob.decode().splitlines()]
    if len(rows) != meta['rows']:
        raise ValueError(f'{label} snapshot row count mismatch')
    return rows, meta


def partition(pool, validation_size, test_size, seed, previously_seen=()):
    """Reserve unseen test rows first, then validation, then all remaining rows."""
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    seen = set(previously_seen)
    test = [r for r in shuffled if r['id'] not in seen][:test_size]
    if len(test) != test_size:
        raise ValueError('Not enough previously unseen rows for the test split')
    test_ids = {r['id'] for r in test}
    rest = [r for r in shuffled if r['id'] not in test_ids]
    if len(rest) <= validation_size:
        raise ValueError('Not enough rows for validation and training')
    return {'test': test, 'validation': rest[:validation_size], 'candidates': rest[validation_size:]}


def prepare(directory, tokenizer, cfg):
    pools, audit = {}, {'sources': {}, 'excluded': {}}
    for label in ('harmful', 'harmless'):
        source, meta = read_snapshot(directory, label, cfg)
        audit['sources'][label] = meta
        unique, excluded = {}, []
        prompts = [row['text'].strip() for row in source]
        # Count the actual chat template, including the assistant prefix.
        strings = [tokenizer.apply_chat_template([
            {'role': 'system', 'content': cfg['system_prompt']},
            {'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            for prompt in prompts]
        lengths = [len(ids) for ids in tokenizer(strings, add_special_tokens=False)['input_ids']]
        for index, (prompt, length) in enumerate(zip(prompts, lengths)):
            key = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            reason = 'empty' if not prompt else 'duplicate' if key in unique else 'too_long' if length > cfg['max_prompt_tokens'] else None
            if reason:
                excluded.append({'source_row': index, 'id': key, 'reason': reason, 'tokens': length})
            else:
                unique[key] = {'id': key, 'prompt': prompt}
        pools[label] = list(unique.values())
        audit['excluded'][label] = excluded
    overlap = {r['id'] for r in pools['harmful']} & {r['id'] for r in pools['harmless']}
    for label in pools:
        audit['excluded'][label].extend({'id': key, 'reason': 'cross_label_duplicate'} for key in sorted(overlap))
        pools[label] = [r for r in pools[label] if r['id'] not in overlap]
    # Full harmful coverage is a hard requirement; never silently truncate it.
    if len(pools['harmful']) != audit['sources']['harmful']['rows']:
        raise ValueError('Full harmful coverage failed; inspect duplicates or raise max_prompt_tokens')
    random.Random(cfg['seed']).shuffle(pools['harmless'])
    matched_size = len(pools['harmful'])
    if len(pools['harmless']) < matched_size:
        raise ValueError('Not enough harmless prompts for a matched comparison')
    audit['harmless_eligible'] = len(pools['harmless'])
    audit['harmless_not_sampled'] = len(pools['harmless']) - matched_size
    pools['harmless'] = pools['harmless'][:matched_size]
    exposure = json.loads((Path(directory) / 'pilot-exposure.json').read_text())['ids']
    record = {'train': {}, 'validation': {}, 'test': {}, 'candidates': {}, 'all': {}, 'filter': {}}
    for label, pool in pools.items():
        parts = partition(pool, cfg['validation_size'], cfg['test_size'], cfg['seed'], exposure[label])
        for split, rows in parts.items():
            record[split][label] = rows
        record['all'][label] = pool
    audit['split_counts'] = {split: {label: len(rows) for label, rows in record[split].items()}
                             for split in ('candidates', 'validation', 'test', 'all')}
    audit['test_excludes_pilot_exposure'] = True
    return record, audit
