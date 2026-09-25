"""Prepare document-separated GPT-2 token streams. Run from nanoGPT's root."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/domain_texts'))
    parser.add_argument('--output', type=Path, default=Path('data/domain_lab'))
    parser.add_argument('--block-size', type=int, default=128)
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error('--block-size must be positive')
    if args.output.exists():
        parser.error('Output already exists. Choose a new --output; existing data is preserved.')
    documents = []
    seen = set()
    for path in sorted(args.input.rglob('*.txt')):
        text = path.read_text(encoding='utf-8').strip()
        normalized = ' '.join(text.split())
        digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        if len(normalized) < 200 or digest in seen:
            continue
        seen.add(digest)
        documents.append({'path': str(path.relative_to(args.input)), 'sha256': digest, 'text': text})
    if len(documents) < 20:
        parser.error('Need at least 20 distinct .txt documents, each at least 200 characters. This is only a technical minimum.')
    random.Random(1337).shuffle(documents)
    holdout = max(2, len(documents) // 10)
    splits = {
        'train': documents[2 * holdout:],
        'val': documents[holdout:2 * holdout],
        'test': documents[:holdout],
    }
    encoder = tiktoken.get_encoding('gpt2')
    token_streams = {}
    for name, docs in splits.items():
        tokens = []
        for doc in docs:
            tokens.extend(encoder.encode_ordinary(doc['text']))
            tokens.append(encoder.eot_token)
        if len(tokens) <= args.block_size:
            parser.error(f'{name}: needs more than {args.block_size} tokens. Add documents.')
        if max(tokens) > np.iinfo(np.uint16).max:
            parser.error('Token IDs do not fit uint16.')
        token_streams[name] = np.asarray(tokens, dtype=np.uint16)
    args.output.mkdir(parents=True)
    manifest = {'seed': 1337, 'encoding': 'gpt2', 'vocab_size': encoder.n_vocab, 'splits': {}}
    for name, docs in splits.items():
        token_streams[name].tofile(args.output / f'{name}.bin')
        text_dir = args.output / f'{name}_text'
        text_dir.mkdir()
        for index, doc in enumerate(docs):
            (text_dir / f'{index:06d}.txt').write_text(doc['text'], encoding='utf-8')
        manifest['splits'][name] = {
            'documents': [{k: d[k] for k in ('path', 'sha256')} for d in docs],
            'tokens': len(token_streams[name]),
        }
        print(f'{name}: {len(docs)} documents, {len(token_streams[name]):,} tokens')
    # No meta.pkl: unmodified nanoGPT sample.py then uses its GPT-2 decoder.
    # nanoGPT pads its output vocabulary to 50304; see the course explanation.
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Done. Review train/val/test separation, including near-duplicates, before training.')


if __name__ == '__main__':
    main()
