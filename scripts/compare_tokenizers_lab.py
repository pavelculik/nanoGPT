"""Train an educational byte BPE on train text, compare on held-out validation text."""
import argparse
import csv
import sys
import time
from pathlib import Path

import tiktoken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minbpe', type=Path, default=Path('../minbpe'))
    parser.add_argument('--data', type=Path, default=Path('data/domain_lab'))
    parser.add_argument('--vocab-size', type=int, default=512)
    args = parser.parse_args()
    if not 257 <= args.vocab_size <= 4096:
        parser.error('For this small educational run, use vocabulary size 257–4096.')
    sys.path.insert(0, str(args.minbpe.resolve()))
    from minbpe import BasicTokenizer

    def read_split(name, limit):
        pieces = []
        remaining = limit
        for path in sorted((args.data / name).glob('*.txt')):
            part = path.read_text(encoding='utf-8')[:remaining]
            pieces.append(part)
            remaining -= len(part)
            if remaining <= 0:
                break
        text = '\n\n'.join(pieces)
        if not text:
            parser.error(f'No text in {args.data / name}; run preparation first.')
        return text

    train = read_split('train_text', 100000)
    validation = read_split('val_text', 20000)
    custom = BasicTokenizer()
    custom.train(train, vocab_size=args.vocab_size)
    actual_size = len(custom.vocab)
    output = Path('data/tokenizer_lab')
    output.mkdir(parents=True, exist_ok=True)
    custom.save(str(output / 'custom'))
    gpt2 = tiktoken.get_encoding('gpt2')
    cases = [('validation', validation), ('diacritics', 'Žltý kôň kráča cez lúku.'), ('numbers', 'Cena je 12,50 €. Rok 2026.'), ('emoji', 'Ahoj 👋🙂'), ('spaces', 'A  B\nC\tD'), ('code', 'def f(x):\n    return x + 1')]
    rows = []
    for label, text in cases:
        for name, encode, decode, vocab_size in [
            ('custom_byte_bpe', custom.encode, custom.decode, actual_size),
            ('gpt2_tiktoken', gpt2.encode_ordinary, gpt2.decode, gpt2.n_vocab),
        ]:
            encode(text)  # warm-up before timing
            begin = time.perf_counter()
            for _ in range(3):
                ids = encode(text)
            elapsed = (time.perf_counter() - begin) / 3
            assert decode(ids) == text, f'Round trip failed: {name}, {label}'
            rows.append({'sample': label, 'tokenizer': name, 'vocabulary': vocab_size, 'tokens': len(ids), 'utf8_bytes_per_token': round(len(text.encode('utf-8')) / len(ids), 3), 'seconds': round(elapsed, 6)})
    report = Path('reports/tokenizers_lab.csv')
    report.parent.mkdir(exist_ok=True)
    with report.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Train: {len(train):,} characters; validation: {len(validation):,} characters')
    print(f'Actual custom vocabulary: {actual_size}; GPT-2 vocabulary: {gpt2.n_vocab}')
    print(f'Saved {report}. Different vocabulary sizes and implementations are not a controlled comparison.')


if __name__ == '__main__':
    main()
