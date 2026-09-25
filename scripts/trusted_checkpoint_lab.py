"""Explicit compatibility patch, ONLY for checkpoints created by you in this lab."""
from pathlib import Path

old = 'torch.load(ckpt_path, map_location=device)'
new = 'torch.load(ckpt_path, map_location=device, weights_only=False)'
paths = [Path('train.py'), Path('sample.py')]
changes = []
for path in paths:
    text = path.read_text(encoding='utf-8')
    if new in text:
        print(f'{path}: already adjusted')
    elif text.count(old) == 1:
        changes.append((path, text.replace(old, new)))
    else:
        raise SystemExit(f'{path}: unexpected source. No files changed. Ask your mentor to inspect.')
for path, text in changes:
    path.write_text(text, encoding='utf-8')
    print(f'{path}: adjusted for your own trusted checkpoints')
