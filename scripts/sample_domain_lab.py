"""Generate text from your own GPT-2-tokenized domain checkpoint."""
import argparse
import sys
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trust-local-checkpoint', action='store_true')
    parser.add_argument('--prompt', default='Dnes')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--tokens', type=int, default=150)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.trust_local_checkpoint:
        parser.error('Only load a checkpoint you created yourself. Confirm with --trust-local-checkpoint.')
    if args.temperature <= 0 or args.tokens < 1 or not args.prompt:
        parser.error('Use positive temperature/tokens and a nonempty prompt.')
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if not checkpoint['config']['dataset'].startswith('domain_lab'):
        parser.error('This sampler is only for the GPT-2-tokenized domain_lab dataset.')
    model = GPT(GPTConfig(**checkpoint['model_args']))
    weights = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(weights)
    model.to(args.device).eval()
    encoder = tiktoken.get_encoding('gpt2')
    torch.manual_seed(args.seed)
    ids = encoder.encode_ordinary(args.prompt)
    sequence = torch.tensor([ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        for _ in range(args.tokens):
            context = sequence[:, -model.config.block_size:]
            logits, _ = model(context)
            # nanoGPT pads vocab to 50304; only 50257 GPT-2 IDs are decodable.
            logits = logits[:, -1, :encoder.n_vocab] / args.temperature
            cutoff = torch.topk(logits, min(50, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            sequence = torch.cat((sequence, next_id), dim=1)
    print(encoder.decode(sequence[0].tolist()))


if __name__ == '__main__':
    main()
