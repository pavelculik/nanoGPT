# Doménový jazykový model na báze nanoGPT

## Čo projekt robí
Trénuje malý GPT (7,23 M parametrov) na vlastnom korpuse slovenských
básnických textov a generuje z neho ukážky. Projekt vychádza z nanoGPT
od Andreja Karpathyho (https://github.com/karpathy/nanoGPT).

## Dáta
Texty pochádzajú z <zdroj, edícia, licencia>. Korpus nie je súčasťou
repozitára. Každý dokument je samostatný vyčistený UTF-8 súbor v priečinku
`data/domain_texts/`.

## Príprava
1. Vyčistené texty patria do `data/domain_texts/`.
2. `python scripts/prepare_domain_lab.py`

## Tréning
`python train.py config/train_domain_lab.py`

## Ukážka
`python scripts/sample_domain_lab.py --checkpoint out-domain-lab-01/ckpt.pt --trust-local-checkpoint --prompt "Dnes" --temperature 0.8`

## Výsledok
Val loss po 2000 iteráciách: 3.7538.
iter 7000: loss 3.6931
