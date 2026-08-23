#!/usr/bin/env python3
import os
import random
import math
from typing import List, Tuple
from collections import Counter

# import your tokenizer implementations
from my_tokenizer import MyBPETok
from my_tokenizer_domain2 import Domain2BPETok

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
SEED = 42
OUTPUT_DIR = "trained_tokenizers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Domain-1 settings
DOM1_FILE       = "data/domain_1_train.txt"
DOM1_VOCAB      = 3000
DOM1_SAMPLE_SZ  = 30000

# Domain-2 settings
DOM2_FILE       = "data/domain_2_train.txt"
DOM2_VOCAB      = 4000
DOM2_SAMPLE_SZ  = 30000

# Domain-3 (mixed) settings
DOM3_FILE1      = DOM1_FILE
DOM3_FILE2      = DOM2_FILE
DOM3_VOCAB      = 5000
DOM3_SAMPLE_SZ  = 50000

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def read_texts(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [l.rstrip("\n") for l in f if l.strip()]

# a little “headline‐likeness” scorer for Domain3’s second half
def score_domain2(line: str) -> float:
    words = line.split()
    s = sum(2 for w in words if w.istitle())
    s += sum(1 for w in words if any(ch.isdigit() for ch in w))
    s += math.log(len(words) + 1)
    return s

# ─── TRAIN DOMAIN 1 ──────────────────────────────────────────────────────────────
def train_domain1():
    random.seed(SEED)
    lines = read_texts(DOM1_FILE)

    sample_file = os.path.join(OUTPUT_DIR, f"sample_dom1_{DOM1_SAMPLE_SZ}_seed{SEED}.txt")
    # delete old sample if any
    try: os.remove(sample_file)
    except FileNotFoundError: pass

    tok1 = MyBPETok(
        vocab_size=DOM1_VOCAB,
        sample_size=DOM1_SAMPLE_SZ,
        sample_file=sample_file,
        seed=SEED
    )
    print("⮞ Training Domain1 tokenizer …")
    tok1.train(lines)
    print(f"    → learned {len(tok1.merges)} merges")

    out_pkl = os.path.join(OUTPUT_DIR, "tokenizer_1.pkl")
    tok1.save(out_pkl)
    print("    • saved →", out_pkl)


# ─── TRAIN DOMAIN 2 ──────────────────────────────────────────────────────────────
def train_domain2():
    random.seed(SEED)
    lines = read_texts(DOM2_FILE)

    sample_file = os.path.join(OUTPUT_DIR, f"sample_dom2_{DOM2_SAMPLE_SZ}_seed{SEED}.txt")
    try: os.remove(sample_file)
    except FileNotFoundError: pass

    tok2 = Domain2BPETok(
        vocab_size=DOM2_VOCAB,
        sample_size=DOM2_SAMPLE_SZ,
        sample_file=sample_file,
        seed=SEED
    )
    print("⮞ Training Domain2 tokenizer …")
    tok2.train(lines)
    print(f"    → learned {len(tok2.merges)} merges")

    out_pkl = os.path.join(OUTPUT_DIR, "tokenizer_2.pkl")
    tok2.save(out_pkl)
    print("    • saved →", out_pkl)


# ─── TRAIN DOMAIN 3 (MIXED) ───────────────────────────────────────────────────────
def train_domain3():
    random.seed(SEED)
    dom1 = read_texts(DOM3_FILE1)
    dom2 = read_texts(DOM3_FILE2)

    # scoreers
    tok_dummy = MyBPETok(vocab_size=DOM3_VOCAB)  # just to get .score_line
    score1 = tok_dummy.score_line
    score2 = score_domain2

    N1 = int(DOM3_SAMPLE_SZ * 0.4)
    N2 = int(DOM3_SAMPLE_SZ * 0.4)
    NR = DOM3_SAMPLE_SZ - N1 - N2

    top1 = sorted(dom1, key=score1, reverse=True)[:N1]
    top2 = sorted(dom2, key=score2, reverse=True)[:N2]
    rest = [l for l in (dom1 + dom2) if l not in top1 and l not in top2]
    rand20 = random.sample(rest, NR)

    sampled = top1 + top2 + rand20
    random.shuffle(sampled)

    sample_file = os.path.join(OUTPUT_DIR, f"sample_dom3_{DOM3_SAMPLE_SZ}_seed{SEED}.txt")
    with open(sample_file, 'w', encoding='utf-8') as f:
        for ln in sampled:
            f.write(ln + "\n")
    print(f"⮞ Wrote {len(sampled)} lines to {sample_file}")

    tok3 = MyBPETok(
        vocab_size=DOM3_VOCAB,
        sample_size=DOM3_SAMPLE_SZ,
        sample_file=sample_file,
        seed=SEED
    )
    print("⮞ Training Domain3 tokenizer …")
    tok3.train(sampled)    # train(...) will read sample_file first
    print(f"    → learned {len(tok3.merges)} merges")

    out_pkl = os.path.join(OUTPUT_DIR, "tokenizer_3.pkl")
    tok3.save(out_pkl)
    print("    • saved →", out_pkl)


# ─── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_domain1()
    train_domain2()
    train_domain3()
    print("✅ All three tokenizers trained and saved in", OUTPUT_DIR)
