# # train_tokenizer.py

import os
from typing import List, Optional
import random

from my_tokenizer import MyBPETok

# === Configuration ===
DOMAIN_FILE = "data/domain_1_train.txt"
DEV_FILE: Optional[str] = None
OUTPUT_DIR = "domain1_final"
VOCAB_SIZE = 3000

# the various sample sizes you asked for:
SAMPLE_SIZES = [30000]

# fix the global seed here:
SEED = 42


def read_text_file(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def train_and_save(
        train_lines: List[str],
        dev_lines: Optional[List[str]],
        vocab_size: int,
        sample_size: int,
        output_dir: str
):
    # ensure per‐size sample_file so they don't overwrite one another
    os.makedirs(output_dir, exist_ok=True)
    sample_file = os.path.join(output_dir, f"sampled_{sample_size}.txt")

    # reset RNG for reproducible sampling at this sample_size
    random.seed(SEED)

    tok = MyBPETok(
        vocab_size=vocab_size,
        sample_size=sample_size,
        sample_file=sample_file,
        seed=SEED
    )

    # # 3) instantiate Domain2 tokenizer
    # tok = Domain2BPETok(
    #     vocab_size=vocab_size,
    #     sample_size=None,
    #     sample_file=sample_file,
    #     seed=seed
    # )

    # (optionally) store it on the object
    tok.trained_seed = SEED

    print(f"\n>>> Training tokenizer (sample_size={sample_size}) …")
    tok.train(train_lines)
    print(f"    → learned {len(tok.merges)} merges")

    os.makedirs(output_dir, exist_ok=True)

    # a) pickle
    pkl_path = os.path.join(output_dir, "bpe_tok_improved.pkl")
    tok.save(pkl_path)
    print(f"    pickled → {pkl_path}")

    # b) vocab.txt
    with open(os.path.join(output_dir, "vocab.txt"), 'w', encoding='utf-8') as vf:
        for tok_str, tid in sorted(tok.token_to_id.items(), key=lambda x: x[1]):
            vf.write(f"{tok_str}\t{tid}\n")
    print(f"    wrote vocab.txt")

    # c) merges.txt
    with open(os.path.join(output_dir, "merges.txt"), 'w', encoding='utf-8') as mf:
        for a, b in tok.merges:
            mf.write(f"{a} {b}\n")
    print(f"    wrote merges.txt")


def main():
    # also seed once here, in case you add more randomness later
    random.seed(SEED)

    # read once
    train_lines = read_text_file(DOMAIN_FILE)
    dev_lines = read_text_file(DEV_FILE) if DEV_FILE else None

    for size in SAMPLE_SIZES:
        subdir = os.path.join(OUTPUT_DIR, f"domain_1_tokenizer")
        train_and_save(
            train_lines=train_lines,
            dev_lines=dev_lines,
            vocab_size=VOCAB_SIZE,
            sample_size=size,
            output_dir=subdir
        )


if __name__ == "__main__":
    main()