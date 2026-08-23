import random
import re
from typing import List
from collections import Counter
from functools import lru_cache


# make sure you import MyBPETok (with the new min_merge_freq support)
from my_tokenizer import MyBPETok

class Domain2BPETok(MyBPETok):
    """
    BPE tokenizer specialized for news/article domain (Domain 2).
    - Simplified preprocessing for headlines/articles
    - Stratified sampling by headline-likeness + random
    """

    def __init__(
        self,
        vocab_size: int = 4000,
        sample_size: int = 30000,
        sample_file: str = "sampled_domain2.txt",
        seed: int = 42,
        min_merge_freq: int = 2,
    ):
        super().__init__(
            vocab_size=vocab_size,
            sample_size=sample_size,
            sample_file=sample_file,
            seed = seed
        )
        self.seed = seed
        self.min_merge_freq = min_merge_freq
        # disable tweet-specific placeholders from base
        self.mention_pattern = None
        self.hashtag_pattern = None
        self.emoji_pattern   = None

    def preprocess(self, text: str) -> str:
        # replace URLs with a generic placeholder
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
        # collapse repeated punctuation (e.g., “!!!” → “!!”)
        text = re.sub(r'([!?.,;:\-"\'])\1{2,}', r'\1\1', text)
        return text.strip()

    def _sample_tweets(self, texts: List[str]) -> List[str]:
        random.seed(self.seed)
        # 50% random sample
        subset = random.sample(texts, k=len(texts)//2)
        lines = [self.preprocess(t) for t in subset]
        # write out so base.train() reuses it
        with open(self.sample_file, 'w', encoding='utf-8') as f:
            for ln in lines:
                f.write(ln + '\n')
        return lines
