## clean my tokenizer

### new my tokenizer
# # my_tokenizer.py

import math
import os
import re
import heapq
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Set
from tqdm import tqdm
from functools import lru_cache

from base_tokenizer import BaseTokenizer


class MyBPETok(BaseTokenizer):
    """
    Fast BPE tokenizer for tweet‐domain NER.
    - Tweet‐aware placeholders (<url>, <user>, <tag>, <emoji>)
    - One‐time stratified sampling persisted to disk
    - Initial word‐bigram collapse
    - Char‐level BPE via max‐heap + incremental updates
    - Final bigram addition
    """

    def __init__(
            self,
            vocab_size: int = 3000,
            sample_size: int = 30000,
            sample_file: str = "sampled_tweets.txt",
            seed: int = 42,  # ← new!

    ):
        super().__init__()
        # Config
        self.max_vocab_size = vocab_size
        self.sample_size = sample_size
        self.sample_file = sample_file
        self.seed = seed  # ← store it
        # Preprocessing patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@(\w{1,15})')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
        # BPE state
        self.space_token = '▁'
        self.word_symbols: List[List[str]] = []
        self.word_freqs: List[int] = []
        self.pair_freqs: Counter = Counter()
        self.pair_to_words: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.symbol_set: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        # For final bigram pass
        self.reserved_bigrams = 100
        # Corpus cache
        self._initialized = False

        # wrap the per-word BPE logic in an LRU‐cache
        # up to 50 k distinct words
        self._encode_word = lru_cache(maxsize=50000)(self._encode_word_impl)

    def __getstate__(self):
        # drop the cache itself from the pickled state
        state = self.__dict__.copy()
        state.pop('_encode_word', None)
        return state

    def __setstate__(self, state):
        # restore everything else…
        self.__dict__.update(state)
        # …and then re-bind your LRU-cached word‐encoder
        self._encode_word = lru_cache(maxsize=50000)(self._encode_word_impl)

    def _encode_word_impl(self, w: str) -> Tuple[int, ...]:
        """
        Turn one raw word (no entity markers) into a tuple of token‐IDs.
        This is exactly the body of your old inner BPE loop.
        """
        merge_map = self.merge_map
        tok2id = self.token_to_id
        unk = tok2id["<UNK>"]

        # 1) build initial symbol list
        syms = list(w)
        syms[-1] += "⟂"

        # 2) apply merges
        made = True
        while made and len(syms) > 1:
            made = False
            new = []
            i = 0
            L = len(syms)
            while i < L:
                if i + 1 < L and (pair := (syms[i], syms[i + 1])) in merge_map:
                    new.append(merge_map[pair])
                    i += 2
                    made = True
                else:
                    new.append(syms[i])
                    i += 1
            syms = new

        # 3) map to IDs
        return tuple(tok2id.get(s, unk) for s in syms)

    def _register_merge(self, pair: Tuple[str, str], idx: int):
        """
        Add `pair`→merged token exactly once, updating symbol_set,
        merges list, and merge_ranks.
        """
        if pair not in self.merge_ranks:
            self.merges.append(pair)
            self.merge_ranks[pair] = idx
            merged = pair[0] + pair[1]
            self.symbol_set.add(merged)

    def preprocess(self, text: str) -> str:
        # wrap url with its content
        t = self.url_pattern.sub(lambda m: f"<URL url=\"{m.group(0)}\" >", text)
        t = self.mention_pattern.sub(lambda m: f"<USER name=\"{m.group(1)}\" >", t)
        t = self.hashtag_pattern.sub(lambda m: f"<TAG tag=\"{m.group(1).lower()}\" >", t)
        t = self.emoji_pattern.sub(lambda m: f"<EMOJI uni=\"{ord(m.group(0)):x}\" >", t)
        return re.sub(r'(.)\1{2,}', r'\1\1', t)

    """
    ANOTHER ENHANCED VERSION
    WE SHOULD SEE HOW GOOD IT IS
    """

    def score_line(self, line: str) -> float:
        """
        Heuristic score for how “entity‐rich” a line is.
        """
        words = line.split()
        s = 0.0

        # 1) word‐level entity cues
        for w in words:
            # hashtagged entities
            if w.startswith("#"):
                s += 5

            # handles / mentions
            elif w.startswith("@"):
                s += 4

            # mixed initials+digits, e.g. BB11, A7X, AG-HMC40P
            elif re.fullmatch(r"[A-Z]+[\dA-Z\-]+", w):
                s += 4

            # ALL‐CAPS of length ≥2
            elif w.isupper() and len(w) > 1:
                s += 3

            # Title‐case tokens (“Bangkok”, “NewDelhi”)
            elif w.istitle():
                s += 3

            # CamelCase inside (e.g. BattlestarGalactica)
            elif re.search(r"[a-z][A-Z]", w):
                s += 2

            # hyphenated names (“Velasquez-Manoff”)
            elif "-" in w and re.search(r"[A-Za-z]-[A-Za-z]", w):
                s += 2

            # quoted or starred (“*Boston*”, "'Rickey'")
            elif (w.startswith("'") and w.endswith("'")) or (w.startswith("*") and w.endswith("*")):
                s += 2

            # HTML entities (&amp;, &quot;)
            elif re.fullmatch(r"&\w+;", w):
                s += 1

            # any digits (dates, IDs)
            elif any(ch.isdigit() for ch in w):
                s += 1

        # 2) placeholder bonus (<USER>, <URL>, etc.)
        placeholders = sum(
            1 for w in words
            if w.startswith("<USER") or w.startswith("<URL") or
            w.startswith("<TAG") or w.startswith("<EMOJI")
        )
        s += 5 * placeholders

        # 3) multi‐word Title‐Case spans (“New York”, “House of Cards”)
        if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line):
            s += 5

        # 4) reward medium‐length lines (not one‐word, not huge)
        s += math.log(len(words) + 1)

        return s

    def _sample_tweets(self, texts: List[str]) -> List[str]:
        """
        50/50 stratified sampling:
        - 50% top-scoring
        - 50% random from the rest
        Always re-sample fresh (no reuse).
        """
        # 1) delete any old sample file
        # if os.path.exists(self.sample_file):
        #     os.remove(self.sample_file)
        # if os.path.exists(self.sample_file):
        #     with open(self.sample_file, encoding="utf-8") as f:
        #         return [l.strip() for l in f]
        random.seed(self.seed)

        # 2) preprocess all lines
        proc = [self.preprocess(t) for t in texts]
        N_top = int(self.sample_size * 0.5)
        N_rand = self.sample_size - N_top

        # A) placeholder-rich (~30% of top picks)
        ph = [l for l in proc if any(tag in l for tag in ("<USER", "<URL", "<TAG", "<EMOJI"))]
        ph_sorted = sorted(ph, key=self.score_line, reverse=True)[: int(N_top * 0.3)]

        # B) special-pattern (~20% of top picks)
        def is_special(line: str) -> bool:
            return bool(
                re.search(r"#\w+", line) or
                re.search(r"[A-Z]+[\dA-Z\-]{2,}", line) or
                re.search(r"[a-z][A-Z]", line) or
                re.search(r"[A-Za-z]-[A-Za-z]", line) or
                re.search(r"&\w+;", line) or
                # balanced quote or star: ('...') or (*...*)
                re.search(r"(^|\s)['*].+?['*]($|\s)", line)
            )

        sp = [l for l in proc if is_special(l)]
        sp_sorted = sorted(sp, key=self.score_line, reverse=True)[: int(N_top * 0.2)]

        # C) fill the rest of top picks by overall score
        scored = sorted(proc, key=self.score_line, reverse=True)
        top_combined: List[str] = []
        for batch in (ph_sorted, sp_sorted, scored):
            for ln in batch:
                if ln not in top_combined:
                    top_combined.append(ln)
                    if len(top_combined) >= N_top:
                        break
            if len(top_combined) >= N_top:
                break

        # D) random half from the remaining (order preserved)
        remaining = [l for l in proc if l not in top_combined]
        random_part = random.sample(remaining, min(N_rand, len(remaining)))

        sampled = top_combined + random_part

        # 3) write out the sample
        with open(self.sample_file, 'w', encoding='utf-8') as f:
            for ln in sampled:
                f.write(ln + '\n')

        return sampled

    def _is_probable_entity(self, w: str) -> bool:
        # 5+ simple heuristics for “looks like an entity”
        if w.startswith("#") or w.startswith("@"):
            return True
        if re.fullmatch(r"[A-Z]+[\dA-Z\-]+", w):  # BB11, A7X, AG-HMC40P
            return True
        if w.isupper() and len(w) > 1:  # ALL-CAPS
            return True
        if w.istitle():  # Title-case
            return True
        if re.search(r"[a-z][A-Z]", w):  # camelCase inside
            return True
        if "-" in w and re.search(r"[A-Za-z]-[A-Za-z]", w):
            return True
        return False

    def _initialize(self, texts: List[str]) -> None:
        # 1) sample & preprocess tweets
        lines = self._sample_tweets(texts)

        # 2) pick our top 5 “real” word-bigrams (freq ≥2) and stash them
        word_bigram_ctr = Counter()
        for ln in lines:
            ws = ln.split()
            for w1, w2 in zip(ws, ws[1:]):
                word_bigram_ctr[(w1, w2)] += 1

        # keep only those with frequency ≥2
        self.top_word_bigrams = [
            pair for pair, freq in word_bigram_ctr.most_common(5)
            if freq >= 2
        ]
        print("⮞ collapsing these word-bigrams up-front:", self.top_word_bigrams)

        # 3) now build a char-sequence corpus, collapsing those bigrams first
        corpus = Counter()
        for ln in lines:
            ws = ln.split()
            idx = 0
            seq = []
            while idx < len(ws):
                # if the next two words match one of our top bigrams, collapse them
                for (b1, b2) in self.top_word_bigrams:
                    if idx < len(ws) - 1 and (ws[idx], ws[idx + 1]) == (b1, b2):
                        seq.append(ws[idx] + " " + ws[idx + 1])  # note the space will be part of the token
                        idx += 2
                        break
                else:
                    # no two-word collapse, fall back to char-level
                    word = ws[idx]
                    # prepend your space_token, append end-of-word marker
                    chars = list(self.space_token + word)
                    chars[-1] += '⟂'
                    seq.extend(chars)
                    idx += 1

            corpus[tuple(seq)] += 1

        # 4) unpack into word_symbols & word_freqs, initialize pair_freqs
        for seq, freq in corpus.items():
            self.word_symbols.append(list(seq))
            self.word_freqs.append(freq)

        for wi, (syms, freq) in enumerate(zip(self.word_symbols, self.word_freqs)):
            # add every symbol to the symbol set
            for s in syms:
                self.symbol_set.add(s)
            # count adjacent-symbol pairs
            for a, b in zip(syms, syms[1:]):
                self.pair_freqs[(a, b)] += freq
                self.pair_to_words[(a, b)].add(wi)

        self._initialized = True

    def _learn_merges(self) -> None:
        # build heap of pair frequencies
        heap = [(-cnt, pair) for pair, cnt in self.pair_freqs.items()]
        heapq.heapify(heap)

        # reset merges
        self.merges = []
        self.merge_ranks = {}
        merge_idx = 0

        # merge until we've used up our BPE budget
        limit = self.max_vocab_size - self.reserved_bigrams
        while heap and len(self.symbol_set) < limit:
            negcnt, pair = heapq.heappop(heap)
            cnt = -negcnt

            # — skip any merge touching a placeholder token —
            if any(tok.startswith("<URL") or tok.startswith("<USER") or
                   tok.startswith("<TAG") or tok.startswith("<EMOJI")
                   for tok in pair):  # ←
                continue  # ←

            # — and skip anything that starts with '<' (other tags, etc.) —
            if (pair[0].startswith("<")) or (pair[1].startswith("<")):  # ←
                continue  # ←

            # skip stale entries or too-rare pairs
            if self.pair_freqs.get(pair, 0) != cnt or cnt < getattr(self, "min_merge_freq", 2):
                continue

            # register this merge exactly once
            self._register_merge(pair, merge_idx)
            merge_idx += 1

            # update all words containing that pair
            affected = list(self.pair_to_words.pop(pair, []))
            del self.pair_freqs[pair]
            for wi in affected:
                syms = self.word_symbols[wi]
                freq = self.word_freqs[wi]
                i = 0
                while i < len(syms) - 1:
                    if (syms[i], syms[i + 1]) == pair:
                        # decrement neighbors
                        if i > 0:
                            self._update_pair((syms[i - 1], syms[i]), -freq, heap, wi)
                        if i + 2 < len(syms):
                            self._update_pair((syms[i + 1], syms[i + 2]), -freq, heap, wi)
                        # perform merge
                        merged = pair[0] + pair[1]
                        syms[i:i + 2] = [merged]
                        # increment new neighbors
                        if i > 0:
                            self._update_pair((syms[i - 1], syms[i]), freq, heap, wi)
                        if i + 1 < len(syms):
                            self._update_pair((syms[i], syms[i + 1]), freq, heap, wi)
                        i += 1
                    else:
                        i += 1
        # no duplicate filtering or re-ranking needed—_register_merge prevents duplicates

    def _update_pair(self, pair, delta, heap, wi):
        self.pair_freqs[pair] = self.pair_freqs.get(pair, 0) + delta
        if self.pair_freqs[pair] <= 0:
            self.pair_freqs.pop(pair, None)
            self.pair_to_words.pop(pair, None)
        else:
            self.pair_to_words[pair].add(wi)
        heapq.heappush(heap, (-self.pair_freqs.get(pair, 0), pair))

    def _add_reserved_bigrams(self):
        # pick top reserved_bigrams bigrams from the final pair_freqs
        bigr = Counter()
        for syms, freq in zip(self.word_symbols, self.word_freqs):
            for a, b in zip(syms, syms[1:]):
                bigr[(a, b)] += freq

        idx = len(self.merges)
        for pair, _ in bigr.most_common(self.reserved_bigrams):
            # this will silently skip any pair already merged
            self._register_merge(pair, idx)
            idx += 1

    def train(self, texts: List[str]) -> None:
        if not self._initialized:
            self._initialize(texts)
        self._learn_merges()
        self._add_reserved_bigrams()

        # make sure our entity‐tags are in the vocab
        self.symbol_set.update({"<ENT_START>", "<ENT_END>"})

        # rebuild token_to_id / id_to_token
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>"}
        idx = 2
        for sym in sorted(self.symbol_set):
            if sym in ("<PAD>", "<UNK>"): continue
            self.token_to_id[sym] = idx
            self.id_to_token[idx] = sym
            idx += 1

        if not hasattr(self, 'merge_map'):
            # build a dict for O(1) bigram→merged lookup
            self.merge_map = {(a, b): a + b for (a, b) in self.merges}

    def encode(self, text: str) -> List[int]:
        """
        Exactly the same logic as before:
         - split on whitespace,
         - optionally surround entities,
         - run each raw word (no further wrapping) through the cached BPE,
         - then close off any entities again.
        """
        out: List[int] = []
        ent_start = self.token_to_id["<ENT_START>"]
        ent_end = self.token_to_id["<ENT_END>"]
        is_ent = self._is_probable_entity

        # NB: we do NOT call self.preprocess(...) here.
        for w in text.split():
            if is_ent(w):
                out.append(ent_start)

            # use the cached BPE word‐piece encoder if present,
            # falling back to the impl if not
            bpe_fn = getattr(self, "_encode_word", self._encode_word_impl)
            out.extend(bpe_fn(w))

            if is_ent(w):
                out.append(ent_end)

        return out

    def decode(self, ids: List[int]) -> str:
        # pull dict locally to avoid attribute lookup in loop
        id2tok = self.id_to_token
        words: List[str] = []
        cur: List[str] = []

        for tid in ids:
            tok = id2tok.get(tid, "")
            if tok in ("<ENT_START>", "<ENT_END>"):
                continue
            if tok.endswith("⟂"):
                cur.append(tok[:-1])
                words.append("".join(cur))
                cur = []
            else:
                cur.append(tok)

        if cur:
            words.append("".join(cur))
        return " ".join(words)
