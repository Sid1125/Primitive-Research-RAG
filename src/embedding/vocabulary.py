"""
Vocabulary Builder
Builds a word-to-index mapping from the ingested corpus.
"""

import json
import os
from typing import List, Dict, Optional
from collections import Counter


class Vocabulary:
    """
    Build and manage a vocabulary from corpus text.

    Special tokens:
        0 = <PAD>
        1 = <UNK>
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_size: int = 20000):
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts: List[str]):
        """Build vocabulary from a list of preprocessed text strings."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Most common words (minus 2 for special tokens)
        most_common = counter.most_common(self.max_size - 2)

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  Vocabulary built: {len(self.word2idx)} tokens")

    def encode(self, text: str, max_length: int) -> List[int]:
        """Convert text to padded list of token indices."""
        tokens = text.split()[:max_length]
        indices = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK

        # Pad to max_length
        indices += [0] * (max_length - len(indices))
        return indices

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path: str):
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    @property
    def size(self) -> int:
        return len(self.word2idx)
