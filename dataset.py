"""
dataset.py — Multi30k Dataset + Vocabulary
DA6401 Assignment 3

Loads bentrevett/multi30k from HuggingFace, tokenises with spaCy
(de_core_news_sm / en_core_web_sm), and builds integer vocabularies
with four fixed special tokens: <unk>(0)  <pad>(1)  <sos>(2)  <eos>(3).

Quick-start:
    from dataset import build_dataloaders
    train_dl, val_dl, test_dl, src_vocab, tgt_vocab = build_dataloaders()
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """String ↔ integer vocabulary with four fixed special tokens."""

    SPECIALS           = ["<unk>", "<pad>", "<sos>", "<eos>"]
    UNK, PAD, SOS, EOS = 0, 1, 2, 3

    def __init__(self) -> None:
        self.itos: List[str] = list(self.SPECIALS)
        self.stoi: dict      = {t: i for i, t in enumerate(self.SPECIALS)}

    # ── building ──────────────────────────────────────────────────────

    def build_from_token_lists(
        self,
        token_lists: List[List[str]],
        min_freq: int = 2,
    ) -> None:
        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    # ── lookup ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.UNK)

    def lookup_token(self, idx: int) -> str:
        """idx → token string (supports tgt_vocab.lookup_token(idx))."""
        return self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>"

    def numericalize(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.UNK) for t in tokens]

    def denumericalize(
        self,
        indices: List[int],
        strip_special: bool = True,
    ) -> List[str]:
        tokens = [self.lookup_token(i) for i in indices]
        if strip_special:
            tokens = [t for t in tokens if t not in self.SPECIALS]
        return tokens


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    PyTorch Dataset wrapping bentrevett/multi30k.

    Each item is a (src_ids, tgt_ids) pair of LongTensors that include
    <sos> and <eos> boundary markers.
    """

    def __init__(
        self,
        split:          str                  = "train",
        src_vocab:      Optional[Vocabulary] = None,
        tgt_vocab:      Optional[Vocabulary] = None,
        src_tokenizer                        = None,
        tgt_tokenizer                        = None,
        min_freq:       int                  = 2,
    ) -> None:
        self.split = split

        # ── HuggingFace dataset ───────────────────────────────────────
        from datasets import load_dataset
        raw           = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self.raw_data = raw[split]

        # ── spaCy tokenisers ──────────────────────────────────────────
        import spacy
        self.de_nlp = src_tokenizer or spacy.load("de_core_news_sm")
        self.en_nlp = tgt_tokenizer or spacy.load("en_core_web_sm")

        # ── tokenise all sentences ─────────────────────────────────────
        self.src_tokens: List[List[str]] = [
            self._tok_de(row["de"]) for row in self.raw_data
        ]
        self.tgt_tokens: List[List[str]] = [
            self._tok_en(row["en"]) for row in self.raw_data
        ]

        # ── build / reuse vocabularies ─────────────────────────────────
        if src_vocab is None:
            src_vocab = Vocabulary()
            src_vocab.build_from_token_lists(self.src_tokens, min_freq)
        if tgt_vocab is None:
            tgt_vocab = Vocabulary()
            tgt_vocab.build_from_token_lists(self.tgt_tokens, min_freq)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # ── numericalize ───────────────────────────────────────────────
        self.src_ids: List[List[int]] = [
            [Vocabulary.SOS] + src_vocab.numericalize(t) + [Vocabulary.EOS]
            for t in self.src_tokens
        ]
        self.tgt_ids: List[List[int]] = [
            [Vocabulary.SOS] + tgt_vocab.numericalize(t) + [Vocabulary.EOS]
            for t in self.tgt_tokens
        ]

    def _tok_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def _tok_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self.src_ids[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_ids[idx], dtype=torch.long)
        return src, tgt

    def build_vocab(self):
        """Return (src_vocab, tgt_vocab) — kept for API compatibility."""
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """Return (src_ids, tgt_ids) lists of integer lists."""
        return self.src_ids, self.tgt_ids


# ══════════════════════════════════════════════════════════════════════
#  COLLATE  (pads variable-length sequences in a batch)
# ══════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_idx: int = Vocabulary.PAD):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#  CONVENIENCE BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_dataloaders(
    batch_size:  int = 128,
    min_freq:    int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Build train / val / test DataLoaders for Multi30k.

    Returns:
        train_dl, val_dl, test_dl, src_vocab, tgt_vocab
    """
    print("Loading Multi30k train split …")
    train_ds  = Multi30kDataset(split="train", min_freq=min_freq)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab
    print(f"  src vocab : {len(src_vocab):,}   tgt vocab : {len(tgt_vocab):,}")

    print("Loading validation split …")
    val_ds = Multi30kDataset(
        split="validation",
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        src_tokenizer=train_ds.de_nlp, tgt_tokenizer=train_ds.en_nlp,
    )

    print("Loading test split …")
    test_ds = Multi30kDataset(
        split="test",
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        src_tokenizer=train_ds.de_nlp, tgt_tokenizer=train_ds.en_nlp,
    )

    _col = lambda b: collate_fn(b, Vocabulary.PAD)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=_col, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          collate_fn=_col, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                          collate_fn=_col, num_workers=num_workers)

    return train_dl, val_dl, test_dl, src_vocab, tgt_vocab


if __name__ == "__main__":
    train_dl, val_dl, test_dl, sv, tv = build_dataloaders(batch_size=32)
    src, tgt = next(iter(train_dl))
    print("src:", src.shape, "  tgt:", tgt.shape)
    print("src sample:", " ".join(sv.denumericalize(src[0].tolist())))
    print("tgt sample:", " ".join(tv.denumericalize(tgt[0].tolist())))
