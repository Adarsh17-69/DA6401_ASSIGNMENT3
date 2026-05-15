"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

Design choices:
  • Pre-LayerNorm  — normalise *before* each sub-layer (more stable than
                     Post-LN in the original paper; avoids early divergence).
  • Mask convention — True  → position is MASKED OUT (set to −∞ before softmax).
  • Transformer.decode() returns final logits (vocab projection included).
  • Transformer() callable with ZERO arguments — auto-loads config + weights
    from a saved checkpoint (required by the autograder contract).
"""

import os
import math
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q, K, V  : (..., seq, d_k / d_v)
        mask     : BoolTensor broadcastable to (..., seq_q, seq_k).
                   True → masked out (receives −∞ before softmax).
        use_scale: If False, skip the √dₖ divisor (ablation §2.2).
    Returns:
        output   : (..., seq_q, d_v)
        attn_w   : (..., seq_q, seq_k)  post-softmax weights
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)   # handle all-masked rows
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Padding mask for the encoder.

    Returns BoolTensor [batch, 1, 1, src_len]
        True  → PAD token → masked out in attention.
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """
    Combined padding + causal mask for the decoder.

    Returns BoolTensor [batch, 1, tgt_len, tgt_len]
        True  → PAD token OR future position → masked out.
    """
    B, T   = tgt.shape
    pad_m  = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)              # [B,1,1,T]
    causal = ~torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=tgt.device)
    ).unsqueeze(0).unsqueeze(0)                                      # [1,1,T,T]
    return pad_m | causal                                            # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head₁,…,headₕ) · W_O
        headᵢ = Attention(Q·W_Qᵢ, K·W_Kᵢ, V·W_Vᵢ)

    NOT using torch.nn.MultiheadAttention.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        dropout:   float = 0.1,
        use_scale: bool  = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout      = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None  # saved for visualisation

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, d_model] → [B, h, S, d_k]"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, S, d_k] → [B, S, d_model]"""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value : [batch, seq, d_model]
            mask : BoolTensor broadcastable to [B, h, sq, sk]; True → masked out.
        Returns:
            output : [batch, seq_q, d_model]
        """
        Q = self._split(self.W_q(query))
        K = self._split(self.W_k(key))
        V = self._split(self.W_v(value))

        attn_out, attn_w = scaled_dot_product_attention(
            Q, K, V, mask=mask, use_scale=self.use_scale
        )
        self.attn_weights = attn_w.detach()
        return self.W_o(self._merge(self.dropout(attn_out)))


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE (registered as a buffer, NOT a parameter):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int   = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                  # [1, max, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [batch, seq_len, d_model] → same shape"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  LEARNED POSITIONAL ENCODING  (ablation §2.4)
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Drop-in replacement using nn.Embedding (trainable)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(pos))


# ══════════════════════════════════════════════════════════════════════
#  POSITION-WISE FEED-FORWARD
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  TOKEN EMBEDDING  (scales by √d_model as per the paper §3.4)
# ══════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model   = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  (Pre-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
        use_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.dropout(self.self_attn(n, n, n, src_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Pre-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.1,
        use_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.dropout(self.self_attn(n, n, n, tgt_mask))
        n = self.norm2(x)
        x = x + self.dropout(self.cross_attn(n, memory, memory, src_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """N identical encoder layers + final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """N identical decoder layers + final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

# Ordered list of checkpoint paths the autograder might place the file at
_CKPT_SEARCH = [
    "best_checkpoint.pt",
    "checkpoint.pt",
    "checkpoints/best_checkpoint.pt",
    "checkpoints/checkpoint.pt",
]


class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer.

    ZERO-ARGUMENT CONSTRUCTION
    --------------------------
    The autograder calls ``Transformer()`` with no arguments, then evaluates
    the loaded model on a held-out test set.  To support this, the constructor:

      1. Searches for a saved checkpoint in standard locations.
      2. Reads ``model_config`` from that checkpoint.
      3. Builds the network with those dimensions.
      4. Loads the saved ``model_state_dict``.

    This means you must run training and save a checkpoint before submitting.
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:    int   = 256,
        N:          int   = 3,
        num_heads:  int   = 8,
        d_ff:       int   = 512,
        dropout:    float = 0.1,
        use_scale:  bool  = True,
        learned_pe: bool  = False,
        max_len:    int   = 256,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Capture zero-arg intent BEFORE anything is resolved
        _zero_arg = (src_vocab_size is None and
                     tgt_vocab_size is None and
                     checkpoint_path is None)

        # ── Step 1: find and load checkpoint (if needed) ──────────────
        _state_dict: Optional[dict] = None
        _cfg:        dict           = {}

        search_paths = ([checkpoint_path] if checkpoint_path else []) + _CKPT_SEARCH
        for p in search_paths:
            if p and os.path.exists(p):
                ckpt       = torch.load(p, map_location="cpu")
                _cfg       = ckpt.get("model_config", {})
                _state_dict = ckpt.get("model_state_dict")
                break

        # ── Step 2: resolve vocab sizes (explicit arg > checkpoint) ───
        src_vocab_size = src_vocab_size if src_vocab_size is not None \
                         else _cfg.get("src_vocab_size")
        tgt_vocab_size = tgt_vocab_size if tgt_vocab_size is not None \
                         else _cfg.get("tgt_vocab_size")

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError(
                "src_vocab_size and tgt_vocab_size must be provided explicitly "
                "or be present in a checkpoint at one of: " + str(_CKPT_SEARCH)
            )

        # ── Step 3: resolve architecture ──────────────────────────────
        # If called with zero args, pull ALL dimensions from the checkpoint.
        if _zero_arg and _state_dict is not None:
            src_vocab_size = _cfg.get("src_vocab_size", src_vocab_size)
            tgt_vocab_size = _cfg.get("tgt_vocab_size", tgt_vocab_size)
            d_model   = _cfg.get("d_model",   d_model)
            N         = _cfg.get("N",         N)
            num_heads = _cfg.get("num_heads", num_heads)
            d_ff      = _cfg.get("d_ff",      d_ff)
            dropout   = _cfg.get("dropout",   dropout)
            use_scale = _cfg.get("use_scale", use_scale)
        else:
            # Explicit construction — Step 2 already resolved vocab sizes
            pass

        # Store for save_checkpoint
        self.d_model   = d_model
        self.use_scale = use_scale

        # ── Step 4: build architecture ────────────────────────────────
        self.src_tok_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_embed = TokenEmbedding(tgt_vocab_size, d_model)

        PE = LearnedPositionalEncoding if learned_pe else PositionalEncoding
        self.src_pos_enc = PE(d_model, dropout, max_len)
        self.tgt_pos_enc = PE(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Vocabulary references (set after dataset construction)
        self.src_vocab     = None
        self.tgt_vocab     = None
        self.src_tokenizer = None
        self.pad_idx       = 1
        self.sos_idx       = 2
        self.eos_idx       = 3

        # ── Step 5: init weights, then overwrite with checkpoint ──────
        self._init_weights()
        if _state_dict is not None:
            self.load_state_dict(_state_dict)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        src      : [B, S]  token indices
        src_mask : [B, 1, 1, S]
        Returns  : memory [B, S, d_model]
        """
        x = self.src_pos_enc(self.src_tok_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        memory   : [B, S, d_model]
        tgt      : [B, T]  token indices
        Returns  : logits [B, T, tgt_vocab_size]
        """
        x = self.tgt_pos_enc(self.tgt_tok_embed(tgt))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full encoder-decoder pass → logits [B, T, tgt_vocab]."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    # ── vocabulary helpers ────────────────────────────────────────────

    def set_vocab(
        self,
        src_vocab,
        tgt_vocab,
        src_tokenizer=None,
        pad_idx: int = 1,
        sos_idx: int = 2,
        eos_idx: int = 3,
    ) -> None:
        self.src_vocab     = src_vocab
        self.tgt_vocab     = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.pad_idx       = pad_idx
        self.sos_idx       = sos_idx
        self.eos_idx       = eos_idx

    # ── raw-string inference ──────────────────────────────────────────

    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        """Translate a raw German string to English (greedy decoding)."""
        if self.src_vocab is None:
            raise RuntimeError("Call model.set_vocab() before infer().")

        self.eval()
        device = next(self.parameters()).device
        tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        unk    = self.src_vocab.stoi.get("<unk>", 0)
        ids    = (
            [self.sos_idx]
            + [self.src_vocab.stoi.get(t, unk) for t in tokens]
            + [self.eos_idx]
        )
        src      = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, self.pad_idx)

        from train import greedy_decode
        out = greedy_decode(
            self, src, src_mask, max_len,
            self.sos_idx, self.eos_idx, str(device),
        )
        result = []
        for idx in out[0].tolist()[1:]:
            if idx == self.eos_idx:
                break
            tok = self.tgt_vocab.lookup_token(idx)
            if tok not in ("<sos>", "<eos>", "<pad>", "<unk>"):
                result.append(tok)
        return " ".join(result)
