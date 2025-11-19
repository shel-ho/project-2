"""
The module implements models (and all their components) to train on in this project.
"""

from math import log
from typing import Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_padding_mask(seq: torch.Tensor, pad_idx=0):
    """
    seq: (batch, seq_len)
    Returns a mask (batch, seq_len) of booleans: True for non-pad tokens, False for pad.
    """
    return seq != pad_idx


def make_causal_mask(sz):
    """
    Creates a causal mask: shape (sz, sz),
    where positions (i, j) are True if j <= i, False if j > i.
    """
    mask = torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)
    mask = ~mask
    return mask  # (sz, sz)


def make_decoder_self_mask(tgt_seq, pad_idx=0):
    """
    Combines causal + padding mask for decoder self-attention.
    Returns mask of shape (batch, 1, T, T).
    """
    B, T = tgt_seq.size()
    pad_mask = make_padding_mask(tgt_seq, pad_idx)  # (B, T)
    causal = make_causal_mask(T).to(tgt_seq.device)  # (T, T)

    # Expand padding mask for keys: (B, T) -> (B, 1, 1, T)
    # We only mask keys (columns), not queries (rows), to avoid all-inf rows
    mask_k = pad_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T) - for keys

    # Expand causal mask to batch and head dimensions
    causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    # Combine masks: causal AND (no padding in keys)
    combined = causal & mask_k  # (B, 1, T, T)

    return combined


class PosEncoding(nn.Module):
    """Positional encoding module"""

    def __init__(self, d_model: int, seq_len: int):
        """
        Assume a single data instance is a list of tokens

        Args:
            d_model (int): the size of the input embeddings
            seq_len (int): the length of input sequence

        Returns: None
        """
        super().__init__()
        assert d_model % 2 == 0, "The embedding size must be divisible by 2"

        pe = torch.zeros(seq_len, d_model)

        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-log(10000) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)  # even-numbered position
        pe[:, 1::2] = torch.cos(pos * div_term)  # odd-numbered position

        pe = pe.unsqueeze(0)  # dim: (1, seq_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function
        Args:
            x (torch.Tensor): word embeddings for input sequence with the shape (B, seq_len, d_model)

        Returns: new embeddings that add positional embeddings
        """
        seq_len = x.size(1)
        x_new = x + self.pe[:, :seq_len, :]
        return x_new


class LearnablePosEncoding(nn.Module):
    """Positional encoding module with learnable positional encodings"""

    def __init__(self, d_model: int, seq_len: int):
        """
        Assume a single data instance is a list of tokens

        Args:
            d_model (int): the size of the input embeddings
            seq_len (int): the length of input sequence

        Returns: None
        """
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function
        Args:
            x (torch.Tensor): word embeddings for input sequence with the shape (B, seq_len, d_model)

        Returns: new embeddings that add positional embeddings
        """
        seq_len = x.size(1)
        x_new = x + self.pe(torch.tensor(range(seq_len), device=x.device))
        return x_new


class MHA(nn.Module):
    """Multihead attention"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head**-0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None
    ) -> Tuple[torch.Tensor, ...]:
        """

        Args:
            q: 'Query' matrix
            k: 'Key' matrix
            v: 'Value' matrix
            mask: (optional) attention mask

        Returns: a tuple of attention scores and attention output

        """
        B, Tq, _ = q.size()
        _, Tk, _ = k.size()

        Q = self.w_q(q).view(B, Tq, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(B, Tk, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(B, Tk, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, Tq, Tk)

        if mask is not None:
            assert mask.dim() == scores.dim(), (
                f"Provided mask has different dimension ({mask.dim()}) as QK scores ({scores.dim()})"
            )
            scores = scores.masked_fill(mask == 0, -torch.inf)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, heads, Tq, d_head)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.num_heads * self.d_head)
        out = self.w_o(out)
        return out, attn


class FFN(nn.Module):
    """Feedforward neural network"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = F.relu(self.lin1(x))
        x2 = self.dropout(x2)
        x2 = self.lin2(x2)
        return x2


class EncoderLayer(nn.Module):
    """Single layer of Transformer encoder"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MHA(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FFN(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x, attn


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        pad_idx: int,
        dropout=0.1,
        exp_1: bool =False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if exp_1: 
            self.pos_emb = LearnablePosEncoding(d_model, max_len)
        else:
            self.pos_emb = PosEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pad_idx = pad_idx

    def forward(self, input_ids: torch.Tensor):
        x = self.pos_emb(self.token_emb(input_ids))
        mask = make_padding_mask(input_ids, pad_idx=self.pad_idx)  # (B, T)

        # Expand mask for multi-head attention: (B, T) -> (B, 1, 1, T)
        mask_exp = mask.unsqueeze(1).unsqueeze(2)

        attns = []
        for layer in self.layers:
            x, a = layer(x, mask_exp)
            attns.append(a)
        x = self.norm(x)

        return x, mask, attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout=0.1):
        super().__init__()
        self.self_attn = MHA(d_model, num_heads, dropout)
        self.cross_attn = MHA(d_model, num_heads, dropout)
        self.ff = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_outputs, self_mask=None, cross_mask=None):
        sa, a1 = self.self_attn(x, x, x, mask=self_mask)
        x = x + self.dropout(sa)
        x = self.norm1(x)
        ca, a2 = self.cross_attn(x, enc_outputs, enc_outputs, mask=cross_mask)
        x = x + self.dropout(ca)
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm3(x)
        return x, (a1, a2)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float =0.1,
        pad_idx: int =0,
        exp_1: bool =False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if exp_1: 
            self.pos_emb = LearnablePosEncoding(d_model, max_len)
        else:
            self.pos_emb = PosEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pad_idx = pad_idx

    def forward(self, tgt_ids, enc_outputs, enc_mask):
        # Build decoder self-attention mask
        self_mask = make_decoder_self_mask(tgt_ids, pad_idx=self.pad_idx)

        # Expand encoder mask to broadcast shape for cross-attention:
        # enc_mask: (B, src_len) -> (B, 1, 1, src_len)
        enc_mask_exp = enc_mask.unsqueeze(1).unsqueeze(2)

        x = self.pos_emb(self.token_emb(tgt_ids))

        attn_all = []
        for layer in self.layers:
            x, attns = layer(
                x, enc_outputs, self_mask=self_mask, cross_mask=enc_mask_exp
            )
            attn_all.append(attns)
        x = self.norm(x)
        return x, attn_all


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        max_len: int = 50,
        dropout: float = 0.1,
        pad_idx: int = 0,
        exp_1: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_enc_layers,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx,
            exp_1=exp_1,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_dec_layers,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx,
            exp_1=exp_1,
        )
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids):
        enc_out, enc_mask, _ = self.encoder(src_ids)
        dec_out, _ = self.decoder(tgt_ids, enc_out, enc_mask)
        logits = self.out_proj(dec_out)
        return logits

    def _greedy_decode(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        max_len: int,
        eos_id: int
    ) -> Sequence[int]:
        """
        Greedy decoding: at each step, select the token with highest probability.

        Args:
            src_ids: source sequence tensor of shape (1, src_len)
            tgt_ids: initial target sequence (typically just [bos_id]), shape (1, 1)
            max_len: maximum generation length
            eos_id: end-of-sequence token id

        Returns:
            List of generated token ids
        """
        for _ in range(max_len):
            # Forward pass through model
            logits = self(src_ids, tgt_ids)  # (1, current_len, vocab_size)

            # Get logits for the last position
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # Greedy: select token with highest probability
            next_token = torch.argmax(next_token_logits, dim=-1)  # (1,)

            # Append to target sequence
            tgt_ids = torch.cat([tgt_ids, next_token.unsqueeze(0)], dim=1)  # (1, current_len + 1)

            # Stop if EOS token is generated
            if next_token.item() == eos_id:
                break

        return tgt_ids.squeeze(0).tolist()

    def _beam_search(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        beam_width: int,
        max_len: int,
        eos_id: int
    ) -> Sequence[int]:
        """
        Beam search decoding: maintain top-k most probable sequences at each step.

        Args:
            src_ids: source sequence tensor of shape (1, src_len)
            tgt_ids: initial target sequence (typically just [bos_id]), shape (1, 1)
            beam_width: number of beams to keep
            max_len: maximum generation length
            eos_id: end-of-sequence token id

        Returns:
            List of generated token ids (best sequence)
        """
        # Initialize beams: list of (sequence, log_prob)
        beams = [(tgt_ids, 0.0)]
        completed_sequences = []

        for _ in range(max_len):
            candidates = []

            for seq, score in beams:
                # If sequence already ended, keep it as is
                if seq[0, -1].item() == eos_id:
                    completed_sequences.append((seq, score))
                    continue

                # Forward pass
                logits = self(src_ids, seq)  # (1, seq_len, vocab_size)

                # Get log probabilities for next token
                next_token_logits = logits[:, -1, :]  # (1, vocab_size)
                log_probs = torch.log_softmax(next_token_logits, dim=-1)  # (1, vocab_size)

                # Get top-k tokens
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                # Create new candidates
                for i in range(beam_width):
                    token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # (1, 1)
                    token_log_prob = topk_log_probs[0, i].item()

                    new_seq = torch.cat([seq, token_id], dim=1)  # (1, seq_len + 1)
                    new_score = score + token_log_prob

                    candidates.append((new_seq, new_score))

            # Select top beam_width candidates
            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            # Check if all beams have ended
            if all(seq[0, -1].item() == eos_id for seq, _ in beams):
                completed_sequences.extend(beams)
                break

        # Add remaining beams to completed sequences
        completed_sequences.extend(beams)

        # Return best sequence
        if completed_sequences:
            best_seq, _ = max(completed_sequences, key=lambda x: x[1])
            return best_seq.squeeze(0).tolist()
        else:
            return beams[0][0].squeeze(0).tolist()

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 100,
        strategy: str = "greedy",
        beam_width: int = 5
    ) -> Sequence[Sequence[int]]:
        """
        Generate sequences in batch using specified decoding strategy.

        Args:
            src_ids: Source sequences tensor of shape (batch_size, src_len)
            bos_id: Beginning-of-sequence token id
            eos_id: End-of-sequence token id
            max_len: Maximum generation length
            strategy: Decoding strategy - "greedy" or "beam_search"
            beam_width: Beam width for beam search (only used if strategy="beam_search")

        Returns:
            List of generated sequences, where each sequence is a list of token ids
        """
        self.eval()
        device = next(self.parameters()).device
        src_ids = src_ids.to(device)
        batch_size = src_ids.size(0)

        generated_sequences = []

        # Process each sample in the batch
        for i in range(batch_size):
            # Get single source sequence
            single_src = src_ids[i:i+1]  # (1, src_len)

            # Initialize with BOS token
            tgt_ids = torch.LongTensor([[bos_id]]).to(device)  # (1, 1)

            # Generate using specified strategy
            if strategy == "greedy":
                generated_ids = self._greedy_decode(
                    src_ids=single_src,
                    tgt_ids=tgt_ids,
                    max_len=max_len,
                    eos_id=eos_id
                )
            elif strategy == "beam_search":
                generated_ids = self._beam_search(
                    src_ids=single_src,
                    tgt_ids=tgt_ids,
                    beam_width=beam_width,
                    max_len=max_len,
                    eos_id=eos_id
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'greedy' or 'beam_search'.")

            generated_sequences.append(generated_ids)

        return generated_sequences
