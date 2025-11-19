"""
This module implements class(es) and function(s) for dataset representation
"""
from typing import Tuple
from dataclasses import dataclass
import json

import torch
from torch.utils.data import Dataset

from tokenizer import Pair, Tokenizer


@dataclass
class Sample:
    enc_inp_ids: torch.LongTensor
    dec_inp_ids: torch.LongTensor
    label_ids: torch.LongTensor


class SeqPairDataset(Dataset):
    def __init__(
            self,
            data_file: str,
            tokenizer: Tokenizer,
            max_src_len: int,
            max_tgt_len: int
    ):
        self.samples = []
        self.tokenizer = tokenizer

        with open(data_file, 'r') as json_file:
            data = json.load(json_file)

        for datapoint in data:
            pair = Pair(src=datapoint["src"], tgt=datapoint["tgt"])

            src_ids = tokenizer.encode(tokenizer.tokenize(pair.src))
            tgt_ids = tokenizer.encode(tokenizer.tokenize(pair.tgt))

            src_ids = self._add_specials_and_trim(src_ids, max_src_len)
            tgt_ids = self._add_specials_and_trim(tgt_ids, max_tgt_len)
            
            src_ids = self._pad(src_ids, max_src_len)
            tgt_ids = self._pad(tgt_ids, max_tgt_len)

            self.samples.append(Sample(src_ids, tgt_ids[:-1], tgt_ids[1:]))

    def _add_specials_and_trim(self, token_ids, max_len):
        trimmed_ids = token_ids[:max_len - 2]
        trimmed_ids.append(self.tokenizer.eos_id)
        tok_ids = [self.tokenizer.bos_id]
        tok_ids.extend(trimmed_ids)
        return tok_ids

    def _pad(self, token_ids, max_len):
        diff = max_len - len(token_ids)
        if diff > 0:
            token_ids.extend([self.tokenizer.pad_id for _ in range(diff)])
        return torch.LongTensor(token_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[idx]
        return (sample.enc_inp_ids, sample.dec_inp_ids, sample.label_ids)
        # Returns (encoder_input_ids, decoder_input_ids, label_ids)
        # All three should be torch.LongTensor objects

