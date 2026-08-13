"""Character-level tokenizer wrapper."""
import json
from typing import List

class CharTokenizer:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]["token_to_id"]
        self.id_to_t = {int(k): v for k, v in data["vocab"].get("id_to_token", {}).items()}
        if not self.id_to_t:
            self.id_to_t = {v: k for k, v in self.vocab.items()}
        special = data["vocab"]["special_names"]
        self._pad = self.vocab[special["pad"]]
        self._unk = self.vocab[special["unk"]]
        self._bos = self.vocab[special["bos"]]
        self._eos = self.vocab[special["eos"]]

    def encode(self, text, add_special_tokens=True):
        ids = [self.vocab.get(c, self._unk) for c in text]
        if add_special_tokens:
            ids = [self._bos] + ids + [self._eos]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for i in ids:
            if skip_special_tokens and i in (self._pad, self._unk, self._bos, self._eos):
                continue
            tokens.append(self.id_to_t.get(i, ""))
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self._pad

    @property
    def eos_token_id(self):
        return self._eos

    @property
    def bos_token_id(self):
        return self._bos
