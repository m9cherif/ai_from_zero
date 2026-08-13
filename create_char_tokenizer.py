"""Creates a character-level tokenizer that preserves spaces exactly."""
import json

with open("data/tinyshakespeare.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))
base_offset = 6

special = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
    "[MASK]": 4,
    " ": 5,
}

char_ids = {ch: i + base_offset for i, ch in enumerate(chars)}
all_vocab = {**special, **char_ids}
id_to_token = {v: k for k, v in all_vocab.items()}

tokenizer_data = {
    "vocab": {
        "token_to_id": all_vocab,
        "id_to_token": id_to_token,
        "special_names": {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "bos": "[BOS]",
            "eos": "[EOS]",
            "mask": "[MASK]",
        },
        "max_size": None,
    },
    "merges": [],
    "max_token_length": None,
}

with open("output/tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, ensure_ascii=False)

print(f"Tokenizer created with {len(all_vocab)} tokens")
text_test = "To be, or not to be, that is the question:"
ids_test = [all_vocab.get(c, 1) for c in text_test]
decoded_test = "".join(id_to_token[i] for i in ids_test)
print(f"Original: [{text_test}]")
print(f"Decoded:  [{decoded_test}]")
print(f"Match: {text_test == decoded_test}")
