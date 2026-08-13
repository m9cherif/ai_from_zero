"""Creates a character-level tokenizer, ensuring spaces are properly encoded."""

import json

with open("data/tinyshakespeare.txt", "r") as f:
    text = f.read()

all_chars = sorted(list(set(text)))
char_to_id = {}
id_to_char = {}

pad_id = 0
unk_id = 1
bos_id = 2
eos_id = 3
mask_id = 4

next_id = 5

# Add special tokens
char_to_id["[PAD]"] = pad_id
id_to_char[pad_id] = "[PAD]"

char_to_id["[UNK]"] = unk_id
id_to_char[unk_id] = "[UNK]"

char_to_id["[BOS]"] = bos_id
id_to_char[bos_id] = "[BOS]"

char_to_id["[EOS]"] = eos_id
id_to_char[eos_id] = "[EOS]"

char_to_id["[MASK]"] = mask_id
id_to_char[mask_id] = "[MASK]"

# Add all characters (including space)
for ch in all_chars:
    if ch not in char_to_id:
        char_to_id[ch] = next_id
        id_to_char[next_id] = ch
        next_id += 1

# Verify space is in vocab
assert " " in char_to_id, "Space must be in vocabulary!"
space_id = char_to_id[" "]
print(f"Space ID: {space_id}")
print(f"Total vocab: {len(char_to_id)}")

tokenizer_data = {
    "vocab": {
        "token_to_id": char_to_id,
        "id_to_token": id_to_char,
        "special_names": {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "bos": "[BOS]",
            "eos": "[EOS]",
            "mask": "[MASK]",
        },
    },
    "merges": [],
}

with open("output/tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, ensure_ascii=False)

# Test round-trip
test = "To be, or not to be, that is the question:"
ids = [char_to_id.get(c, unk_id) for c in test]
decoded = "".join(id_to_char[i] for i in ids)
print(f"Original: [{test}]")
print(f"Decoded:  [{decoded}]")
print(f"Match: {test == decoded}")
