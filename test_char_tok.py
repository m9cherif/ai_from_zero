from char_tokenizer import CharTokenizer

t = CharTokenizer("output/tokenizer.json")
text = "To be, or not to be"

# Without special tokens
ids_no_special = t.encode(text, add_special_tokens=False)
dec_no_special = t.decode(ids_no_special, skip_special_tokens=True)
print(f"Original:      [{text}]")
print(f"No_special:    [{dec_no_special}]")

# With special tokens
ids_special = t.encode(text, add_special_tokens=True)
dec_special = t.decode(ids_special, skip_special_tokens=True)
print(f"With_special:  [{dec_special}]")

# Check space
space_id = t.vocab.get(" ")
print(f"Space char ID: {space_id}")
chars = [t.id_to_t.get(i, "?") for i in ids_no_special[:15]]
print(f"First 15 chars: {chars}")
print(f"First 15 ids:   {ids_no_special[:15]}")
