from char_tokenizer import CharTokenizer

t = CharTokenizer("output/tokenizer.json")
text = "To be, or not to be, that is the question:"

ids = t.encode(text, add_special_tokens=False)
dec = t.decode(ids, skip_special_tokens=True)
print(f"Original: [{text}]")
print(f"Decoded:  [{dec}]")
print(f"Match:    {text == dec}")
print(f"Vocab size: {t.vocab_size}")
print(f"Space ID: {t.vocab.get(' ')}")
