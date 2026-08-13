import json
with open("output/tokenizer.json", "r") as f:
    data = json.load(f)
t2i = data["vocab"]["token_to_id"]
print(f"Total vocab entries: {len(t2i)}")
has_space = " " in t2i
print(f"Space in vocab: {has_space}")
if has_space:
    print(f"Space value: {t2i[' ']}")
# check if any key is space-like
for k, v in list(t2i.items())[-10:]:
    print(f"  key={repr(k)} val={v}")
