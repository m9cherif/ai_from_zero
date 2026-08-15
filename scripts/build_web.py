"""Inline the model and the inference engine into one self-contained page.

    python scripts/export_web.py --out web/model.json
    python scripts/build_web.py --out web/marginalia.html

The result needs no server and no network: weights, tokenizer, inference and UI
all travel in the file.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=os.path.join(HERE, "web", "template.html"))
    parser.add_argument("--engine", default=os.path.join(HERE, "web", "inference.js"))
    parser.add_argument("--model", default=os.path.join(HERE, "web", "model.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "web", "marginalia.html"))
    args = parser.parse_args()

    html = open(args.template, encoding="utf-8").read()
    engine = open(args.engine, encoding="utf-8").read()
    payload = open(args.model, encoding="utf-8").read()

    # The payload rides in a <script type="application/json"> block, so the only
    # sequence that could end it early is a literal "</script>". Base64 and JSON
    # cannot produce one, but escape defensively rather than rely on that.
    payload = payload.replace("</", "<\\/")

    html = html.replace("/*INFERENCE*/", engine).replace("/*PAYLOAD*/", payload)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    size = os.path.getsize(args.out)
    print(f"  wrote {args.out}")
    print(f"  size  {size/1e6:.2f} MB  (artifact limit 16 MB)")
    if size > 16_000_000:
        print("  OVER THE LIMIT - quantize further or shrink the model", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
