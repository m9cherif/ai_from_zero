"""Search the HuggingFace Hub for text datasets worth training on.

    python scripts/list_datasets.py                       # top 200 English text
    python scripts/list_datasets.py --limit 20000 --out catalogue.json
    python scripts/list_datasets.py --search shakespeare --language en

Sorting the Hub by downloads alone returns image folders and chat logs, so the
listing is filtered to text modalities and ranked by a score that rewards both
reach and approval. The result is a catalogue you can hand to fetch_corpus.py,
which downloads from it up to a size budget you choose.

Size comes from the Hub's ``size_categories`` tag and is a row count, not bytes -
useful for ordering, not for arithmetic. Exact sizes are resolved at download
time, where they matter.
"""

import argparse
import json
import math
import os
import subprocess
import sys

API = "https://huggingface.co/api/datasets"

# size_categories tag -> representative row count, for ordering only.
SIZE_ROWS = {
    "n<1K": 500, "1K<n<10K": 5_000, "10K<n<100K": 50_000,
    "100K<n<1M": 500_000, "1M<n<10M": 5_000_000, "10M<n<100M": 50_000_000,
    "100M<n<1B": 500_000_000, "1B<n<10B": 5_000_000_000,
    "10B<n<100B": 50_000_000_000, "n>1T": 1_000_000_000_000,
}

TEXT_FORMATS = ("text", "json", "parquet", "csv")
REJECT_MODALITIES = ("image", "audio", "video", "3d", "geospatial")
# Tasks whose data is labels and short fragments rather than running prose.
# Running prose, which is what a language model wants. Benchmarks of short
# question/answer fragments are text but poor pretraining material, so they
# qualify without earning the bonus in score().
TEXT_TASKS = (
    "text-generation", "fill-mask", "text2text-generation", "summarization",
    "translation", "question-answering", "text-classification",
    "sentence-similarity", "feature-extraction",
)

PROSE_TASKS = ("text-generation", "fill-mask", "text2text-generation", "summarization")

REJECT_TASKS = (
    "image-classification", "object-detection", "image-segmentation",
    "text-to-image", "image-to-text", "automatic-speech-recognition",
    "audio-classification", "text-to-speech", "depth-estimation",
    "tabular-classification", "tabular-regression", "reinforcement-learning",
)


def api_page(params, limit, cursor=None):
    query = dict(params)
    query["limit"] = str(limit)
    url = API + "?" + "&".join(f"{k}={v}" for k, v in query.items())
    if cursor:
        url = cursor
    # Headers go to their own file: through a proxy, stdout carries the CONNECT
    # response as well, so splitting stdout on the first blank line lands in the
    # wrong place and the body never parses.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as handle:
        header_path = handle.name
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "90", "-D", header_path, url],
            capture_output=True, text=True,
        )
        try:
            rows = json.loads(result.stdout)
        except json.JSONDecodeError:
            return [], None
        # The Hub paginates with a Link: <url>; rel="next" header.
        next_url = None
        for line in open(header_path, encoding="utf-8", errors="replace"):
            if line.lower().startswith("link:") and 'rel="next"' in line:
                next_url = line.split("<", 1)[1].split(">", 1)[0]
                break
    finally:
        os.unlink(header_path)
    return (rows if isinstance(rows, list) else []), next_url


def tag_value(tags, prefix):
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return None


def usable(entry, language, prose_only=False):
    """Keep datasets that plausibly contain running text in the right language."""
    tags = entry.get("tags") or []

    modality = [t[len("modality:"):] for t in tags if t.startswith("modality:")]
    if any(m in REJECT_MODALITIES for m in modality):
        return False

    formats = [t[len("format:"):] for t in tags if t.startswith("format:")]
    tasks = [t[len("task_categories:"):] for t in tags if t.startswith("task_categories:")]
    if any(t in REJECT_TASKS for t in tasks):
        return False

    # Require positive evidence that this is text. Absence of an image tag is
    # not evidence of prose: robotics logs, 3D asset dumps and scratch caches
    # carry no modality tag at all and would otherwise rank near the top.
    is_text = (
        "text" in modality
        or any(f in TEXT_FORMATS for f in formats)
        or any(t in TEXT_TASKS for t in tasks)
    )
    if not is_text:
        return False

    # Pretraining wants running prose. Benchmarks of question/answer pairs are
    # text but teach a language model little about continuing a sentence.
    if prose_only and not any(t in PROSE_TASKS for t in tasks):
        return False

    if language:
        languages = [t[len("language:"):] for t in tags if t.startswith("language:")]
        # No language tag at all is common and usually means English.
        if languages and language not in languages:
            return False

    if entry.get("private") or entry.get("gated"):
        return False
    return True


def score(entry):
    """Reach and approval on comparable scales.

    Downloads span six orders of magnitude and likes three, so both are taken
    logarithmically; otherwise one popular mirror outranks everything curated.
    """
    downloads = entry.get("downloads") or 0
    likes = entry.get("likes") or 0
    tags = entry.get("tags") or []
    tasks = [t[len("task_categories:"):] for t in tags if t.startswith("task_categories:")]
    # Running prose beats question/answer fragments for pretraining.
    bonus = 1.5 if any(t in PROSE_TASKS for t in tasks) else 0.0
    return round(math.log10(downloads + 1) * 2 + math.log10(likes + 1) * 3 + bonus, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="Datasets to keep")
    parser.add_argument("--scan", type=int, default=None,
                        help="Hub entries to examine (default: 6x --limit)")
    parser.add_argument("--search", default=None, help="Substring to search for")
    parser.add_argument("--language", default="en", help="Language tag; '' for any")
    parser.add_argument("--sort", default="downloads", choices=["downloads", "likes", "lastModified"])
    parser.add_argument("--out", default=None, help="Write the catalogue as JSON")
    parser.add_argument("--show", type=int, default=30, help="Rows to print")
    parser.add_argument(
        "--prose-only", action="store_true",
        help="Keep only datasets tagged for text generation or similar - "
             "running prose rather than question/answer benchmarks.",
    )
    args = parser.parse_args()

    params = {"sort": args.sort, "direction": "-1", "full": "true"}
    if args.search:
        params["search"] = args.search

    scan = args.scan or args.limit * 6
    kept, seen, cursor = [], 0, None
    page = min(1000, max(100, scan))

    print(f"Scanning the Hub (target {args.limit} usable of ~{scan} examined)...",
          file=sys.stderr)
    while seen < scan and len(kept) < args.limit:
        rows, cursor = api_page(params, page, cursor)
        if not rows:
            break
        seen += len(rows)
        for entry in rows:
            if len(kept) >= args.limit:
                break
            if not usable(entry, args.language, args.prose_only):
                continue
            tags = entry.get("tags") or []
            size_tag = tag_value(tags, "size_categories:")
            kept.append({
                "id": entry["id"],
                "downloads": entry.get("downloads") or 0,
                "likes": entry.get("likes") or 0,
                "score": score(entry),
                "size_category": size_tag,
                "rows": SIZE_ROWS.get(size_tag or "", 0),
                "license": tag_value(tags, "license:"),
                "url": f"https://huggingface.co/datasets/{entry['id']}",
            })
        if not cursor:
            break

    kept.sort(key=lambda d: -d["score"])
    print(f"  examined {seen:,}, kept {len(kept):,}\n", file=sys.stderr)

    width = max((len(d["id"]) for d in kept[:args.show]), default=10)
    print(f"{'#':>4}  {'dataset':<{width}}  {'downloads':>10}  {'likes':>6}  {'size':<12}  license")
    print("-" * (width + 52))
    for i, d in enumerate(kept[:args.show], 1):
        print(f"{i:>4}  {d['id']:<{width}}  {d['downloads']:>10,}  {d['likes']:>6,}  "
              f"{(d['size_category'] or '?'):<12}  {d['license'] or '?'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=1)
        print(f"\nWrote {len(kept):,} datasets to {args.out}")
        print(f"Train on a slice of it:\n"
              f"  python scripts/fetch_corpus.py --hf-from {args.out} --budget-mb 500")


if __name__ == "__main__":
    main()
