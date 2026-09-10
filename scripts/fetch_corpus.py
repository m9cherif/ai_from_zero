"""Download and prepare a real training corpus.

    python scripts/fetch_corpus.py

Pulls public-domain literature (Project Gutenberg via the GITenberg mirror),
curated Wikipedia (WikiText-2) and reference prose, strips the Gutenberg
boilerplate, and splits the result into data/train and data/val. Validation
files are held out whole, so no document appears on both sides of the split.

Everything comes from raw.githubusercontent.com, which is reachable from
restricted network environments where huggingface.co and gutenberg.org are not.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys

RAW = "https://raw.githubusercontent.com"

# Curated Wikipedia and reference prose.
SOURCES = [
    ("wikitext2_train.txt", f"{RAW}/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"),
    ("wikitext2_valid.txt", f"{RAW}/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"),
    ("wikitext2_test.txt", f"{RAW}/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"),
    ("norvig_big.txt", f"{RAW}/dscape/spell/master/test/resources/big.txt"),
    # TinyShakespeare is deliberately absent: it is an excerpt of the Complete
    # Works (Gutenberg 100), which is a validation candidate. Including both put
    # 20% of the validation document into the training set and made validation
    # loss read better than it was.
]

# Project Gutenberg: GITenberg mirrors each book at <Slug>_<id>/master/<id>.txt
GUTENBERG = [
    ("Moby-Dick--Or-The-Whale", 2701), ("Pride-and-Prejudice", 1342),
    ("Alice-s-Adventures-in-Wonderland", 11), ("The-Adventures-of-Sherlock-Holmes", 1661),
    ("A-Tale-of-Two-Cities", 98), ("Great-Expectations", 1400),
    ("The-Picture-of-Dorian-Gray", 174), ("Dracula", 345),
    ("Adventures-of-Huckleberry-Finn", 76), ("War-and-Peace", 2600),
    ("The-Prince", 1232), ("Metamorphosis", 5200),
    ("The-Count-of-Monte-Cristo", 1184), ("Wuthering-Heights", 768),
    ("The-Time-Machine", 35), ("The-War-of-the-Worlds", 36),
    ("The-Iliad", 6130), ("Don-Quixote", 996),
    ("Anna-Karenina", 1399), ("Crime-and-Punishment", 2554),
    ("The-Brothers-Karamazov", 28054), ("Ulysses", 4300),
    ("The-Republic", 1497), ("Beyond-Good-and-Evil", 4363),
    ("Relativity-The-Special-and-General-Theory", 5001),
    ("The-Souls-of-Black-Folk", 408),
    ("Walden-and-On-The-Duty-Of-Civil-Disobedience", 205),
    ("The-Art-of-War", 132), ("Grimms-Fairy-Tales", 2591),
    ("The-Complete-Works-of-William-Shakespeare", 100),
    ("Treasure-Island", 120), ("The-Call-of-the-Wild", 215),
    ("Heart-of-Darkness", 219), ("The-Scarlet-Letter", 25344),
    # Second tranche - roughly doubles the corpus.
    ("The-Adventures-of-Tom-Sawyer", 74), ("Emma", 158),
    ("Sense-and-Sensibility", 161), ("Mansfield-Park", 141),
    ("Persuasion", 105), ("Northanger-Abbey", 121),
    ("Peter-Pan", 16), ("The-Jungle-Book", 236),
    ("Dubliners", 2814), ("A-Portrait-of-the-Artist-as-a-Young-Man", 4217),
    ("The-Problems-of-Philosophy", 5827), ("Leviathan", 3207),
    ("Thus-Spake-Zarathustra-A-Book-for-All-and-None", 1998),
    ("The-Strange-Case-of-Dr-Jekyll-and-Mr-Hyde", 43),
    ("The-Turn-of-the-Screw", 209), ("Little-Women", 514),
    ("The-Wonderful-Wizard-of-Oz", 55), ("The-Secret-Garden", 113),
    ("The-Wind-in-the-Willows", 289), ("Black-Beauty", 271),
    ("Anne-of-Green-Gables", 45), ("Oliver-Twist", 730),
    ("David-Copperfield", 766), ("Bleak-House", 1023),
    ("A-Study-in-Scarlet", 244), ("The-Sign-of-the-Four", 2097),
    ("The-Return-of-Sherlock-Holmes", 108), ("The-Memoirs-of-Sherlock-Holmes", 834),
    ("The-Hound-of-the-Baskervilles", 2852),
    ("Twenty-Thousand-Leagues-under-the-Sea", 164),
    ("Around-the-World-in-Eighty-Days", 103),
    ("A-Journey-to-the-Centre-of-the-Earth", 18857),
    ("The-Three-Musketeers", 1257), ("Robinson-Crusoe", 521),
    ("Gulliver-s-Travels", 829), ("Vanity-Fair", 599),
    ("Middlemarch", 145), ("Silas-Marner", 550),
    ("The-Mill-on-the-Floss", 6688),
    ("Tess-of-the-d-Urbervilles-A-Pure-Woman", 110),
    ("Far-from-the-Madding-Crowd", 27),
    ("The-Awakening-and-Selected-Short-Stories", 160),
    ("Ethan-Frome", 4517), ("The-Age-of-Innocence", 541),
    ("Sister-Carrie", 233), ("The-Red-Badge-of-Courage", 73),
    ("White-Fang", 910), ("The-Sea-Wolf", 1074), ("Martin-Eden", 1056),
    ("Nostromo-A-Tale-of-the-Seaboard", 2021), ("Lord-Jim", 5658),
    ("The-Moonstone", 155), ("The-Woman-in-White", 583),
    ("Uncle-Tom-s-Cabin", 203),
    ("Narrative-of-the-Life-of-Frederick-Douglass-an-American-Slave", 23),
    ("Common-Sense", 147), ("The-Federalist-Papers", 1404),
    ("Second-Treatise-of-Goverment", 7370),
    ("An-Enquiry-Concerning-Human-Understanding", 9662),
    ("The-Critique-of-Pure-Reason", 4280), ("Utopia", 2130),
    ("The-Confessions-of-St-Augustine", 3296),
    ("The-Imitation-of-Christ", 1653), ("Aesop-s-Fables", 21),
    ("The-Arabian-Nights-Entertainments", 5667),
    ("Fairy-Tales-of-Hans-Christian-Andersen", 27200),
    ("The-Happy-Prince-and-Other-Tales", 902),
    ("The-Importance-of-Being-Earnest-A-Trivial-Comedy-for-Serious-People", 844),
    ("A-Doll-s-House-a-play", 2542),
    ("The-Tragical-History-of-Doctor-Faustus", 811),
    ("Paradise-Lost", 26), ("The-Divine-Comedy-by-Dante-Illustrated", 8800),
    ("Leaves-of-Grass", 1322), ("Meditations", 2680),
    ("An-Inquiry-into-the-Nature-and-Causes-of-the-Wealth-of-Nations", 3300),
    ("Candide", 19942), ("The-Metamorphoses-of-Ovid", 21765), ("Faust", 14591),
]

START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


def strip_boilerplate(text: str) -> str:
    """Remove the Gutenberg licence header/footer and pagination artefacts."""
    match = START.search(text)
    if match:
        text = text[match.end():]
    match = END.search(text)
    if match:
        text = text[: match.start()]
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"(?m)^\s*Produced by .*$", "", text)
    return text.strip() + "\n"


def fetch(dest_dir, name, url):
    dest = os.path.join(dest_dir, name)
    result = subprocess.run(
        ["curl", "-sL", "--max-time", "120", "-o", dest, "-w", "%{http_code}", url],
        capture_output=True, text=True,
    )
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if result.stdout.strip() != "200" or size < 5000:
        if os.path.exists(dest):
            os.remove(dest)
        return name, False
    return name, True


HF_BASE = "https://huggingface.co"


def hf_list_text_files(repo: str, revision: str = "main"):
    """Text-ish files in a dataset repo, via the public tree API."""
    url = f"{HF_BASE}/api/datasets/{repo}/tree/{revision}?recursive=1"
    result = subprocess.run(["curl", "-sL", "--max-time", "60", url],
                            capture_output=True, text=True)
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []
    keep = (".txt", ".jsonl", ".json", ".parquet")
    return [(e["path"], int(e.get("size") or 0)) for e in entries
            if e.get("type") == "file" and e.get("path", "").endswith(keep)]


def hf_resolve_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF_BASE}/datasets/{repo}/resolve/{revision}/{path}"


def extract_text_to(src: str, dst: str, limit_bytes: int = 0) -> int:
    """Stream text out of .txt/.jsonl/.parquet into dst. Returns bytes written.

    Nothing is accumulated in memory: a 2 GB parquet expands to several GB of
    text, and building that as one Python string is what exhausts a 30 GB
    machine. Rows are written as they are read, and the writer stops once
    limit_bytes is reached.
    """
    written = 0

    def emit(handle, text):
        nonlocal written
        if not text:
            return False
        chunk = text if text.endswith("\n") else text + "\n"
        handle.write(chunk)
        written += len(chunk)
        return bool(limit_bytes) and written >= limit_bytes

    if src.endswith(".txt"):
        with open(src, encoding="utf-8", errors="replace") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                if emit(fout, line.rstrip("\n")):
                    break
        return written

    if src.endswith((".jsonl", ".json")):
        with open(src, encoding="utf-8", errors="replace") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = pick_text(row)
                if text and emit(fout, text):
                    break
        return written

    if src.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print("  parquet needs pyarrow: pip install pyarrow", file=sys.stderr)
            return 0
        parquet = pq.ParquetFile(src)
        column = None
        for candidate in TEXT_COLUMNS:
            if candidate in parquet.schema_arrow.names:
                column = candidate
                break
        if column is None:
            strings = [n for n, t in zip(parquet.schema_arrow.names,
                                         parquet.schema_arrow.types)
                       if str(t) in ("string", "large_string")]
            if not strings:
                return 0
            column = strings[0]

        with open(dst, "w", encoding="utf-8") as fout:
            stop = False
            # Row groups, not the whole file: peak memory is one batch.
            for batch in parquet.iter_batches(batch_size=1000, columns=[column]):
                for value in batch.column(0).to_pylist():
                    if value and emit(fout, str(value)):
                        stop = True
                        break
                if stop:
                    break
        return written

    return 0


TEXT_COLUMNS = ("text", "content", "raw_content", "story", "document", "article")


def pick_text(row):
    """The text field of a JSON row, by name where possible."""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""
    for key in TEXT_COLUMNS:
        if isinstance(row.get(key), str):
            return row[key]
    longest = max((v for v in row.values() if isinstance(v, str)), key=len, default="")
    return longest if len(longest) > 40 else ""


def fetch_hf(dest_dir, specs, urls, max_files, workers, text_cap=0):
    """Download HuggingFace dataset files into dest_dir as .txt.

    Each spec is 'owner/dataset' or 'owner/dataset:path/inside/repo'. Without a
    path, the repo's text files are listed and the first --hf-max-files taken.
    """
    jobs = []
    for spec in specs or []:
        repo, _, inner = spec.partition(":")
        revision = "main"
        if "@" in repo:
            repo, _, revision = repo.partition("@")
        if inner:
            paths = [inner]
        else:
            paths = [p for p, _ in hf_list_text_files(repo, revision)[:max_files]]
        if not paths:
            print(f"  no text files found in {repo}", file=sys.stderr)
        for path in paths:
            # Full inner path in the name: gsm8k ships main/test and
            # socratic/test, and a basename-only name makes two jobs collide
            # on one file - which crashed the second one mid-download.
            flat = path.replace("/", "-")
            name = f"hf_{repo.replace('/', '_')}_{flat}"
            jobs.append((name, hf_resolve_url(repo, path, revision)))

    for url in urls or []:
        jobs.append((f"hf_{os.path.basename(url.split('?')[0])}", url))

    if not jobs:
        return 0

    print(f"Fetching {len(jobs)} HuggingFace file(s)...")
    kept = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for name, ok in pool.map(lambda a: fetch(dest_dir, *a), jobs):
            if not ok:
                print(f"  unavailable: {name}", file=sys.stderr)
                continue
            raw = os.path.join(dest_dir, name)
            if not os.path.exists(raw):
                continue
            # Land it as .txt so the rest of the pipeline treats it uniformly.
            out_path = os.path.join(dest_dir, os.path.splitext(name)[0] + ".txt")
            written = extract_text_to(raw, out_path, limit_bytes=text_cap)
            try:
                os.remove(raw)
            except FileNotFoundError:
                pass
            if written < 5000:
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue
            kept += 1
    print(f"  kept {kept} file(s)")
    return kept


def select_within_budget(catalogue, budget_bytes, max_per_dataset,
                         min_bytes=1_000_000, max_file_bytes=250_000_000,
                         seed=None, revision="main"):
    """Walk a ranked catalogue and pick files until the byte budget is met.

    Sizes come from the Hub's tree listing, so the budget is enforced before
    anything is downloaded rather than discovered afterwards.
    """
    import random
    rng = random.Random(seed)
    chosen, total = [], 0
    # Rank order still leads, but the head of the catalogue is shuffled so
    # successive runs do not all start from the same dataset.
    catalogue = list(catalogue)
    head = catalogue[:max(8, max_per_dataset * 4)]
    rng.shuffle(head)
    catalogue = head + catalogue[len(head):]
    for entry in catalogue:
        if total >= budget_bytes:
            break
        repo = entry["id"] if isinstance(entry, dict) else str(entry)
        try:
            files = hf_list_text_files(repo, revision)
        except Exception:
            continue
        if not files:
            continue
        # Keep only moderate files, then shuffle. One 2 GB shard would swallow
        # the whole budget, and taking the largest deterministically means every
        # run trains on the same shard of the same dataset.
        files = [(p, sz) for p, sz in files if min_bytes <= sz <= max_file_bytes]
        rng.shuffle(files)
        taken = 0
        for path, size in files:
            if taken >= max_per_dataset or total >= budget_bytes:
                break
            if total + size > budget_bytes:
                continue
            chosen.append((repo, path, size))
            total += size
            taken += 1
    return chosen, total


def check_contamination(train_dir: str, val_dir: str, samples: int = 40) -> float:
    """Fraction of sampled validation lines that also appear in the training set.

    A held-out document is only held out if its text is absent from training.
    Public-domain corpora make this easy to get wrong - anthologies, excerpt
    collections and "complete works" editions overlap constantly.
    """
    import random

    val_text = "".join(
        open(os.path.join(val_dir, n), encoding="utf-8", errors="replace").read()
        for n in sorted(os.listdir(val_dir))
    )
    train_text = "".join(
        open(os.path.join(train_dir, n), encoding="utf-8", errors="replace").read()
        for n in sorted(os.listdir(train_dir))
    )
    # Both sides get the same whitespace normalisation, otherwise a probe taken
    # across a line break can never match however much text is shared.
    train_text = " ".join(train_text.split())

    # Sample at random character offsets rather than by line. Line-based
    # sampling with a length filter silently misses overlap in sources whose
    # lines are short - play dialogue, verse - which is exactly where
    # public-domain corpora overlap.
    span = 80
    if len(val_text) <= span:
        return 0.0
    random.seed(0)
    hits = 0
    for _ in range(samples):
        start = random.randrange(0, len(val_text) - span)
        probe = " ".join(val_text[start:start + span].split())
        if probe and probe in train_text:
            hits += 1
    return hits / samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data", help="Corpus root")
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--hf", nargs="+", default=None, metavar="REPO[:FILE]",
        help="HuggingFace datasets, e.g. roneneldan/TinyStories or "
             "wikitext:wikitext-103-raw-v1/train.parquet. Pin a revision with "
             "owner/name@revision.",
    )
    parser.add_argument(
        "--hf-url", nargs="+", default=None, metavar="URL",
        help="Direct HuggingFace resolve URLs.",
    )
    parser.add_argument(
        "--hf-max-files", type=int, default=4,
        help="Files to take per dataset when no path is given.",
    )
    parser.add_argument(
        "--hf-from", default=None, metavar="CATALOGUE.json",
        help="A catalogue from scripts/list_datasets.py. Combined with "
             "--budget-mb, files are taken in rank order until the budget "
             "is met.",
    )
    parser.add_argument(
        "--budget-mb", type=float, default=500.0,
        help="How much text to download from --hf-from, in megabytes. "
             "Roughly 3.7 chars per token, so 500 MB is about 135M tokens.",
    )
    parser.add_argument(
        "--max-file-mb", type=float, default=250.0,
        help="Skip files larger than this. One multi-gigabyte shard would take "
             "the whole budget and expand to more text than RAM holds.",
    )
    parser.add_argument(
        "--text-cap-mb", type=float, default=400.0,
        help="Stop extracting a single file after this much text.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fixes which shards are chosen. Omit for a different mix each run.",
    )
    parser.add_argument(
        "--min-file-mb", type=float, default=1.0,
        help="Ignore files smaller than this - benchmark shards and prompt "
             "fragments are text but not prose.",
    )
    parser.add_argument(
        "--max-per-dataset", type=int, default=3,
        help="Files taken from any one dataset, so a budget spans sources.",
    )
    parser.add_argument(
        "--skip-gutenberg", action="store_true",
        help="Use only the HuggingFace sources.",
    )
    args = parser.parse_args()

    staging = os.path.join(args.output, "_raw")
    train_dir = os.path.join(args.output, "train")
    val_dir = os.path.join(args.output, "val")
    for path in (staging, train_dir, val_dir):
        os.makedirs(path, exist_ok=True)

    hf_specs = list(args.hf or [])
    if args.hf_from:
        catalogue = json.load(open(args.hf_from, encoding="utf-8"))
        budget = int(args.budget_mb * 1e6)
        picked, total = select_within_budget(
            catalogue, budget, args.max_per_dataset,
            min_bytes=int(args.min_file_mb * 1e6),
            max_file_bytes=int(args.max_file_mb * 1e6),
            seed=args.seed)
        print(f"Budget {args.budget_mb:.0f} MB -> {len(picked)} files, "
              f"{total/1e6:.1f} MB, ~{total/3.7/1e6:.0f}M tokens")
        for repo, path, size in picked:
            print(f"    {size/1e6:8.1f} MB  {repo}/{path}")
        hf_specs += [f"{repo}:{path}" for repo, path, _ in picked]

    if hf_specs or args.hf_url:
        fetch_hf(staging, hf_specs, args.hf_url, args.hf_max_files, args.workers,
                 text_cap=int(args.text_cap_mb * 1e6))

    jobs = [] if args.skip_gutenberg else list(SOURCES)
    if not args.skip_gutenberg:
        for slug, gid in GUTENBERG:
            jobs.append((f"gutenberg_{gid}.txt", f"{RAW}/GITenberg/{slug}_{gid}/master/{gid}.txt"))

    print(f"Fetching {len(jobs)} files...")
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, ok in pool.map(lambda a: fetch(staging, *a), jobs):
            if not ok:
                failed.append(name)
    if failed:
        print(f"  {len(failed)} unavailable (skipped): {', '.join(sorted(failed))}")

    documents = []
    raw_bytes = 0
    for path in sorted(os.listdir(staging)):
        full = os.path.join(staging, path)
        raw = open(full, encoding="utf-8", errors="replace").read()
        raw_bytes += len(raw)
        text = strip_boilerplate(raw)
        if len(text) >= 5000:
            documents.append((path, text))
        os.remove(full)
    os.rmdir(staging)

    if not documents:
        print("No corpus could be downloaded - check network access.", file=sys.stderr)
        sys.exit(1)

    clean_bytes = sum(len(t) for _, t in documents)
    val_budget = clean_bytes * args.val_fraction
    val_bytes = n_val = 0

    # Prefer holding out whole documents, so train and val never share text.
    # Largest first, but only documents that fit the budget - a single document
    # bigger than the whole budget would otherwise take half the corpus with it.
    remaining = val_budget
    held = set()
    for i in sorted(range(len(documents)), key=lambda i: -len(documents[i][1])):
        if remaining <= 0:
            break
        size = len(documents[i][1])
        if size <= remaining * 1.5:          # modest overshoot is fine
            held.add(i)
            remaining -= size

    if held:
        for i, (name, text) in enumerate(documents):
            target = val_dir if i in held else train_dir
            open(os.path.join(target, name), "w", encoding="utf-8").write(text)
            if i in held:
                val_bytes += len(text)
                n_val += 1
    else:
        # Every document is larger than the budget - the usual case for a
        # HuggingFace dataset shipped as one big file. Split the smallest one
        # by offset: still disjoint, just not on a document boundary.
        smallest = min(range(len(documents)), key=lambda i: len(documents[i][1]))
        for i, (name, text) in enumerate(documents):
            if i != smallest:
                open(os.path.join(train_dir, name), "w", encoding="utf-8").write(text)
                continue
            cut = len(text) - max(int(len(text) * args.val_fraction), 1)
            open(os.path.join(train_dir, name), "w", encoding="utf-8").write(text[:cut])
            open(os.path.join(val_dir, name), "w", encoding="utf-8").write(text[cut:])
            val_bytes, n_val = len(text) - cut, 1

    print(f"\n  downloaded   {raw_bytes/1e6:7.1f} MB ({len(documents)} documents)")
    print(f"  cleaned      {clean_bytes/1e6:7.1f} MB "
          f"(stripped {(raw_bytes-clean_bytes)/1e6:.1f} MB of boilerplate)")
    print(f"  train        {(clean_bytes-val_bytes)/1e6:7.1f} MB -> {train_dir}")
    print(f"  val          {val_bytes/1e6:7.1f} MB -> {val_dir} ({n_val} held-out documents)")
    overlap = check_contamination(train_dir, val_dir)
    if overlap > 0.02:
        print(f"\n  WARNING: {overlap:.0%} of sampled validation lines also appear in "
              f"training.\n  Validation loss will read better than it is. Remove the "
              f"overlapping source or hold out a different document.")
    else:
        print(f"  leakage check {overlap:.0%} of sampled val lines found in train")

    print(f"\nNext:  python scripts/build_tokenizer.py --type bpe --vocab-size 4096 "
          f"--data '{train_dir}/*.txt'")


if __name__ == "__main__":
    main()
