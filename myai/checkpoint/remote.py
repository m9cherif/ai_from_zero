"""Load checkpoints from a URL as well as from disk.

A trained checkpoint has to live somewhere durable. Ephemeral machines - Kaggle
and Colab runtimes, CI containers, cloud sandboxes - lose local files when the
session ends, so the model needs a permanent address that every script can point
at instead of a path that exists on one box.

Accepted forms:

    /path/to/checkpoint.pt                      local file, returned as-is
    https://host/path/checkpoint.pt             any HTTPS URL
    hf://owner/repo/checkpoint.pt               HuggingFace model repo
    hf://owner/repo@revision/checkpoint.pt      pinned to a revision

Downloads are cached, so the second call is free.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from ..core.logging import logger

HF_BASE = "https://huggingface.co"
DEFAULT_CACHE = os.environ.get("MYAI_CACHE", os.path.expanduser("~/.cache/myai"))


def is_remote(path: str) -> bool:
    return path.startswith(("http://", "https://", "hf://"))


def hf_to_url(spec: str) -> str:
    """hf://owner/repo[@revision]/file -> a resolve URL."""
    body = spec[len("hf://"):]
    parts = body.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Malformed hf:// spec {spec!r}; expected hf://owner/repo/file.pt"
        )
    owner, repo = parts[0], parts[1]
    filename = "/".join(parts[2:])
    revision = "main"
    if "@" in repo:
        repo, _, revision = repo.partition("@")
    return f"{HF_BASE}/{owner}/{repo}/resolve/{revision}/{filename}"


def resolve_checkpoint(
    path_or_url: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Return a local path for ``path_or_url``, downloading it if remote."""
    if not is_remote(path_or_url):
        return path_or_url

    url = hf_to_url(path_or_url) if path_or_url.startswith("hf://") else path_or_url

    cache = Path(cache_dir or DEFAULT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    # Key on the whole URL so two files with the same basename cannot collide.
    import hashlib
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    name = os.path.basename(url.split("?")[0]) or "checkpoint.pt"
    dest = cache / f"{digest}_{name}"

    if dest.exists() and not force:
        logger.info(f"Using cached checkpoint {dest}")
        return str(dest)

    logger.info(f"Downloading checkpoint from {url}")
    partial = dest.with_suffix(dest.suffix + ".part")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    command = ["curl", "-fL", "--retry", "3", "-o", str(partial), "-w", "%{http_code}"]
    if token and "huggingface.co" in url:
        # Private repos need the token; it stays in the header, never the URL.
        command += ["-H", f"Authorization: Bearer {token}"]
    command.append(url)

    result = subprocess.run(command, capture_output=True, text=True)
    code = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "000"
    if result.returncode != 0 or not code.startswith("2") or not partial.exists():
        if partial.exists():
            partial.unlink()
        raise RuntimeError(
            f"Could not download {url} (HTTP {code}). For a private repo set "
            f"HF_TOKEN."
        )

    partial.rename(dest)
    logger.info(f"Cached to {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return str(dest)
