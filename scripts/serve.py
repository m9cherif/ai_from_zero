"""Serve a checkpoint over HTTP so a local web page can talk to it.

    python scripts/serve.py --checkpoint output/ckpt_900m/checkpoint_latest.pt
    open http://localhost:8000

Loads the model once and answers POST /api/generate with JSON. Inference needs
only the weights - no gradients, no optimizer state - so a model far too large
to train on a machine may still be served by it.

Bound to localhost by default. This is a development server with no auth, no
rate limiting and no sandboxing; do not expose it to a network you do not
control.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myai.core.logging import logger, LogLevel
from myai.inference import InferenceEngine

STATE = {}


def find_latest_checkpoint(directory: str) -> str:
    latest = os.path.join(directory, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest

    def step_of(path: str) -> int:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.replace("\\", "/"))
        return int(match.group(1)) if match else -1

    found = sorted(glob.glob(os.path.join(directory, "checkpoint_step_*.pt")), key=step_of)
    return found[-1] if found else ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # keep the console readable
        pass

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        # A page opened from file:// has a null origin, so allow any origin -
        # acceptable only because this binds to localhost.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self._send(204, b"")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "web", "local.html")
            if os.path.exists(page):
                self._send(200, open(page, "rb").read(), "text/html; charset=utf-8")
            else:
                self._send(404, json.dumps({"error": "web/local.html not found"}))
            return
        if self.path == "/api/info":
            self._send(200, json.dumps(STATE["info"]))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            prompt = (req.get("prompt") or "").strip()
            if not prompt:
                self._send(400, json.dumps({"error": "prompt is required"}))
                return

            started = time.time()
            text = STATE["engine"].generate(
                prompt,
                max_new_tokens=int(req.get("max_tokens", 60)),
                temperature=float(req.get("temperature", 0.8)),
                top_k=int(req.get("top_k", 40)),
                top_p=float(req.get("top_p", 0.95)),
                min_p=float(req.get("min_p", 0.05)),
                repetition_penalty=float(req.get("repetition_penalty", 1.15)),
            )
            elapsed = time.time() - started
            continuation = text[len(prompt):]
            self._send(200, json.dumps({
                "prompt": prompt,
                "completion": continuation,
                "seconds": round(elapsed, 2),
                "tokens_per_second": round(int(req.get("max_tokens", 60)) / max(elapsed, 1e-6), 2),
            }))
        except Exception as exc:                        # report, do not die
            self._send(500, json.dumps({"error": str(exc)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="output/checkpoints")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logger.set_level(LogLevel.INFO)
    checkpoint = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"No checkpoint found (looked in {args.checkpoint_dir})", file=sys.stderr)
        sys.exit(1)

    engine = InferenceEngine(device=args.device)
    engine.load_checkpoint(checkpoint, tokenizer_path=args.tokenizer)
    model = engine.model

    STATE["engine"] = engine
    STATE["info"] = {
        "checkpoint": checkpoint,
        "parameters": model.num_parameters(),
        "d_model": model.config.d_model,
        "n_layers": model.config.n_layers,
        "n_heads": model.config.n_heads,
        "n_kv_heads": model.config.n_kv_heads or model.config.n_heads,
        "vocab_size": model.config.vocab_size,
        "context": model.config.max_seq_len,
    }

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  {model.num_parameters():,} parameters from {checkpoint}")
    print(f"  serving on http://{args.host}:{args.port}  (Ctrl-C to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
