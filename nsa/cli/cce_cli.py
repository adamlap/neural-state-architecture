"""Unified Command-Line Interface for CCE (Phase CCE-8).

Usage:
  python -m nsa.cli.cce_cli status [--url http://localhost:8000]
  python -m nsa.cli.cce_cli inject --text "sensor event" --importance 0.8 [--url http://localhost:8000]
  python -m nsa.cli.cce_cli checkpoint [--id custom_id] [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict


def _http_get(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "CCE-CLI/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "CCE-CLI/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def cmd_status(args: argparse.Namespace) -> None:
    url = f"{args.url}/api/cce/state"
    try:
        data = _http_get(url)
        print("=" * 60)
        print("  CCE CONTINUOUS ENGINE STATUS")
        print("=" * 60)
        print(f"  Elapsed Wall-Clock Time : {data.get('elapsed_seconds', 0.0):.2f}s")
        print(f"  Integration Updates     : #{data.get('update_count', 0)}")
        print(f"  Epistemic Uncertainty   : {data.get('uncertainty', 0.0) * 100:.2f}%")
        print(f"  Working State Channels  : {data.get('working', [])}")
        print(f"  Self-Model State        : {data.get('self_state', [])}")
        print(f"  Active Goal Tensor      : {data.get('goal', [])}")
        print("=" * 60)
    except urllib.error.URLError as exc:
        print(f"[ERROR] Could not connect to CCE server at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_inject(args: argparse.Namespace) -> None:
    url = f"{args.url}/api/cce/sensor"
    payload = {
        "text": args.text,
        "source": args.source,
        "importance": args.importance,
    }
    try:
        res = _http_post(url, payload)
        print("[SUCCESS] Ingested sensory event into continuous substrate:")
        print(json.dumps(res, indent=2))
    except urllib.error.URLError as exc:
        print(f"[ERROR] Sensory injection failed at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_checkpoint(args: argparse.Namespace) -> None:
    url = f"{args.url}/api/cce/checkpoint"
    payload = {"checkpoint_id": args.id} if args.id else {}
    try:
        res = _http_post(url, payload)
        print("[SUCCESS] Saved atomic continuous state checkpoint:")
        print(json.dumps(res, indent=2))
    except urllib.error.URLError as exc:
        print(f"[ERROR] Checkpoint failed at {args.url}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="CCE Management CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of NSA Cognitive Server")
    sub = parser.add_subparsers(dest="command", required=True)

    # Status
    p_status = sub.add_parser("status", help="Inspect live CCE state")
    p_status.set_defaults(func=cmd_status)

    # Inject
    p_inject = sub.add_parser("inject", help="Inject asynchronous sensor/text input")
    p_inject.add_argument("--text", required=True, help="Input text or telemetry payload")
    p_inject.add_argument("--source", default="cce_cli", help="Provenance source identifier")
    p_inject.add_argument("--importance", type=float, default=0.7, help="Salience importance in [0, 1]")
    p_inject.set_defaults(func=cmd_inject)

    # Checkpoint
    p_chk = sub.add_parser("checkpoint", help="Trigger atomic state checkpoint")
    p_chk.add_argument("--id", default=None, help="Optional custom checkpoint identifier")
    p_chk.set_defaults(func=cmd_checkpoint)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
