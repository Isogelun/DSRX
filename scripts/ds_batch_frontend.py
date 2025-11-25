import argparse
import json
import socket
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _log_local(message: str):
    print(f"[frontend] {message}")


def _log_backend(event: Dict):
    tag = event.get("job_id", "unknown")
    state = event.get("state") or event.get("event")
    extra = {k: v for k, v in event.items() if k not in {"event", "job_id"} and v is not None}
    print(f"[backend:{tag}] {state} {extra}".strip())


def _read_ds(ds_path: Path):
    with open(ds_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, list):
        content = [content]
    normalized = []
    for seg in content:
        seg = dict(seg)
        seg.setdefault("offset", 0.0)
        normalized.append(seg)
    return normalized


def _iter_events(payload: Dict, host: str, port: int) -> Iterable[Dict]:
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    with socket.create_connection((host, port)) as sock:
        fp = sock.makefile("rwb")
        fp.write(raw)
        fp.flush()
        for line in fp:
            if not line:
                break
            try:
                yield json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                continue


def parse_args():
    parser = argparse.ArgumentParser(description="Front-end for batched DiffSinger inference backend")
    parser.add_argument("ds_file", type=Path, help="Path to DS file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Backend host")
    parser.add_argument("--port", type=int, default=10086, help="Backend port")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to infer_output/<ds_name>.wav or .pt",
    )
    parser.add_argument(
        "--return-type",
        choices=["wav", "mel"],
        default="wav",
        help="Choose wav to save audio or mel to save stitched mel tensor",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size hint for backend")
    parser.add_argument("--seed", type=int, default=-1, help="Global seed fallback")
    parser.add_argument("--job-id", type=str, default=None, help="Optional job id to trace the run")
    parser.add_argument("--lang", type=str, default=None, help="Default language name for multilingual models")
    parser.add_argument("--spk", type=str, default=None, help="Speaker name or mix (e.g., a|b or a:0.5|b:0.5)")
    parser.add_argument("--task", choices=["acoustic", "variance"], default="acoustic", help="Inference task type")
    parser.add_argument(
        "--predict",
        type=str,
        default="all",
        help="Comma list for variance task, e.g., dur,pitch,energy (default all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ds_file = args.ds_file.resolve()
    if not ds_file.exists():
        raise SystemExit(f"DS file not found: {ds_file}")
    segments = _read_ds(ds_file)
    _log_local(f"Loaded {len(segments)} segments from {ds_file.name}")

    job_id = args.job_id or uuid.uuid4().hex
    if args.output:
        output_path = args.output.resolve()
    else:
        if args.task == "variance":
            suffix = ".ds"
        else:
            suffix = ".pt" if args.return_type == "mel" else ".wav"
        output_path = (Path("infer_output") / ds_file.stem).with_suffix(suffix)

    payload = {
        "action": "infer",
        "job_id": job_id,
        "segments": segments,
        "options": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "return_type": args.return_type,
            "output_path": str(output_path),
            "output_dir": str(output_path.parent),
            "lang": args.lang,
            "spk": args.spk,
            "task": args.task,
            "predict": args.predict,
        },
    }

    _log_local(f"Connecting to backend {args.host}:{args.port} with job {job_id}")
    final_event: Tuple[str, Dict] = ("", {})
    try:
        for event in _iter_events(payload, args.host, args.port):
            if event.get("event") == "result":
                final_event = ("result", event)
                _log_backend(event)
                break
            if event.get("event") == "error":
                final_event = ("error", event)
                _log_backend(event)
                break
            _log_backend(event)
    except ConnectionRefusedError:
        raise SystemExit(f"Cannot connect to backend at {args.host}:{args.port}. Is it running?")

    if final_event[0] == "result":
        result = final_event[1]
        mode = result.get("mode", args.return_type)
        if mode == "wav":
            _log_local(f"Inference finished. Audio saved to {result.get('path')}")
        else:
            _log_local(f"Inference finished. Mel tensor saved to {result.get('path')}")
    else:
        msg = final_event[1].get("message") if final_event[1] else "Unknown error"
        raise SystemExit(f"Inference failed: {msg}")


if __name__ == "__main__":
    sys.exit(main())
