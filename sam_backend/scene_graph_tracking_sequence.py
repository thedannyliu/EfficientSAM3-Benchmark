from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


class RuntimeClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self, path: str, data: bytes | None = None, content_type: str | None = None
    ) -> bytes:
        headers = {"Content-Type": content_type} if content_type else {}
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def wait_for_masks(self, input_sequence: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                payload = self.request("/masks.npz")
            except urllib.error.HTTPError as error:
                if error.code != 503:
                    raise
                time.sleep(0.01)
                continue
            snapshot = _decode_masks(payload)
            if snapshot["input_sequence"] == input_sequence:
                return snapshot
            time.sleep(0.01)
        raise TimeoutError(f"timed out waiting for input_sequence={input_sequence}")


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = args.output_dir / "predictions"
    predictions.mkdir(exist_ok=True)
    client = RuntimeClient(args.runtime_url, args.timeout)

    client.request("/reset", b"", "application/octet-stream")
    client.request(
        "/prompt",
        json.dumps({"text": ",".join(args.prompt)}, separators=(",", ":")).encode(),
        "application/json",
    )

    profile_path = args.output_dir / "profile.jsonl"
    sequence_started = time.perf_counter()
    with profile_path.open("w", encoding="utf-8") as profile_file:
        for row in rows:
            image_path = args.image_dir / Path(row["image"]).name
            payload = image_path.read_bytes()
            started_wall_ns = time.time_ns()
            started = time.perf_counter()
            frame_response = json.loads(
                client.request("/frame.jpg", payload, "image/jpeg")
            )
            submitted = time.perf_counter()
            input_sequence = int(frame_response["input_sequence"])
            snapshot = client.wait_for_masks(input_sequence)
            masks_ready = time.perf_counter()
            status = json.loads(client.request("/status.json"))
            completed = time.perf_counter()
            output_path = predictions / f"{int(row['frame_index']):05d}.npz"
            _save_snapshot(output_path, snapshot)
            profile_file.write(
                json.dumps(
                    {
                        "frame_index": int(row["frame_index"]),
                        "source_stamp_ns": int(row["source_stamp_ns"]),
                        "image": str(image_path),
                        "prediction": str(output_path),
                        "input_sequence": input_sequence,
                        "runtime_frame": snapshot["frame"],
                        "mask_count": int(len(snapshot["masks"])),
                        "active_mask_count": int(np.logical_not(snapshot["lost"]).sum()),
                        "submit_ms": (submitted - started) * 1000.0,
                        "wait_ms": (masks_ready - submitted) * 1000.0,
                        "status_ms": (completed - masks_ready) * 1000.0,
                        "total_ms": (completed - started) * 1000.0,
                        "started_wall_ns": started_wall_ns,
                        "ended_wall_ns": time.time_ns(),
                        "status": status,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            profile_file.flush()

    summary = {
        "frames": len(rows),
        "prompts": args.prompt,
        "sequence_wall_seconds": time.perf_counter() - sequence_started,
        "runtime_url": args.runtime_url,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _decode_masks(payload: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        shape = tuple(int(value) for value in archive["mask_shape"])
        values = np.unpackbits(
            archive["masks_packed"], bitorder=str(archive["bitorder"].item())
        )[: int(np.prod(shape))]
        return {
            "schema_version": int(archive["schema_version"]),
            "frame": int(archive["frame"]),
            "input_sequence": int(archive["input_sequence"]),
            "masks": values.astype(bool).reshape(shape),
            "ids": archive["ids"].astype(np.int64),
            "scores": archive["scores"].astype(np.float32),
            "lost": archive["lost"].astype(bool),
            "labels": archive["labels"].astype(str),
        }


def _save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    masks = np.asarray(snapshot["masks"], dtype=bool)
    np.savez(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        mask_shape=np.asarray(masks.shape, dtype=np.int64),
        masks_packed=np.packbits(masks.reshape(-1), bitorder="little"),
        bitorder=np.asarray("little"),
        ids=np.asarray(snapshot["ids"], dtype=np.int64),
        labels=np.asarray(snapshot["labels"], dtype=np.str_),
        scores=np.asarray(snapshot["scores"], dtype=np.float32),
        lost=np.asarray(snapshot["lost"], dtype=np.bool_),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one prompt initialization followed by an ordered frame sequence."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8767")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
