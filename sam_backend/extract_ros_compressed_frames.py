from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main() -> None:
    args = parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "images"
    frames_dir.mkdir(exist_ok=True)

    class Extractor(Node):
        def __init__(self) -> None:
            super().__init__("compressed_frame_extractor")
            self.rows: list[dict[str, Any]] = []
            self.done = False
            self.started_wall_ns = time.time_ns()
            qos = QoSProfile(depth=100)
            qos.reliability = ReliabilityPolicy.RELIABLE
            self.create_subscription(CompressedImage, args.topic, self._frame, qos)

        def _frame(self, message: Any) -> None:
            stamp_ns = _stamp_ns(message.header.stamp)
            if stamp_ns < args.start_stamp_ns or len(self.rows) >= args.count:
                return
            payload = bytes(message.data)
            frame_index = len(self.rows)
            filename = f"{frame_index:05d}-{stamp_ns}.jpg"
            path = frames_dir / filename
            path.write_bytes(payload)
            self.rows.append(
                {
                    "frame_index": frame_index,
                    "source_stamp_ns": stamp_ns,
                    "image": filename,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            if len(self.rows) == args.count:
                self.done = True

        def close(self) -> None:
            manifest = args.output_dir / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in self.rows),
                encoding="utf-8",
            )
            summary = {
                "topic": args.topic,
                "requested_start_stamp_ns": args.start_stamp_ns,
                "requested_frames": args.count,
                "extracted_frames": len(self.rows),
                "first_source_stamp_ns": self.rows[0]["source_stamp_ns"] if self.rows else None,
                "last_source_stamp_ns": self.rows[-1]["source_stamp_ns"] if self.rows else None,
                "source_duration_seconds": (
                    (self.rows[-1]["source_stamp_ns"] - self.rows[0]["source_stamp_ns"]) / 1e9
                    if len(self.rows) > 1
                    else 0.0
                ),
                "wall_duration_seconds": (time.time_ns() - self.started_wall_ns) / 1e9,
            }
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )

    rclpy.init()
    extractor = Extractor()

    def stop(*_: Any) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while rclpy.ok() and not extractor.done:
            rclpy.spin_once(extractor, timeout_sec=0.1)
    finally:
        extractor.close()
        extractor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if len(extractor.rows) != args.count:
        raise RuntimeError(f"extracted {len(extractor.rows)} frames, expected {args.count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract consecutive ROS CompressedImage messages by source timestamp."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-stamp-ns", type=int, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--topic", default="/d435/color/image_raw_jpeg")
    return parser.parse_args()


if __name__ == "__main__":
    main()
