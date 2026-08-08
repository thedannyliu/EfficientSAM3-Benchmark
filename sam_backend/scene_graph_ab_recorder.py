from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _edge_count(edges: Any) -> int:
    if isinstance(edges, list):
        return len(edges)
    if isinstance(edges, dict):
        return sum(len(value) if isinstance(value, list) else 1 for value in edges.values())
    return 0


def main() -> None:
    args = parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    args.output_dir.mkdir(parents=True, exist_ok=True)

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("scene_graph_ab_recorder")
            self.started_wall_ns = time.time_ns()
            self.first_camera_stamp_ns: int | None = None
            self.last_camera_stamp_ns: int | None = None
            self.next_sample_stamp_ns: int | None = None
            self.camera_count = 0
            self.detection_count = 0
            self.detection_frame_stamps: set[int] = set()
            self.graph_count = 0
            self.pose_count = 0
            self.sample_count = 0
            self.camera_wall_by_stamp: dict[int, int] = {}

            self.camera_file = (args.output_dir / "camera.jsonl").open("w", encoding="utf-8")
            self.detection_file = (args.output_dir / "detections.jsonl").open("w", encoding="utf-8")
            self.graph_file = (args.output_dir / "graph_counts.jsonl").open("w", encoding="utf-8")
            self.sample_file = (args.output_dir / "samples.jsonl").open("w", encoding="utf-8")
            self.frames_dir = args.output_dir / "sample_frames"
            self.frames_dir.mkdir(exist_ok=True)

            stream_qos = QoSProfile(depth=100)
            stream_qos.reliability = ReliabilityPolicy.RELIABLE
            graph_qos = QoSProfile(depth=10)
            graph_qos.reliability = ReliabilityPolicy.RELIABLE
            graph_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

            self.create_subscription(
                CompressedImage, args.camera_topic, self._camera, stream_qos
            )
            self.create_subscription(String, args.detections_topic, self._detection, stream_qos)
            self.create_subscription(String, args.graph_topic, self._graph, graph_qos)
            self.create_subscription(PoseStamped, args.pose_topic, self._pose, stream_qos)

        def _write(self, handle: Any, row: dict[str, Any]) -> None:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()

        def _camera(self, msg: Any) -> None:
            received_wall_ns = time.time_ns()
            source_stamp_ns = _stamp_ns(msg.header.stamp)
            self.camera_count += 1
            self.last_camera_stamp_ns = source_stamp_ns
            if self.first_camera_stamp_ns is None:
                self.first_camera_stamp_ns = source_stamp_ns
                self.next_sample_stamp_ns = source_stamp_ns

            self.camera_wall_by_stamp[source_stamp_ns] = received_wall_ns
            if len(self.camera_wall_by_stamp) > 10_000:
                oldest = next(iter(self.camera_wall_by_stamp))
                self.camera_wall_by_stamp.pop(oldest, None)

            self._write(
                self.camera_file,
                {
                    "source_stamp_ns": source_stamp_ns,
                    "received_wall_ns": received_wall_ns,
                    "sequence": self.camera_count,
                    "bytes": len(msg.data),
                    "format": msg.format,
                },
            )

            assert self.next_sample_stamp_ns is not None
            if source_stamp_ns < self.next_sample_stamp_ns:
                return
            frame_path = self.frames_dir / f"{source_stamp_ns}.jpg"
            frame_path.write_bytes(bytes(msg.data))
            self.sample_count += 1
            self._write(
                self.sample_file,
                {
                    "sample_index": self.sample_count - 1,
                    "source_stamp_ns": source_stamp_ns,
                    "received_wall_ns": received_wall_ns,
                    "image": str(frame_path),
                },
            )
            period_ns = int(args.sample_period * 1_000_000_000)
            while self.next_sample_stamp_ns <= source_stamp_ns:
                self.next_sample_stamp_ns += period_ns

        def _detection(self, msg: Any) -> None:
            received_wall_ns = time.time_ns()
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                payload = {"raw": msg.data, "decode_error": True}
            source_stamp_ns = int(payload.get("timestamp_ns", 0) or 0)
            camera_wall_ns = self.camera_wall_by_stamp.get(source_stamp_ns)
            self.detection_count += 1
            if source_stamp_ns:
                self.detection_frame_stamps.add(source_stamp_ns)
            self._write(
                self.detection_file,
                {
                    "received_wall_ns": received_wall_ns,
                    "source_stamp_ns": source_stamp_ns,
                    "camera_to_detection_ms": (
                        (received_wall_ns - camera_wall_ns) / 1_000_000.0
                        if camera_wall_ns is not None
                        else None
                    ),
                    "detection": payload,
                },
            )

        def _graph(self, msg: Any) -> None:
            received_wall_ns = time.time_ns()
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                payload = {}
            self.graph_count += 1
            nodes = payload.get("nodes", {})
            edges = payload.get("edges", {})
            self._write(
                self.graph_file,
                {
                    "received_wall_ns": received_wall_ns,
                    "sequence": self.graph_count,
                    "nodes": len(nodes) if isinstance(nodes, (dict, list)) else 0,
                    "edges": _edge_count(edges),
                    "graph_timestamp": payload.get("timestamp"),
                },
            )

        def _pose(self, _msg: Any) -> None:
            self.pose_count += 1

        def close(self) -> None:
            ended_wall_ns = time.time_ns()
            summary = {
                "label": args.label,
                "sample_period_seconds": args.sample_period,
                "started_wall_ns": self.started_wall_ns,
                "ended_wall_ns": ended_wall_ns,
                "wall_duration_seconds": (ended_wall_ns - self.started_wall_ns) / 1e9,
                "first_camera_stamp_ns": self.first_camera_stamp_ns,
                "last_camera_stamp_ns": self.last_camera_stamp_ns,
                "source_duration_seconds": (
                    (self.last_camera_stamp_ns - self.first_camera_stamp_ns) / 1e9
                    if self.first_camera_stamp_ns is not None
                    and self.last_camera_stamp_ns is not None
                    else 0.0
                ),
                "camera_messages": self.camera_count,
                "sampled_frames": self.sample_count,
                "pose_messages": self.pose_count,
                "detection_messages": self.detection_count,
                "detection_frames": len(self.detection_frame_stamps),
                "graph_messages": self.graph_count,
            }
            (args.output_dir / "recorder_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            for handle in (
                self.camera_file,
                self.detection_file,
                self.graph_file,
                self.sample_file,
            ):
                handle.close()

    rclpy.init()
    recorder = Recorder()

    def stop(*_: Any) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(recorder)
    finally:
        recorder.close()
        recorder.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record source-time-aligned camera, detection, and scene-graph A/B data."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sample-period", type=float, default=5.0)
    parser.add_argument("--camera-topic", default="/d435/color/image_raw_jpeg")
    parser.add_argument("--detections-topic", default="/scene_graph/detections_3d")
    parser.add_argument("--graph-topic", default="/scene_graph/online_sg")
    parser.add_argument("--pose-topic", default="/tracked_pose")
    return parser.parse_args()


if __name__ == "__main__":
    main()
