from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


FIELDS = [
    "frame_index",
    "latency_ms",
    "callback_total_ms",
    "end_to_end_ms",
    "image_encoder_ms",
    "text_encoder_ms",
    "grounding_ms",
    "other_ms",
    "mask_count",
    "box_count",
    "score_max",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "params_total",
    "params_image_encoder",
    "params_text_encoder",
    "stamp_sec",
    "stamp_nanosec",
    "frame_id",
]


SUMMARY_FIELDS = [
    "frames",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_latency_fps",
    "mean_callback_total_ms",
    "p95_callback_total_ms",
    "mean_end_to_end_ms",
    "p95_end_to_end_ms",
    "mean_image_encoder_ms",
    "mean_text_encoder_ms",
    "mean_grounding_ms",
    "mean_mask_count",
    "mean_score_max",
]


class ResultRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("result_recorder_node")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("csv_output", "results/ros/ros_results.csv")
        self.declare_parameter("summary_output", "results/ros/ros_summary.csv")
        self.declare_parameter("max_messages", 0)

        self.csv_output = Path(self.get_parameter("csv_output").value)
        self.summary_output = Path(self.get_parameter("summary_output").value)
        self.max_messages = int(self.get_parameter("max_messages").value)
        self.rows: list[dict[str, Any]] = []
        self.summary_written = False

        self.csv_output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.csv_output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()

        result_topic = self.get_parameter("result_topic").value
        self.subscription = self.create_subscription(String, result_topic, self.on_result, 100)
        self.get_logger().info(f"recording {result_topic} to {self.csv_output}")

    def on_result(self, msg: String) -> None:
        data = json.loads(msg.data)
        stamp = data.get("stamp", {})
        row = {
            "frame_index": data.get("frame_index", len(self.rows)),
            "latency_ms": data.get("latency_ms"),
            "callback_total_ms": data.get("callback_total_ms"),
            "end_to_end_ms": data.get("end_to_end_ms"),
            "image_encoder_ms": data.get("image_encoder_ms"),
            "text_encoder_ms": data.get("text_encoder_ms"),
            "grounding_ms": data.get("grounding_ms"),
            "other_ms": data.get("other_ms"),
            "mask_count": data.get("mask_count"),
            "box_count": data.get("box_count"),
            "score_max": data.get("score_max"),
            "cuda_allocated_mb": data.get("cuda_allocated_mb"),
            "cuda_reserved_mb": data.get("cuda_reserved_mb"),
            "cuda_peak_allocated_mb": data.get("cuda_peak_allocated_mb"),
            "cuda_peak_reserved_mb": data.get("cuda_peak_reserved_mb"),
            "params_total": data.get("params_total"),
            "params_image_encoder": data.get("params_image_encoder"),
            "params_text_encoder": data.get("params_text_encoder"),
            "stamp_sec": stamp.get("sec"),
            "stamp_nanosec": stamp.get("nanosec"),
            "frame_id": data.get("frame_id"),
        }
        self.rows.append(row)
        self.writer.writerow(row)
        self.file.flush()
        if self.max_messages > 0 and len(self.rows) >= self.max_messages:
            self.write_summary()
            raise SystemExit

    def write_summary(self) -> None:
        self.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_rows(self.rows)
        with self.summary_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(summary)
        self.summary_written = True
        self.get_logger().info(f"wrote summary to {self.summary_output}")

    def destroy_node(self) -> bool:
        if not self.summary_written:
            self.write_summary()
        self.file.close()
        return super().destroy_node()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latency = _values(rows, "latency_ms")
    callback = _values(rows, "callback_total_ms")
    end_to_end = _values(rows, "end_to_end_ms")
    image = _values(rows, "image_encoder_ms")
    text = _values(rows, "text_encoder_ms")
    grounding = _values(rows, "grounding_ms")
    masks = _values(rows, "mask_count")
    scores = _values(rows, "score_max")
    mean_latency = mean(latency) if latency else None
    return {
        "frames": len(rows),
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": _percentile(latency, 0.50),
        "p95_latency_ms": _percentile(latency, 0.95),
        "mean_latency_fps": 1000.0 / mean_latency if mean_latency and mean_latency > 0 else "",
        "mean_callback_total_ms": mean(callback) if callback else "",
        "p95_callback_total_ms": _percentile(callback, 0.95),
        "mean_end_to_end_ms": mean(end_to_end) if end_to_end else "",
        "p95_end_to_end_ms": _percentile(end_to_end, 0.95),
        "mean_image_encoder_ms": mean(image) if image else "",
        "mean_text_encoder_ms": mean(text) if text else "",
        "mean_grounding_ms": mean(grounding) if grounding else "",
        "mean_mask_count": mean(masks) if masks else "",
        "mean_score_max": mean(scores) if scores else "",
    }


def _values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in ("", None):
            continue
        values.append(float(value))
    return values


def _percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    values = sorted(values)
    return values[int((len(values) - 1) * q)]


def main() -> None:
    rclpy.init()
    node = ResultRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
