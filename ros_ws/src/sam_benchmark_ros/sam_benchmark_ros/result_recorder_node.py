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
    "tracking_fps",
    "image_encoder_ms",
    "text_encoder_ms",
    "prompt_encoder_ms",
    "mask_decoder_ms",
    "transformer_ms",
    "geometry_encoder_ms",
    "segmentation_head_ms",
    "grounding_ms",
    "detector_ms",
    "memory_attention_ms",
    "memory_encoder_ms",
    "other_ms",
    "prompt_mode",
    "prompt_text",
    "point_x",
    "point_y",
    "mask_count",
    "box_count",
    "score_max",
    "cuda_allocated_mb",
    "cuda_reserved_mb",
    "cuda_peak_allocated_mb",
    "cuda_peak_reserved_mb",
    "params_total",
    "params_backbone",
    "params_image_encoder",
    "params_text_encoder",
    "params_transformer",
    "params_geometry_encoder",
    "params_segmentation_head",
    "params_prompt_encoder",
    "params_mask_decoder",
    "params_detector",
    "params_memory_attention",
    "params_memory_encoder",
    "weight_total_bytes",
    "weight_backbone_bytes",
    "weight_image_encoder_bytes",
    "weight_text_encoder_bytes",
    "weight_transformer_bytes",
    "weight_geometry_encoder_bytes",
    "weight_segmentation_head_bytes",
    "weight_prompt_encoder_bytes",
    "weight_mask_decoder_bytes",
    "weight_detector_bytes",
    "weight_memory_attention_bytes",
    "weight_memory_encoder_bytes",
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
    "mean_callback_fps",
    "p95_callback_total_ms",
    "mean_end_to_end_ms",
    "mean_end_to_end_fps",
    "p95_end_to_end_ms",
    "mean_tracking_fps",
    "mean_image_encoder_ms",
    "mean_text_encoder_ms",
    "mean_prompt_encoder_ms",
    "mean_mask_decoder_ms",
    "mean_transformer_ms",
    "mean_geometry_encoder_ms",
    "mean_segmentation_head_ms",
    "mean_grounding_ms",
    "mean_detector_ms",
    "mean_memory_attention_ms",
    "mean_memory_encoder_ms",
    "mean_mask_count",
    "mean_score_max",
    "params_total",
    "weight_total_bytes",
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
            "tracking_fps": data.get("tracking_fps"),
            "image_encoder_ms": data.get("image_encoder_ms"),
            "text_encoder_ms": data.get("text_encoder_ms"),
            "prompt_encoder_ms": data.get("prompt_encoder_ms"),
            "mask_decoder_ms": data.get("mask_decoder_ms"),
            "transformer_ms": data.get("transformer_ms"),
            "geometry_encoder_ms": data.get("geometry_encoder_ms"),
            "segmentation_head_ms": data.get("segmentation_head_ms"),
            "grounding_ms": data.get("grounding_ms"),
            "detector_ms": data.get("detector_ms"),
            "memory_attention_ms": data.get("memory_attention_ms"),
            "memory_encoder_ms": data.get("memory_encoder_ms"),
            "other_ms": data.get("other_ms"),
            "prompt_mode": data.get("prompt_mode"),
            "prompt_text": data.get("prompt_text"),
            "point_x": data.get("point_x"),
            "point_y": data.get("point_y"),
            "mask_count": data.get("mask_count"),
            "box_count": data.get("box_count"),
            "score_max": data.get("score_max"),
            "cuda_allocated_mb": data.get("cuda_allocated_mb"),
            "cuda_reserved_mb": data.get("cuda_reserved_mb"),
            "cuda_peak_allocated_mb": data.get("cuda_peak_allocated_mb"),
            "cuda_peak_reserved_mb": data.get("cuda_peak_reserved_mb"),
            "params_total": data.get("params_total"),
            "params_backbone": data.get("params_backbone"),
            "params_image_encoder": data.get("params_image_encoder"),
            "params_text_encoder": data.get("params_text_encoder"),
            "params_transformer": data.get("params_transformer"),
            "params_geometry_encoder": data.get("params_geometry_encoder"),
            "params_segmentation_head": data.get("params_segmentation_head"),
            "params_prompt_encoder": data.get("params_prompt_encoder"),
            "params_mask_decoder": data.get("params_mask_decoder"),
            "params_detector": data.get("params_detector"),
            "params_memory_attention": data.get("params_memory_attention"),
            "params_memory_encoder": data.get("params_memory_encoder"),
            "weight_total_bytes": data.get("weight_total_bytes"),
            "weight_backbone_bytes": data.get("weight_backbone_bytes"),
            "weight_image_encoder_bytes": data.get("weight_image_encoder_bytes"),
            "weight_text_encoder_bytes": data.get("weight_text_encoder_bytes"),
            "weight_transformer_bytes": data.get("weight_transformer_bytes"),
            "weight_geometry_encoder_bytes": data.get("weight_geometry_encoder_bytes"),
            "weight_segmentation_head_bytes": data.get("weight_segmentation_head_bytes"),
            "weight_prompt_encoder_bytes": data.get("weight_prompt_encoder_bytes"),
            "weight_mask_decoder_bytes": data.get("weight_mask_decoder_bytes"),
            "weight_detector_bytes": data.get("weight_detector_bytes"),
            "weight_memory_attention_bytes": data.get("weight_memory_attention_bytes"),
            "weight_memory_encoder_bytes": data.get("weight_memory_encoder_bytes"),
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
    tracking_fps = _values(rows, "tracking_fps")
    image = _values(rows, "image_encoder_ms")
    text = _values(rows, "text_encoder_ms")
    prompt = _values(rows, "prompt_encoder_ms")
    mask = _values(rows, "mask_decoder_ms")
    transformer = _values(rows, "transformer_ms")
    geometry = _values(rows, "geometry_encoder_ms")
    segmentation = _values(rows, "segmentation_head_ms")
    grounding = _values(rows, "grounding_ms")
    detector = _values(rows, "detector_ms")
    memory_attention = _values(rows, "memory_attention_ms")
    memory_encoder = _values(rows, "memory_encoder_ms")
    masks = _values(rows, "mask_count")
    scores = _values(rows, "score_max")
    mean_latency = mean(latency) if latency else None
    mean_callback = mean(callback) if callback else None
    mean_end_to_end = mean(end_to_end) if end_to_end else None
    return {
        "frames": len(rows),
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": _percentile(latency, 0.50),
        "p95_latency_ms": _percentile(latency, 0.95),
        "mean_latency_fps": 1000.0 / mean_latency if mean_latency and mean_latency > 0 else "",
        "mean_callback_total_ms": mean_callback if mean_callback else "",
        "mean_callback_fps": 1000.0 / mean_callback if mean_callback and mean_callback > 0 else "",
        "p95_callback_total_ms": _percentile(callback, 0.95),
        "mean_end_to_end_ms": mean_end_to_end if mean_end_to_end else "",
        "mean_end_to_end_fps": 1000.0 / mean_end_to_end if mean_end_to_end and mean_end_to_end > 0 else "",
        "p95_end_to_end_ms": _percentile(end_to_end, 0.95),
        "mean_tracking_fps": mean(tracking_fps) if tracking_fps else "",
        "mean_image_encoder_ms": mean(image) if image else "",
        "mean_text_encoder_ms": mean(text) if text else "",
        "mean_prompt_encoder_ms": mean(prompt) if prompt else "",
        "mean_mask_decoder_ms": mean(mask) if mask else "",
        "mean_transformer_ms": mean(transformer) if transformer else "",
        "mean_geometry_encoder_ms": mean(geometry) if geometry else "",
        "mean_segmentation_head_ms": mean(segmentation) if segmentation else "",
        "mean_grounding_ms": mean(grounding) if grounding else "",
        "mean_detector_ms": mean(detector) if detector else "",
        "mean_memory_attention_ms": mean(memory_attention) if memory_attention else "",
        "mean_memory_encoder_ms": mean(memory_encoder) if memory_encoder else "",
        "mean_mask_count": mean(masks) if masks else "",
        "mean_score_max": mean(scores) if scores else "",
        "params_total": _first_value(rows, "params_total"),
        "weight_total_bytes": _first_value(rows, "weight_total_bytes"),
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


def _first_value(rows: list[dict[str, Any]], field: str) -> Any:
    for row in rows:
        value = row.get(field)
        if value not in ("", None):
            return value
    return ""


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
