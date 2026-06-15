from __future__ import annotations

import json
from argparse import Namespace
from time import perf_counter
from typing import Any

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sam_backend.overlay import overlay_prediction
from sam_backend.profile_yolo_coco import (
    _build_model,
    _extract_detections,
    _set_open_vocab_classes,
    _sync,
    _yolo_parameter_counts,
)
from sam_backend.profiling import cuda_memory_mb
from sam_backend.streaming import masks_to_mono8
from sam_backend.yolo_streaming import detections_to_arrays, max_score


class YoloeTextBackendNode(Node):
    def __init__(self) -> None:
        super().__init__("yoloe_text_backend_node")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("mask_topic", "/segmentation_mask")
        self.declare_parameter("segmented_image_topic", "/segmented_image")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("weights", "checkpoints/yoloe/yoloe-26m-seg.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("prompt", "monitor")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("max_det", 20)

        self.bridge = CvBridge()
        self.weights = str(self.get_parameter("weights").value)
        self.device = str(self.get_parameter("device").value)
        self.prompt = str(self.get_parameter("prompt").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.frame_index = 0

        args = Namespace(family="yoloe-seg", weights=self.weights, device=self.device)
        self.model, self.torch_module = _build_model(args)
        set_classes_start = perf_counter()
        _set_open_vocab_classes(self.model, [self.prompt])
        _sync(self.torch_module)
        self.set_classes_ms = (perf_counter() - set_classes_start) * 1000.0
        self.params = _yolo_parameter_counts(getattr(self.model, "model", self.model))

        image_topic = str(self.get_parameter("image_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        mask_topic = str(self.get_parameter("mask_topic").value)
        segmented_image_topic = str(self.get_parameter("segmented_image_topic").value)
        overlay_topic = str(self.get_parameter("overlay_topic").value)

        self.result_publisher = self.create_publisher(String, result_topic, 10)
        self.mask_publisher = self.create_publisher(Image, mask_topic, 10) if mask_topic else None
        self.segmented_image_publisher = (
            self.create_publisher(Image, segmented_image_topic, 10) if segmented_image_topic else None
        )
        self.overlay_publisher = self.create_publisher(Image, overlay_topic, 10) if overlay_topic else None
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)

        self.get_logger().info(
            f"YOLOE text prompt '{self.prompt}' on {image_topic}; publishing {result_topic}"
        )
        if mask_topic:
            self.get_logger().info(f"publishing mono8 masks on {mask_topic}")
        if segmented_image_topic:
            self.get_logger().info(f"publishing segmented images on {segmented_image_topic}")
        if overlay_topic:
            self.get_logger().info(f"publishing overlays on {overlay_topic}")

    def on_image(self, msg: Image) -> None:
        callback_start = perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        predict_start = perf_counter()
        results = self.model.predict(frame, **self._predict_kwargs())
        _sync(self.torch_module)
        predict_ms = (perf_counter() - predict_start) * 1000.0

        postprocess_start = perf_counter()
        result = results[0] if results else None
        detections = _extract_detections(result, frame.shape[:2])
        masks, boxes, scores = detections_to_arrays(detections)
        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0
        callback_total_ms = (perf_counter() - callback_start) * 1000.0

        result_json = self._result(msg, detections, scores, predict_ms, postprocess_ms, callback_total_ms)
        self.result_publisher.publish(String(data=json.dumps(result_json)))
        self._publish_images(msg, frame, masks, boxes, scores)
        self.frame_index += 1

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
            "device": self.device,
        }
        if self.max_det > 0:
            kwargs["max_det"] = self.max_det
        return kwargs

    def _result(
        self,
        msg: Image,
        detections: list[dict[str, Any]],
        scores: list[float],
        predict_ms: float,
        postprocess_ms: float,
        callback_total_ms: float,
    ) -> dict[str, Any]:
        memory = cuda_memory_mb(self.torch_module) if self.torch_module is not None else cuda_memory_mb(None)
        mask_count = sum(1 for det in detections if det.get("mask") is not None)
        box_count = sum(1 for det in detections if det.get("box") is not None)
        return {
            "frame_index": self.frame_index,
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "backend": "yoloe",
            "family": "yoloe-seg",
            "weights": self.weights,
            "prompt_mode": "text",
            "prompt_text": self.prompt,
            "latency_ms": predict_ms + postprocess_ms,
            "callback_total_ms": callback_total_ms,
            "set_classes_ms": self.set_classes_ms if self.frame_index == 0 else 0.0,
            "predict_ms": predict_ms,
            "postprocess_ms": postprocess_ms,
            "all_detection_count": len(detections),
            "mask_count": mask_count,
            "box_count": box_count,
            "score_max": max_score(scores),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "max_det": self.max_det,
            **memory,
            **self.params,
        }

    def _publish_images(
        self,
        msg: Image,
        frame: Any,
        masks: list[Any],
        boxes: list[Any],
        scores: list[float],
    ) -> None:
        if self.mask_publisher is not None:
            mask = masks_to_mono8(masks, frame.shape[:2])
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self.mask_publisher.publish(mask_msg)
        if self.overlay_publisher is None and self.segmented_image_publisher is None:
            return
        overlay = overlay_prediction(frame, masks, boxes, scores)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
        overlay_msg.header = msg.header
        if self.overlay_publisher is not None:
            self.overlay_publisher.publish(overlay_msg)
        if self.segmented_image_publisher is not None:
            self.segmented_image_publisher.publish(overlay_msg)


def main() -> None:
    rclpy.init()
    node = YoloeTextBackendNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
