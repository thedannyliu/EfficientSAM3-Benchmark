from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class OverlayVideoRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("overlay_video_recorder_node")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("video_output", "overlays/ros/ros_overlay.mp4")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("max_frames", 0)

        self.video_output = Path(self.get_parameter("video_output").value)
        self.fps = float(self.get_parameter("fps").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.bridge = CvBridge()
        self.writer: cv2.VideoWriter | None = None
        self.frames = 0

        self.video_output.parent.mkdir(parents=True, exist_ok=True)
        overlay_topic = self.get_parameter("overlay_topic").value
        self.subscription = self.create_subscription(Image, overlay_topic, self.on_image, 10)
        self.get_logger().info(f"recording {overlay_topic} to {self.video_output}")

    def on_image(self, msg: Image) -> None:
        frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        if self.writer is None:
            height, width = frame_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.video_output), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to create overlay video: {self.video_output}")
        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1
        if self.max_frames > 0 and self.frames >= self.max_frames:
            raise SystemExit

    def destroy_node(self) -> bool:
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(f"wrote {self.frames} frames to {self.video_output}")
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OverlayVideoRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
