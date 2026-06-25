from __future__ import annotations

from pathlib import Path

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from .video_recording import TimedVideoRecorder, stamp_to_seconds


class OverlayVideoRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("overlay_video_recorder_node")
        self.declare_parameter("overlay_topic", "/sam/overlay")
        self.declare_parameter("video_output", "overlays/ros/ros_overlay.mp4")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("preserve_timing", True)

        self.video_output = Path(self.get_parameter("video_output").value)
        self.fps = float(self.get_parameter("fps").value)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.preserve_timing = bool(self.get_parameter("preserve_timing").value)
        self.bridge = CvBridge()
        self.recorder = TimedVideoRecorder(
            enabled=True,
            output_path=self.video_output,
            fps=self.fps,
            max_frames=self.max_frames,
            preserve_timing=self.preserve_timing,
        )

        self.video_output.parent.mkdir(parents=True, exist_ok=True)
        overlay_topic = self.get_parameter("overlay_topic").value
        self.subscription = self.create_subscription(Image, overlay_topic, self.on_image, 10)
        self.get_logger().info(
            f"recording {overlay_topic} to {self.video_output}; fps={self.fps:g} "
            f"preserve_timing={self.preserve_timing}"
        )

    def on_image(self, msg: Image) -> None:
        frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.recorder.write(frame_rgb, stamp_to_seconds(msg.header.stamp))

    def destroy_node(self) -> bool:
        self.recorder.release(self.get_logger())
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
