from __future__ import annotations

import argparse


def main() -> None:
    args = parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from tf2_ros import TransformBroadcaster

    class PoseFixture(Node):
        def __init__(self) -> None:
            super().__init__("scene_graph_pose_fixture")
            qos = QoSProfile(depth=100)
            qos.reliability = ReliabilityPolicy.RELIABLE
            self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, qos)
            self.tf_broadcaster = TransformBroadcaster(self)
            self.count = 0
            self.create_subscription(CompressedImage, args.camera_topic, self._camera, qos)

        def _camera(self, msg: CompressedImage) -> None:
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = args.map_frame
            pose.pose.orientation.w = 1.0
            self.pose_pub.publish(pose)

            transform = TransformStamped()
            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = args.map_frame
            transform.child_frame_id = args.camera_frame
            transform.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(transform)

            self.count += 1
            if self.count <= 3 or self.count % 100 == 0:
                self.get_logger().info(
                    f"Published fixed pose and transform for camera frame {self.count}"
                )

    rclpy.init()
    node = PoseFixture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a deterministic identity pose/TF for Scene Graph A/B tests."
    )
    parser.add_argument("--camera-topic", default="/d435/color/image_raw_jpeg")
    parser.add_argument("--pose-topic", default="/tracked_pose")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--camera-frame", default="d435_color_optical_frame")
    return parser.parse_args()


if __name__ == "__main__":
    main()
