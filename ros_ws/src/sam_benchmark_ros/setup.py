from setuptools import setup

package_name = "sam_benchmark_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="eliu354",
    maintainer_email="eliu354@gatech.edu",
    description="ROS 2 wrappers for SAM3/EfficientSAM3 benchmark backend.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_stream_node = sam_benchmark_ros.camera_stream_node:main",
            "live_viewer_node = sam_benchmark_ros.live_viewer_node:main",
            "mobile_sam_interactive_node = sam_benchmark_ros.mobile_sam_interactive_node:main",
            "overlay_video_recorder_node = sam_benchmark_ros.overlay_video_recorder_node:main",
            "result_recorder_node = sam_benchmark_ros.result_recorder_node:main",
            "sam3_native_clip_node = sam_benchmark_ros.sam3_native_clip_node:main",
            "sam_backend_node = sam_benchmark_ros.sam_backend_node:main",
            "video_stream_node = sam_benchmark_ros.video_stream_node:main",
            "yoloe_text_backend_node = sam_benchmark_ros.yoloe_text_backend_node:main",
        ],
    },
)
