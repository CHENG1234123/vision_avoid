import os
from pathlib import Path

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    del args, kwargs
    package_share = get_package_share_directory("vision_avoid")
    profile = LaunchConfiguration("profile").perform(context)
    video_device = LaunchConfiguration("video_device").perform(context)
    web_host = LaunchConfiguration("web_host").perform(context)
    web_port = LaunchConfiguration("web_port").perform(context)
    camera_rotate_180 = (
        LaunchConfiguration("camera_rotate_180").perform(context).lower() == "true"
    )
    drive_enabled = LaunchConfiguration("drive_enabled").perform(context).lower() == "true"
    allow_lan_provider_switch = (
        LaunchConfiguration("allow_lan_provider_switch").perform(context).lower()
        == "true"
    )
    config_file = f"{package_share}/config/{profile}.yaml"
    vehicle_file = LaunchConfiguration("vehicle_file").perform(context)
    serial_device = LaunchConfiguration("serial_device").perform(context)
    segment_m = LaunchConfiguration("segment_m").perform(context)
    if drive_enabled and (not vehicle_file or not serial_device):
        raise ValueError("drive_enabled requires vehicle_file and serial_device")
    venv_python = Path(os.environ.get("VIRTUAL_ENV", ""), "bin", "python")
    if not venv_python.is_file():
        venv_python = Path(package_share).parents[3] / ".venv" / "bin" / "python"
    python_prefix = f"{venv_python} " if venv_python.is_file() else None

    nodes = [
        Node(
            package="vision_avoid",
            executable="vision_camera_node",
            parameters=[
                {
                    "video_device": video_device,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "frame_id": "camera_link",
                    "rotate_180": camera_rotate_180,
                }
            ],
            output="screen",
            prefix=python_prefix,
        ),
        Node(
            package="vision_avoid",
            executable="vision_avoid_node",
            parameters=[
                {
                    "config_file": config_file,
                    "vehicle_file": vehicle_file,
                    "drive_enabled": drive_enabled,
                    "image_timeout_s": 0.3,
                }
            ],
            output="screen",
            emulate_tty=True,
            prefix=python_prefix,
        ),
        Node(
            package="vision_avoid",
            executable="vision_debug_web",
            parameters=[
                {
                    "host": web_host,
                    "port": int(web_port),
                    "allow_lan_provider_switch": allow_lan_provider_switch,
                }
            ],
            output="screen",
            prefix=python_prefix,
        ),
    ]
    if serial_device:
        nodes.append(
            Node(
                package="vision_avoid",
                executable="vision_stm32_bridge",
                parameters=[
                    {
                        "serial_device": serial_device,
                        "drive_enabled": drive_enabled,
                        "segment_m": float(segment_m),
                    }
                ],
                output="screen",
                prefix=python_prefix,
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile", default_value="indoor", choices=["indoor", "outdoor"]
            ),
            DeclareLaunchArgument(
                "video_device",
                default_value=(
                    "/dev/v4l/by-id/"
                    "usb-Sonix_Technology_Co.__Ltd._HD_webcam_SN0001-video-index0"
                ),
            ),
            DeclareLaunchArgument("vehicle_file", default_value=""),
            DeclareLaunchArgument("serial_device", default_value=""),
            DeclareLaunchArgument("segment_m", default_value="0.02"),
            DeclareLaunchArgument("camera_rotate_180", default_value="false"),
            DeclareLaunchArgument("web_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("web_port", default_value="8080"),
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            DeclareLaunchArgument(
                "allow_lan_provider_switch", default_value="false"
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
