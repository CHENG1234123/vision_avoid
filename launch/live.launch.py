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
    drive_enabled = LaunchConfiguration("drive_enabled").perform(context).lower() == "true"
    config_file = f"{package_share}/config/{profile}.yaml"

    return [
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
                }
            ],
            output="screen",
        ),
        Node(
            package="vision_avoid",
            executable="vision_avoid_node",
            parameters=[
                {
                    "config_file": config_file,
                    "drive_enabled": drive_enabled,
                    "image_timeout_s": 0.3,
                }
            ],
            output="screen",
            emulate_tty=True,
        ),
        Node(
            package="vision_avoid",
            executable="vision_debug_web",
            parameters=[{"host": web_host, "port": int(web_port)}],
            output="screen",
        ),
    ]


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
                    "usb-LRCP_USB2.0_LRCP_USB2.0_SN0001-video-index0"
                ),
            ),
            DeclareLaunchArgument("web_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("web_port", default_value="8080"),
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            OpaqueFunction(function=launch_setup),
        ]
    )
