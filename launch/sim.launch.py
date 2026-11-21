from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    del args, kwargs
    package_share = get_package_share_directory("vision_avoid")
    gazebo_share = get_package_share_directory("gazebo_ros")
    world_name = LaunchConfiguration("world").perform(context)
    profile = LaunchConfiguration("profile").perform(context)
    web_host = LaunchConfiguration("web_host").perform(context)
    web_port = LaunchConfiguration("web_port").perform(context)
    drive_enabled = LaunchConfiguration("drive_enabled").perform(context).lower() == "true"
    xacro_file = f"{package_share}/urdf/vision_avoid_car.urdf.xacro"
    world_file = f"{package_share}/worlds/{world_name}.world"
    config_file = f"{package_share}/config/{profile}.yaml"

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(f"{gazebo_share}/launch/gazebo.launch.py"),
            launch_arguments={"world": world_file, "verbose": "false"}.items(),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[
                {
                    "robot_description": ParameterValue(
                        Command(["xacro ", xacro_file]), value_type=str
                    ),
                    "use_sim_time": True,
                }
            ],
            output="screen",
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-topic",
                "robot_description",
                "-entity",
                "vision_avoid_car",
                "-x",
                "0",
                "-y",
                "0",
                "-z",
                "0.05",
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
                    "use_sim_time": True,
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
                "world",
                default_value="avoidance",
                choices=["avoidance", "blocked", "corridor", "transition"],
            ),
            DeclareLaunchArgument(
                "profile", default_value="indoor", choices=["indoor", "outdoor"]
            ),
            DeclareLaunchArgument("web_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("web_port", default_value="8080"),
            DeclareLaunchArgument("drive_enabled", default_value="true"),
            OpaqueFunction(function=launch_setup),
        ]
    )
