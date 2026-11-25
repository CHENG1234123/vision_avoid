#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$script_dir")"
# shellcheck disable=SC1091
source "$script_dir/api_env.sh"
vision_avoid_load_api_env

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    printf 'ROS 2 Humble setup file is missing: /opt/ros/humble/setup.bash\n' >&2
    exit 2
fi
if [[ ! -f "$project_dir/install/setup.bash" ]]; then
    printf 'Workspace is not built: %s/install/setup.bash\n' "$project_dir" >&2
    exit 2
fi

# ROS and colcon setup scripts do not tolerate `set -u`.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$project_dir/.venv/bin/activate"
# shellcheck disable=SC1091
source "$project_dir/install/setup.bash"
set -u
exec ros2 launch vision_avoid live.launch.py "$@"
