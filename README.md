# 单摄像头视觉避障小车

ROS 2 Humble + Gazebo 11 的低速单目视觉避障原型。小车持续使用本地
D-FINE-S 和 Depth Anything V2 Metric Small 做物体、深度和几何障碍检测；
启动或进入新视觉场景时先停车，由本地 Qwen3-VL 4B 扫描一帧并通过门禁后
才继续行驶。

远端视觉 API 默认关闭。只有显式切换 provider 和启用 API 时才会上传图片。
本地场景模型通过 LM Studio 的 OpenAI-compatible 接口加载。

## 模型与许可

| 用途 | 模型 | 许可 |
|---|---|---|
| 物体识别 | [D-FINE-S COCO](https://huggingface.co/ustc-community/dfine-small-coco) | Apache-2.0 |
| 米制深度 | [Depth Anything V2 Metric Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) | Apache-2.0 |
| 场景补充扫描 | [Qwen3-VL-4B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Apache-2.0 |

下载脚本固定 Hugging Face revision。Qwen 模型默认由本地 LM Studio 加载。

## 环境要求

- Ubuntu 22.04、ROS 2 Humble、Gazebo 11、colcon
- Python 3.10（不要使用系统默认的 Python 3.14）
- NVIDIA 驱动和 CUDA 12.8 兼容的 PyTorch 2.7
- 建议至少 12 GB 显存、30 GB 内存和 10 GB 空闲磁盘
- 本地场景扫描需要 LM Studio，本地服务器监听 `127.0.0.1:1234`

当前验证机器为 RTX 5070 Ti Laptop 12 GB。运行时只允许一个 Qwen 实例，
上下文固定为 4096 Token。

## 已验证结果

- D-FINE-S + Depth Anything 连续推理约 15.2 FPS，p50 63 ms，p95 78 ms。
- 两个实时模型加载后约使用 1.3 GB 总显存。
- Qwen3-VL 4B Q4_K_M 完整驻留 GPU 约 3.57 GB，冷启动后的单帧扫描约 1.4–4 秒。
- Qwen 与 D-FINE、Depth 同时加载时，整卡实测占用约 5.8 GB，剩余约 5.9 GB。
- 当前 USB 摄像头实拍中，D-FINE 检出 `person`（置信度约 0.93），Qwen
  返回 `caution/left` 并指出右侧有人、左侧地面可见。
- 新增的 LRCP 摄像头通过稳定 by-id 路径接入，实拍发布 640×480 MJPG，
  完整运行下实测约 23 FPS。
- 真实 Gazebo 中央障碍场景中，Qwen 返回 `caution/left`，小车低速绕过并移动到约
  `(4.58, 0.78)`。
- 真实 Gazebo 死路场景中，扫描未通过，`/vision/scan_gate_open=false`，
  `/cmd_vel` 始终为零。

## 安装

```bash
conda create -n vision-avoid python=3.10 -y
conda activate vision-avoid
pip install -r requirements-cu128.txt
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

当前工作区还已经创建了 `.venv`，也可以直接运行 `source .venv/bin/activate`。

在 LM Studio 中加载 `Qwen3-VL-4B-Instruct-Q4_K_M`，然后从
**Developer → Local Server** 启动服务，端口保持 `1234`。确认接口可用：

```bash
curl -fsS http://127.0.0.1:1234/v1/models
```

下载 D-FINE 和两种 Depth Anything 权重：

```bash
ros2 run vision_avoid vision-download-models
```

LM Studio 模型在 LM Studio 中单独下载。

模型默认保存在 `~/.local/share/vision_avoid/models`。若使用其他目录，设置
`VISION_MODELS_DIR=/absolute/path` 并给下载脚本传 `--models-dir`。

## 运行

默认本地后端是 LM Studio。先确认 LM Studio Local Server 已启动并加载
Qwen3-VL-4B，然后设置模型标识：

```bash
export VLM_LMSTUDIO_MODEL='qwen3-vl-4b-instruct-vision'
export VLM_LMSTUDIO_URL='http://127.0.0.1:1234/v1'
```

仿真默认启动室内 profile 和中央障碍场景：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch vision_avoid sim.launch.py
```

可选场景：

```bash
ros2 launch vision_avoid sim.launch.py world:=blocked profile:=indoor
ros2 launch vision_avoid sim.launch.py world:=corridor profile:=indoor
ros2 launch vision_avoid sim.launch.py world:=transition profile:=indoor
ros2 launch vision_avoid sim.launch.py world:=avoidance profile:=outdoor
```

USB 摄像头默认只观察，不发布 `/cmd_vel`。当前 Sonix HD webcam 使用稳定的
`/dev/v4l/by-id/...-video-index0` 路径：

```bash
ros2 launch vision_avoid live.launch.py profile:=indoor
```

若设备路径变化，可覆盖参数：

```bash
ros2 launch vision_avoid live.launch.py video_device:=/dev/video2
```

查看调试画面：

```bash
bash scripts/view_debug.sh
```

也可以直接打开 `http://127.0.0.1:8080/`。原生 HTML 仪表盘分别显示摄像头检测、
米制深度和障碍俯视图，桌面为左右分栏，手机宽度下自动切换为单列。三路 MJPEG
仅在浏览器打开对应画面时编码：摄像头目标 15 FPS，深度和障碍俯视图各 6 FPS。
深度和俯视图输出为 480×270，JPEG 质量为 70。滚出浏览器视口或切到后台标签页
时暂停对应视频流，重新进入视口后自动恢复。网页不需要 ROS GUI 插件或 Conda
Python 兼容层。

若需让同一局域网内的手机查看，可用 `web_host:=0.0.0.0` 启动。默认仍会锁定
Provider 切换控件；明确需要手机切换本地与云端时，再增加
`allow_lan_provider_switch:=true`。页面没有身份认证，开启后仅应在可信局域网
使用，任何能打开页面的设备都可能触发云端 API 请求和费用。

调试接口：

| 接口 | 用途 |
|---|---|
| `/vision/camera_debug` | 摄像头与检测框 |
| `/vision/depth_debug` | 近红远蓝的深度图 |
| `/vision/bev_debug` | 浅色障碍俯视图 |
| `/vision/debug_status` | JSON 调试状态 |
| `/vision/debug_image` | 旧版综合调试图 |
| `/stream/camera.mjpg` | 摄像头 MJPEG |
| `/stream/depth.mjpg` | 深度 MJPEG |
| `/stream/bev.mjpg` | 障碍俯视图 MJPEG |
| `/stream.mjpg` | 旧版综合调试图 MJPEG |
| `/api/status` | 浏览器状态与会话流统计 |
| `/api/provider` | 本机或显式启用的局域网 Provider 切换 |

## 安全门禁

- 启动后先扫描一次；画面指纹持续变化时先减速停车并重新扫描。
- 只有扫描结果有效且本地深度前方存在通路时才开门。
- `caution` 限速为 0.15 m/s；`blocked`、`unknown`、超时或无效结果保持停车。
- 本地或远端模型只能增加风险约束，不能覆盖本地深度碰撞判断。
- 图片超时、摄像头黑屏、严重模糊、感知异常或 VLM 服务退出都会输出零速度。

相关接口：

| 接口 | 类型 |
|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image` |
| `/cmd_vel` | `geometry_msgs/msg/Twist` |
| `/vision/debug_image` | `sensor_msgs/msg/Image` |
| `/vision/camera_debug` | `sensor_msgs/msg/Image` |
| `/vision/depth_debug` | `sensor_msgs/msg/Image` |
| `/vision/bev_debug` | `sensor_msgs/msg/Image` |
| `/vision/debug_status` | `std_msgs/msg/String`（JSON） |
| `/vision/rescan_scene` | `std_srvs/srv/Trigger` |
| `/vision/rescan_local_vlm` | `std_srvs/srv/Trigger` |
| `/vision/scan_gate_open` | `std_msgs/msg/Bool` |
| `/vision/provider_status` | `diagnostic_msgs/msg/DiagnosticStatus` |
| `/vision/set_api_provider` | `std_srvs/srv/SetBool` |

手动重扫：

```bash
ros2 service call /vision/rescan_scene std_srvs/srv/Trigger "{}"
```

`/vision/rescan_local_vlm` 只在 `VLM_PROVIDER=local` 时工作；API 模式下调用它会
直接拒绝，不会暗中上传图片。

## 可选远端 API

默认配置为 `provider=local`、`api_enabled=false`，不会发起 HTTP 请求。
远端 provider 不会在本地失败时自动启用。要允许网页中的“云端 API”按钮，
启动前预配置 API 但保持默认 provider 为本地：

示例配置阿里云 OpenAI-compatible 视觉接口：

```bash
export VLM_API_ENABLED=true
export VLM_API_BASE_URL='https://<workspace>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
export VLM_API_MODEL='qwen3-vl-flash'
export VLM_STRUCTURED_OUTPUT_MODE=json_schema
export DASHSCOPE_API_KEY='<runtime secret>'
ros2 launch vision_avoid sim.launch.py
```

也可以把非密钥配置放在仓库外的
`~/.config/vision_avoid/api.env`，密钥单独放在权限为 `600` 的文件中，再由
`VLM_API_KEY_FILE` 引用。配置模板见 `config/api.env.example`：

```bash
VLM_PROVIDER=api
VLM_API_ENABLED=true
VLM_API_BASE_URL=https://<api-host>/compatible-mode/v1
VLM_API_MODEL=<vision-model>
VLM_API_TIMEOUT_S=30
VLM_STRUCTURED_OUTPUT_MODE=json_schema
VLM_API_ENABLE_THINKING=false
VLM_API_KEY_ENV=DASHSCOPE_API_KEY
VLM_API_KEY_FILE=/absolute/path/to/api.key
```

`chmod 600 ~/.config/vision_avoid/api.env`。启动前可先用一张场景图验证鉴权、
模型和结构化输出：

```bash
bash scripts/check_api.sh /path/to/scene.jpg
```

验证通过后，用同一份配置启动实车观察模式：

```bash
bash scripts/run_api.sh profile:=indoor drive_enabled:=false
```

百炼 Token Plan 的 `sk-sp-` 密钥只能配合 Token Plan Base URL；它面向交互式
工具。若机器人要作为后端服务长期调用，应改用按量付费的 `sk-` 密钥和对应的
百炼 Base URL。`VLM_API_ENABLE_THINKING=false` 可避免 Qwen 场景扫描因思考过程
变慢；若所用服务不支持该参数，需要移除该项。

手机网页同时启用 Provider 热切换：

```bash
ros2 launch vision_avoid live.launch.py \
  web_host:=0.0.0.0 \
  allow_lan_provider_switch:=true
```

API 地址、模型和密钥仍只能来自启动环境变量，网页不能修改凭据。

启动后点击页面顶部“云端 API”，确认上传提示。切换状态先显示为“待切换”，
当前扫描继续完成；下一次新环境或手动扫描才使用 API。没有扫描进行时立即卸载
本地 Qwen 以释放显存，有扫描时等该扫描结束后卸载。LM Studio 后端通过
`/api/v1/models/unload` 卸载实例；切回“本地 Qwen”同样在下一次扫描生效，
并通过 LM Studio 重新加载模型。

也可通过 ROS 服务切换：

```bash
ros2 service call /vision/set_api_provider std_srvs/srv/SetBool "{data: true}"
ros2 service call /vision/set_api_provider std_srvs/srv/SetBool "{data: false}"
```

`data: true` 请求 API，`data: false` 请求本地。API 未配置、认证失败、超时或
返回无效结果时保持停车并进入 `FAULT`，不会自动回退到本地。

API 密钥只从环境变量读取，不写入配置文件、源码或网页。URL、模型名和密钥
均不能从浏览器修改。若服务不支持 JSON Schema，可把
`VLM_STRUCTURED_OUTPUT_MODE` 改为 `json_object`；输出仍会经过严格校验，
失败不会打开门禁。请求不自动重试，并限制为每小时 20 次。
远端地址必须是 HTTPS（本机代理可用 HTTP）；启用前应确认服务商的图片保留和
隐私政策。

## 相机标定

相机内参和安装几何在 `config/indoor.yaml` 或 `config/outdoor.yaml` 中。
使用棋盘格图片或摄像头生成标定结果：

```bash
ros2 run vision_avoid vision-calibrate-camera --images ./calibration_images --cols 9 --rows 6 --square-size 0.025
```

实时采集模式：

```bash
ros2 run vision_avoid vision-calibrate-camera --camera 0
```

把输出的 `fx/fy/cx/cy` 写入对应 profile。安装高度和俯仰角必须按实车测量；
仿真配置使用 0.32 m 和 8 度。

## 测试

不依赖模型的逻辑测试：

```bash
PYTHONPATH=. python3.10 -m pytest -q test
```

ROS 构建测试：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
colcon test --packages-select vision_avoid
```

需要人工检查的模型与 Gazebo 验收：

1. 三个模型同时加载后无 OOM，显存至少保留 0.5 GB。
2. 检测与深度闭环平均不低于 10 FPS。
3. 进入新场景时 `/vision/scan_gate_open` 先变为 `false`。
4. Qwen 返回安全结果且本地无障碍后，车辆继续。
5. 死路和摄像头断开时 `/cmd_vel` 为零。
6. 中央障碍场景连续运行 5 次不碰撞。
7. 本地扫描进行中切到 API 不中断当前扫描，结束后 Qwen 卸载。
8. 切回本地后下一次扫描重新加载 Qwen，API 失败时车辆保持停车。

## 已知限制

- 不使用 LiDAR、深度相机、SLAM 或全局地图。
- 不包含真实车辆串口、电机、急停或遥控接管。
- 单目深度对玻璃、强反光、无纹理表面和逆光可能不准。
- COCO 之外的物体仍会被深度投影为几何障碍，但未必有正确语义标签。
- Qwen 4B 可能漏检或幻觉；门禁始终同时要求本地深度通路。
- 本项目是低速演示原型，不是经过认证的移动机器人安全系统。
