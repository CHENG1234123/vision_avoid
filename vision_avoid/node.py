from __future__ import annotations

import collections
import json
import subprocess
import threading
import time
from dataclasses import replace
from enum import Enum

import cv2
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from vision_avoid.config import load_config
from vision_avoid.geometry import (
    CommandSelector,
    MotionCommand,
    OccupancyGrid,
    apply_scan_constraints,
    front_clearance,
)
from vision_avoid.perception import (
    LocalPerception,
    PerceptionOutput,
    render_bev_view,
    render_camera_view,
    render_debug,
    render_depth_view,
)
from vision_avoid.providers import SceneScanError, SceneScanner, create_scanner
from vision_avoid.scene import SceneScan
from vision_avoid.scene_change import SceneChangeDetector, SceneEvent


class RunState(str, Enum):
    INIT = "INIT"
    SCANNING = "SCANNING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    FAULT = "FAULT"


class AvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_avoid_node")
        self.declare_parameter("config_file", "")
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("min_gpu_free_mb", 512)
        self.declare_parameter("image_timeout_s", 0.3)

        config_file = str(self.get_parameter("config_file").value)
        if not config_file:
            raise ValueError("config_file parameter is required")
        self.config = load_config(config_file)
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.min_gpu_free_mb = int(self.get_parameter("min_gpu_free_mb").value)
        self.image_timeout_s = float(self.get_parameter("image_timeout_s").value)

        self._bridge = CvBridge()
        self._lock = threading.RLock()
        self._state = RunState.INIT
        self._error = ""
        self._gate_open = False
        self._need_scan = True
        self._scanning = False
        self._active_scan_generation = 0
        self._scan_generation = 0
        self._scan_result: SceneScan | None = None
        self._pending_scan: SceneScan | None = None
        self._pending_scan_generation = 0
        self._pending_scan_error = ""
        self._last_image_at = 0.0
        self._last_command_at = 0.0
        self._last_command = MotionCommand()
        self._last_local_safe = False
        self._last_scan_started_at = 0.0
        self._last_scan_duration_s = 0.0
        self._last_scan_usage: dict[str, int] = {}
        self._scan_prior_until = 0.0
        self._last_fault_message = ""
        self._last_fault_log_at = 0.0
        self._last_processed_at = 0.0
        self._processing_fps = 0.0
        self._last_inference_ms = 0.0
        self._last_perception: PerceptionOutput | None = None
        self._last_frame = None
        self._last_occupancy: OccupancyGrid | None = None
        self._debug_stream_active = {
            "camera": False,
            "depth": False,
            "bev": False,
        }
        self._api_calls: collections.deque[float] = collections.deque()

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._debug_pub = self.create_publisher(
            Image, "/vision/debug_image", qos_profile_sensor_data
        )
        self._camera_debug_pub = self.create_publisher(
            Image, "/vision/camera_debug", qos_profile_sensor_data
        )
        self._depth_debug_pub = self.create_publisher(
            Image, "/vision/depth_debug", qos_profile_sensor_data
        )
        self._bev_debug_pub = self.create_publisher(
            Image, "/vision/bev_debug", qos_profile_sensor_data
        )
        self._camera_jpeg_pub = self.create_publisher(
            CompressedImage, "/vision/camera_jpeg", qos_profile_sensor_data
        )
        self._depth_jpeg_pub = self.create_publisher(
            CompressedImage, "/vision/depth_jpeg", qos_profile_sensor_data
        )
        self._bev_jpeg_pub = self.create_publisher(
            CompressedImage, "/vision/bev_jpeg", qos_profile_sensor_data
        )
        self._debug_status_pub = self.create_publisher(
            String, "/vision/debug_status", 10
        )
        self._gate_pub = self.create_publisher(Bool, "/vision/scan_gate_open", 10)
        self._status_pub = self.create_publisher(
            DiagnosticStatus, "/vision/provider_status", 10
        )

        image_group = MutuallyExclusiveCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()
        debug_group = MutuallyExclusiveCallbackGroup()
        status_group = MutuallyExclusiveCallbackGroup()
        self._image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_callback,
            qos_profile_sensor_data,
            callback_group=image_group,
        )
        self._control_timer = self.create_timer(
            0.05, self._control_tick, callback_group=timer_group
        )
        self._debug_timer = self.create_timer(
            1.0 / 30.0, self._publish_debug_tick, callback_group=debug_group
        )
        self._status_timer = self.create_timer(
            1.0, self._publish_status, callback_group=status_group
        )
        self.create_subscription(
            String,
            "/vision/debug_clients",
            self._debug_clients_callback,
            10,
            callback_group=status_group,
        )
        self._rescan_srv = self.create_service(
            Trigger, "/vision/rescan_scene", self._rescan_callback
        )
        self._local_rescan_srv = self.create_service(
            Trigger, "/vision/rescan_local_vlm", self._local_rescan_callback
        )
        self._provider_srv = self.create_service(
            SetBool, "/vision/set_api_provider", self._set_api_provider_callback
        )

        self._scene_detector = SceneChangeDetector(self.config.scene_change)
        self._selector = CommandSelector(self.config.planner)
        self._scanner: SceneScanner | None = None
        self._perception: LocalPerception | None = None
        self._active_provider = self.config.vlm.provider
        self._pending_provider: str | None = None
        self._provider_change_id = 0
        self._unload_qwen_after_scan = False
        self._local_qwen_unloaded = False
        self._local_qwen_unload_thread: threading.Thread | None = None
        self._api_available = self._probe_api_config()
        self._debug_stream_periods = {
            "camera": 1.0 / 15.0,
            "depth": 1.0 / 6.0,
            "bev": 1.0 / 6.0,
        }
        self._last_stream_debug_at = {
            name: 0.0 for name in self._debug_stream_periods
        }
        self._last_legacy_debug_at = 0.0
        self._last_debug_status_at = 0.0
        self._initialize_runtime()
        if self._state == RunState.FAULT:
            self.get_logger().error("vision_avoid is running with motion disabled")
        else:
            self.get_logger().info(
                f"vision_avoid ready: profile={self.config.profile} "
                f"provider={self._active_provider} "
                f"drive_enabled={self.drive_enabled}"
            )

    def _initialize_runtime(self) -> None:
        try:
            self._scanner = self._create_provider_scanner(
                self._active_provider, warm=True
            )
            self._perception = LocalPerception(self.config)
        except Exception as exc:
            self._state = RunState.FAULT
            self._error = f"startup failed: {type(exc).__name__}: {exc}"
            self.get_logger().error(self._error)

    def _provider_config(self, provider: str):
        return replace(self.config.vlm, provider=provider)

    def _create_provider_scanner(
        self, provider: str, warm: bool = False
    ) -> SceneScanner:
        scanner = create_scanner(self._provider_config(provider))
        if not scanner.health_check():
            raise SceneScanError(f"{provider} provider health check failed")
        if warm and provider == "local":
            scanner.warmup()
            self._check_gpu_headroom()
            with self._lock:
                self._local_qwen_unloaded = False
        return scanner

    def _probe_api_config(self) -> bool:
        try:
            create_scanner(self._provider_config("api"))
            return True
        except SceneScanError:
            return False

    def _check_gpu_headroom(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            )
            free_mb = min(int(line.strip()) for line in output.splitlines() if line.strip())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise RuntimeError("could not verify GPU memory headroom") from exc
        if free_mb < self.min_gpu_free_mb:
            raise RuntimeError(
                f"GPU headroom {free_mb} MiB is below {self.min_gpu_free_mb} MiB"
            )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            self._fault(f"image decode failed: {type(exc).__name__}")
            return

        now = time.monotonic()
        with self._lock:
            self._last_image_at = now
            self._last_frame = frame
            scanning = self._scanning

        if scanning:
            self._publish_stop()
            return

        if self._perception is None:
            self._fault("local perception is unavailable")
            self._publish_stop()
            return

        inference_started = time.monotonic()
        try:
            perception = self._perception.infer(frame)
        except Exception as exc:
            self._fault(f"perception failed: {type(exc).__name__}: {exc}")
            self._publish_stop()
            return
        inference_ms = (time.monotonic() - inference_started) * 1000.0
        if self._last_processed_at:
            period = max(now - self._last_processed_at, 1e-6)
            instantaneous_fps = 1.0 / period
            self._processing_fps = (
                instantaneous_fps
                if self._processing_fps == 0.0
                else 0.85 * self._processing_fps + 0.15 * instantaneous_fps
            )
        self._last_processed_at = now
        self._last_inference_ms = inference_ms
        with self._lock:
            self._last_perception = perception

        observation = self._scene_detector.update(frame, now)
        if observation.event == SceneEvent.QUALITY_REJECTED:
            self._fault(
                "camera quality rejected "
                f"(brightness={observation.brightness:.1f}, "
                f"contrast={observation.contrast:.1f}, "
                f"sharpness={observation.sharpness:.1f})"
            )
            self._publish_stop()
            return

        if observation.event == SceneEvent.NEW_SCENE:
            with self._lock:
                self._trigger_scan_locked("new environment detected")

        with self._lock:
            recoverable = self._state == RunState.FAULT and self._error.startswith(
                ("camera quality rejected", "camera frame timeout")
            )
            if recoverable:
                self._error = ""
                self._trigger_scan_locked("recovering after camera fault")
        self._start_scan_if_needed(frame, now)

        with self._lock:
            scan = self._scan_result
            gate_open = self._gate_open
            scanning = self._scanning or self._need_scan

        speed_cap = self.config.robot.max_linear_mps
        preferred = "unknown"
        if scan is not None:
            speed_cap = scan.speed_cap(
                self.config.robot.max_linear_mps,
                self.config.robot.caution_linear_mps,
            )
            preferred = scan.safe_direction

        occupancy = (
            apply_scan_constraints(perception.occupancy, scan, self.config.robot)
            if scan is not None and now < self._scan_prior_until
            else perception.occupancy
        )
        with self._lock:
            self._last_occupancy = occupancy
        local_command = self._selector.select(
            occupancy,
            self.config.robot,
            preferred,
            now,
            speed_cap,
        )
        local_safe = not local_command.is_stop
        command = local_command if gate_open and not scanning else MotionCommand()

        with self._lock:
            self._last_command = command
            self._last_command_at = now
            self._last_local_safe = local_safe
            if self._state != RunState.FAULT:
                if scanning:
                    self._state = RunState.SCANNING
                elif command.is_stop:
                    self._state = RunState.BLOCKED
                    if gate_open:
                        self._error = "local depth path is blocked"
                else:
                    self._state = RunState.RUNNING
                    if gate_open:
                        self._error = ""

    def _start_scan_if_needed(self, frame, now: float) -> None:
        with self._lock:
            if not self._need_scan or self._scanning:
                return
            if self._scanner is None and self._pending_provider is None:
                return
            target_provider = self._pending_provider or self._active_provider
            if target_provider == "api":
                self._trim_api_calls(now)
                if len(self._api_calls) >= self.config.vlm.max_calls_per_hour:
                    self._pending_scan_error = "API hourly call limit reached"
                    self._pending_scan_generation = self._scan_generation
                    self._need_scan = False
                    return
                self._api_calls.append(now)
                self.get_logger().warning(
                    "uploading one frame to the explicitly configured remote VLM provider"
                )
            generation = self._scan_generation
            self._scanning = True
            self._active_scan_generation = generation
            self._need_scan = False
            self._last_scan_started_at = now
            self._state = RunState.SCANNING

        jpeg = encode_snapshot(frame)
        worker = threading.Thread(
            target=self._scan_worker,
            args=(jpeg, generation, target_provider),
            daemon=True,
        )
        worker.start()

    def _scan_worker(
        self, jpeg: bytes, generation: int, target_provider: str
    ) -> None:
        started_at = time.monotonic()
        try:
            scanner = self._scanner_for_scan(target_provider)
            result = scanner.scan(jpeg)
            with self._lock:
                self._pending_scan = result
                self._pending_scan_generation = generation
                self._pending_scan_error = ""
                self._last_scan_duration_s = time.monotonic() - started_at
                self._last_scan_usage = dict(
                    getattr(scanner, "last_usage", {})
                )
                if generation == self._scan_generation:
                    self._scanner = scanner
                    self._active_provider = target_provider
                    if self._pending_provider == target_provider:
                        self._pending_provider = None
            self.get_logger().info(
                f"scene scan complete: provider={target_provider} "
                f"{result.traversability} "
                f"confidence={result.confidence:.2f}"
            )
        except Exception as exc:
            with self._lock:
                self._pending_scan = None
                self._pending_scan_error = f"{type(exc).__name__}: {exc}"
                self._pending_scan_generation = generation
            self.get_logger().warning(f"scene scan failed: {type(exc).__name__}")
        finally:
            unload_local = False
            with self._lock:
                if self._active_scan_generation == generation:
                    self._scanning = False
                if (
                    self._unload_qwen_after_scan
                    and self._pending_provider == "api"
                ):
                    self._unload_qwen_after_scan = False
                    unload_local = True
            if unload_local:
                self._unload_local_qwen_async()

    def _scanner_for_scan(self, target_provider: str) -> SceneScanner:
        with self._lock:
            scanner = self._scanner
            active_provider = self._active_provider
            unload_thread = self._local_qwen_unload_thread
        if target_provider != active_provider or scanner is None:
            return self._create_provider_scanner(target_provider, warm=True)
        if target_provider != "local":
            return scanner
        if unload_thread is not None and unload_thread is not threading.current_thread():
            unload_thread.join()
        with self._lock:
            unloaded = self._local_qwen_unloaded
        if unloaded:
            scanner.warmup()
            self._check_gpu_headroom()
            with self._lock:
                self._local_qwen_unloaded = False
        return scanner

    def _control_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            if (
                self._pending_scan is not None
                and self._pending_scan_generation == self._scan_generation
            ):
                self._scan_result = self._pending_scan
                self._pending_scan = None
                scan_allowed = self._scan_result.allows_motion(
                    self.config.vlm.confidence_threshold
                )
                if scan_allowed:
                    self._gate_open = True
                    self._scan_prior_until = (
                        now + self.config.vlm.scan_prior_s
                    )
                    if self._last_local_safe:
                        self._state = RunState.READY
                        self._error = ""
                    else:
                        self._state = RunState.BLOCKED
                        self._error = "local depth path is blocked"
                else:
                    self._gate_open = False
                    self._state = RunState.BLOCKED
                    self._error = "scene scan did not permit motion"
            elif (
                self._pending_scan_error
                and self._pending_scan_generation == self._scan_generation
            ):
                self._gate_open = False
                self._state = RunState.FAULT
                self._error = self._pending_scan_error
                self._pending_scan_error = ""

            if now - self._last_image_at > self.image_timeout_s:
                self._gate_open = False
                self._state = RunState.FAULT
                self._error = "camera frame timeout"
                command = MotionCommand()
            elif (
                not self._gate_open
                or self._state in {RunState.FAULT, RunState.SCANNING, RunState.BLOCKED}
                or now - self._last_command_at > self.image_timeout_s
            ):
                command = MotionCommand()
            else:
                command = self._last_command

        active_gate = (
            self._gate_open
            and self._last_local_safe
            and self._state in {RunState.READY, RunState.RUNNING}
            and now - self._last_image_at <= self.image_timeout_s
        )
        self._publish_command(command)
        if rclpy.ok():
            self._gate_pub.publish(Bool(data=active_gate))

    def _publish_command(self, command: MotionCommand) -> None:
        if not self.drive_enabled or not rclpy.ok():
            return
        twist = Twist()
        twist.linear.x = float(command.linear_x)
        twist.angular.z = float(command.angular_z)
        try:
            self._cmd_pub.publish(twist)
        except Exception:
            if rclpy.ok():
                raise

    def _publish_stop(self) -> None:
        self._publish_command(MotionCommand())

    def _publish_debug(
        self,
        frame,
        perception: PerceptionOutput,
        now: float,
        occupancy: OccupancyGrid | None = None,
    ) -> None:
        with self._lock:
            state = self._state
            scan = self._scan_result
            command = self._last_command
            error = self._error
            gate = self._gate_open and self._last_local_safe
            active_provider = self._active_provider
            pending_provider = self._pending_provider
            api_available = self._api_available
        scan_text = scan.traversability if scan else "pending"
        active_occupancy = occupancy or perception.occupancy
        clearance = front_clearance(active_occupancy, self.config.robot)
        provider_text = active_provider
        if pending_provider:
            provider_text += f" -> {pending_provider}"
        lines = [
            f"state={state.value} gate={'open' if gate else 'closed'}",
            f"cmd=({command.linear_x:.2f},{command.angular_z:.2f}) scan={scan_text}",
            f"device={getattr(self._perception, 'device', 'unknown')}",
            f"provider={provider_text}",
            f"fps={self._processing_fps:.0f} inference~{self._last_inference_ms:.0f}ms",
            f"front clear={clearance:.2f}m",
        ]
        if self._last_scan_duration_s and state != RunState.SCANNING:
            lines.append(
                f"scan={self._last_scan_duration_s:.1f}s "
                f"in={self._last_scan_usage.get('input_tokens', 0)} "
                f"out={self._last_scan_usage.get('output_tokens', 0)}"
            )
        if active_provider == "api":
            lines.append("REMOTE API: one current frame is uploaded per scan")
        elif pending_provider == "api":
            lines.append("API pending: next scan will upload one frame")
        if scan:
            lines.append(f"scan: {scan.summary[:70]}")
        if error:
            lines.append(f"error: {error[:80]}")

        if rclpy.ok():
            stamp = self.get_clock().now().to_msg()
            with self._lock:
                active_streams = dict(self._debug_stream_active)
            stream_images = (
                (
                    "camera",
                    self._camera_debug_pub,
                    self._camera_jpeg_pub,
                    lambda: render_camera_view(frame, perception.detections),
                ),
                (
                    "depth",
                    self._depth_debug_pub,
                    self._depth_jpeg_pub,
                    lambda: render_depth_view(perception.depth_m),
                ),
                (
                    "bev",
                    self._bev_debug_pub,
                    self._bev_jpeg_pub,
                    lambda: render_bev_view(active_occupancy),
                ),
            )
            for name, image_pub, jpeg_pub, render in stream_images:
                if (
                    now - self._last_stream_debug_at[name]
                    < self._debug_stream_periods[name] * 0.9
                ):
                    continue
                if active_streams[name] or image_pub.get_subscription_count():
                    self._publish_topic_images(
                        image_pub,
                        jpeg_pub,
                        render(),
                        stamp,
                        encode_jpeg=active_streams[name],
                    )
                    self._last_stream_debug_at[name] = now
        if now - self._last_legacy_debug_at >= 0.4 and rclpy.ok():
            self._last_legacy_debug_at = now
            stamp = self.get_clock().now().to_msg()
            self._publish_topic_image(
                self._debug_pub,
                render_debug(
                    frame,
                    perception.depth_m,
                    active_occupancy,
                    perception.detections,
                    lines,
                ),
                stamp,
            )

        if now - self._last_debug_status_at >= 0.2 and rclpy.ok():
            self._last_debug_status_at = now
            status = {
                "state": state.value,
                "gate_open": gate,
                "active_provider": active_provider,
                "pending_provider": pending_provider,
                "api_available": api_available,
                "effective_next_scan": pending_provider or active_provider,
                "linear_x": round(command.linear_x, 3),
                "angular_z": round(command.angular_z, 3),
                "scan_traversability": scan_text,
                "scan_summary": scan.summary if scan else "",
                "error": error,
                "fps": round(self._processing_fps, 1),
                "inference_ms": round(self._last_inference_ms, 1),
                "front_clearance_m": round(clearance, 2),
                "depth_min_m": round(float(perception.depth_m.min()), 2),
                "depth_max_m": round(float(perception.depth_m.max()), 2),
                "scan_seconds": round(self._last_scan_duration_s, 2),
                "input_tokens": self._last_scan_usage.get("input_tokens", 0),
                "output_tokens": self._last_scan_usage.get("output_tokens", 0),
                "detections": [
                    {
                        "label": detection.label,
                        "confidence": round(detection.confidence, 3),
                    }
                    for detection in perception.detections
                ],
                "device": getattr(self._perception, "device", "unknown"),
            }
            self._debug_status_pub.publish(
                String(
                    data=json.dumps(
                        status, ensure_ascii=False, separators=(",", ":")
                    )
                )
            )

    def _publish_debug_tick(self) -> None:
        with self._lock:
            frame = self._last_frame
            perception = self._last_perception
            occupancy = self._last_occupancy
        if frame is not None and perception is not None:
            self._publish_debug(
                frame, perception, time.monotonic(), occupancy
            )

    def _publish_topic_image(self, publisher, image, stamp) -> None:
        message = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        message.header.stamp = stamp
        message.header.frame_id = "camera_link"
        publisher.publish(message)

    def _publish_topic_images(
        self,
        image_publisher,
        jpeg_publisher,
        image,
        stamp,
        *,
        encode_jpeg: bool,
    ) -> None:
        self._publish_topic_image(image_publisher, image, stamp)
        if not encode_jpeg:
            return
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not ok:
            return
        message = CompressedImage(format="jpeg", data=encoded.tobytes())
        message.header.stamp = stamp
        message.header.frame_id = "camera_link"
        jpeg_publisher.publish(message)

    def _debug_clients_callback(self, message: String) -> None:
        try:
            clients = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not isinstance(clients, dict):
            return
        with self._lock:
            for name in self._debug_stream_active:
                if name in clients:
                    self._debug_stream_active[name] = bool(clients[name])

    def _publish_status(self) -> None:
        if (
            self._scanner is not None
            and self._perception is not None
            and self._active_provider == "local"
            and not self._scanner.health_check()
        ):
            self._fault("local scene scanner service is unavailable")
        with self._lock:
            state = self._state
            error = self._error
            result = self._scan_result
            active_provider = self._active_provider
            pending_provider = self._pending_provider
            api_available = self._api_available
        status = DiagnosticStatus()
        status.name = "vision_avoid/provider"
        status.hardware_id = active_provider
        status.level = (
            DiagnosticStatus.OK
            if state in {RunState.READY, RunState.RUNNING}
            else DiagnosticStatus.WARN
            if state in {RunState.INIT, RunState.SCANNING, RunState.BLOCKED}
            else DiagnosticStatus.ERROR
        )
        status.message = error or state.value
        status.values = [
            self._diagnostic_pair("provider", active_provider),
            self._diagnostic_pair("pending_provider", pending_provider or ""),
            self._diagnostic_pair("api_available", str(api_available).lower()),
            self._diagnostic_pair(
                "model",
                self.config.vlm.local_model
                if active_provider == "local"
                else self.config.vlm.api_model,
            ),
            self._diagnostic_pair("state", state.value),
            self._diagnostic_pair(
                "traversability",
                result.traversability if result else "unknown",
            ),
            self._diagnostic_pair(
                "scan_seconds", f"{self._last_scan_duration_s:.3f}"
            ),
            self._diagnostic_pair(
                "input_tokens",
                str(self._last_scan_usage.get("input_tokens", 0)),
            ),
            self._diagnostic_pair(
                "output_tokens",
                str(self._last_scan_usage.get("output_tokens", 0)),
            ),
        ]
        if rclpy.ok():
            self._status_pub.publish(status)

    @staticmethod
    def _diagnostic_pair(key: str, value: str):
        from diagnostic_msgs.msg import KeyValue

        return KeyValue(key=key, value=value)

    def _trim_api_calls(self, now: float) -> None:
        while self._api_calls and now - self._api_calls[0] >= 3600.0:
            self._api_calls.popleft()

    def _set_api_provider_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        target = "api" if request.data else "local"
        unload_local = False
        with self._lock:
            if target == "api" and not self._api_available:
                response.success = False
                response.message = (
                    "API provider is not configured; set VLM_API_ENABLED, "
                    "base URL, model and API key environment variable"
                )
                return response
            if target == self._active_provider and self._scanner is not None:
                self._pending_provider = None
                self._unload_qwen_after_scan = False
                response.success = True
                response.message = f"{target} provider is already active"
                return response
            if self._pending_provider == target:
                response.success = True
                response.message = (
                    f"{target} provider already pending; effective on next scan"
                )
                return response
            self._provider_change_id += 1
            self._pending_provider = target
            if target == "api":
                if self._scanning:
                    self._unload_qwen_after_scan = True
                else:
                    unload_local = True
            else:
                self._unload_qwen_after_scan = False
        if unload_local:
            self._unload_local_qwen_async()
        response.success = True
        response.message = (
            f"{target} provider accepted; effective on next scene scan"
        )
        return response

    def _unload_local_qwen_async(self) -> None:
        change_id = self._provider_change_id

        def unload() -> None:
            with self._lock:
                if (
                    change_id != self._provider_change_id
                    or self._pending_provider != "api"
                ):
                    return
            try:
                scanner = create_scanner(self._provider_config("local"))
                scanner.unload()
                with self._lock:
                    self._local_qwen_unloaded = True
                self.get_logger().info("local Qwen unloaded after API switch")
            except Exception as exc:
                self.get_logger().warning(
                    f"could not unload local Qwen: {type(exc).__name__}"
                )

        thread = threading.Thread(target=unload, daemon=True)
        with self._lock:
            self._local_qwen_unload_thread = thread
        thread.start()

    def _rescan_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self._perception is None or (
            self._scanner is None and self._pending_provider is None
        ):
            response.success = False
            response.message = "scene scanning is unavailable; check startup diagnostics"
            return response
        with self._lock:
            self._trigger_scan_locked("manual rescan")
        response.success = True
        response.message = "scene scan accepted; vehicle remains stopped until it completes"
        return response

    def _local_rescan_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        if self._active_provider != "local":
            response.success = False
            response.message = "local VLM provider is not active"
            return response
        return self._rescan_callback(request, response)

    def _trigger_scan_locked(self, reason: str) -> None:
        self._scan_generation += 1
        self._gate_open = False
        self._scan_result = None
        self._scan_prior_until = 0.0
        self._need_scan = True
        self._state = RunState.SCANNING
        self._error = reason
        self._last_command = MotionCommand()
        self.get_logger().info(f"scene scan: {reason}")
        self._publish_stop()

    def _fault(self, message: str) -> None:
        now = time.monotonic()
        with self._lock:
            should_log = (
                message != self._last_fault_message
                or now - self._last_fault_log_at >= 10.0
            )
            if should_log:
                self._last_fault_message = message
                self._last_fault_log_at = now
            self._scan_generation += 1
            self._gate_open = False
            self._state = RunState.FAULT
            self._error = message
            self._scan_result = None
            self._last_local_safe = False
            self._last_command = MotionCommand()
        if should_log:
            self.get_logger().error(message)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self._publish_stop()
        return super().destroy_node()


def encode_snapshot(frame_bgr) -> bytes:
    height, width = frame_bgr.shape[:2]
    scale = min(1024.0 / width, 1024.0 / height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    frame = cv2.resize(frame_bgr, target, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("could not encode scene snapshot")
    return encoded.tobytes()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = AvoidanceNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
