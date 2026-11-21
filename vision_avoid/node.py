from __future__ import annotations

import collections
import subprocess
import threading
import time
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
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from vision_avoid.config import load_config
from vision_avoid.geometry import (
    CommandSelector,
    MotionCommand,
    OccupancyGrid,
    apply_scan_constraints,
    front_clearance,
)
from vision_avoid.perception import LocalPerception, PerceptionOutput, render_debug
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
        self._last_debug_heartbeat_at = 0.0
        self._api_calls: collections.deque[float] = collections.deque()

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._debug_pub = self.create_publisher(
            Image, "/vision/debug_image", qos_profile_sensor_data
        )
        self._gate_pub = self.create_publisher(Bool, "/vision/scan_gate_open", 10)
        self._status_pub = self.create_publisher(
            DiagnosticStatus, "/vision/provider_status", 10
        )

        image_group = MutuallyExclusiveCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()
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
        self._status_timer = self.create_timer(
            1.0, self._publish_status, callback_group=status_group
        )
        self._rescan_srv = self.create_service(
            Trigger, "/vision/rescan_scene", self._rescan_callback
        )
        self._local_rescan_srv = self.create_service(
            Trigger, "/vision/rescan_local_vlm", self._local_rescan_callback
        )

        self._scene_detector = SceneChangeDetector(self.config.scene_change)
        self._selector = CommandSelector(self.config.planner)
        self._scanner: SceneScanner | None = None
        self._perception: LocalPerception | None = None
        self._initialize_runtime()
        if self._state == RunState.FAULT:
            self.get_logger().error("vision_avoid is running with motion disabled")
        else:
            self.get_logger().info(
                f"vision_avoid ready: profile={self.config.profile} "
                f"provider={self.config.vlm.provider} drive_enabled={self.drive_enabled}"
            )

    def _initialize_runtime(self) -> None:
        try:
            self._scanner = create_scanner(self.config.vlm)
            if not self._scanner.health_check():
                raise SceneScanError("scene scanner health check failed")
            self._perception = LocalPerception(self.config)
            if self.config.vlm.provider == "local":
                self._scanner.warmup()
                self._check_gpu_headroom()
        except Exception as exc:
            self._state = RunState.FAULT
            self._error = f"startup failed: {type(exc).__name__}: {exc}"
            self.get_logger().error(self._error)

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
            scanning = self._scanning

        if scanning:
            if (
                self._last_perception is not None
                and now - self._last_debug_heartbeat_at >= 0.2
            ):
                self._publish_debug(frame, self._last_perception, now)
                self._last_debug_heartbeat_at = now
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
        self._last_perception = perception

        observation = self._scene_detector.update(frame, now)
        if observation.event == SceneEvent.QUALITY_REJECTED:
            self._fault(
                "camera quality rejected "
                f"(brightness={observation.brightness:.1f}, "
                f"contrast={observation.contrast:.1f}, "
                f"sharpness={observation.sharpness:.1f})"
            )
            self._publish_debug(frame, perception, now)
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

        self._publish_debug(frame, perception, now, occupancy)

    def _start_scan_if_needed(self, frame, now: float) -> None:
        with self._lock:
            if not self._need_scan or self._scanning:
                return
            if self._scanner is None:
                return
            if self.config.vlm.provider == "api":
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
            target=self._scan_worker, args=(jpeg, generation), daemon=True
        )
        worker.start()

    def _scan_worker(self, jpeg: bytes, generation: int) -> None:
        started_at = time.monotonic()
        try:
            if self._scanner is None:
                raise SceneScanError("scene scanner is unavailable")
            result = self._scanner.scan(jpeg)
            with self._lock:
                self._pending_scan = result
                self._pending_scan_generation = generation
                self._pending_scan_error = ""
                self._last_scan_duration_s = time.monotonic() - started_at
                self._last_scan_usage = dict(
                    getattr(self._scanner, "last_usage", {})
                )
            self.get_logger().info(
                f"scene scan complete: {result.traversability} "
                f"confidence={result.confidence:.2f}"
            )
        except Exception as exc:
            with self._lock:
                self._pending_scan = None
                self._pending_scan_error = f"{type(exc).__name__}: {exc}"
                self._pending_scan_generation = generation
            self.get_logger().warning(f"scene scan failed: {type(exc).__name__}")
        finally:
            with self._lock:
                if self._active_scan_generation == generation:
                    self._scanning = False

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
        del now
        with self._lock:
            state = self._state
            scan = self._scan_result
            command = self._last_command
            error = self._error
            gate = self._gate_open and self._last_local_safe
        scan_text = scan.traversability if scan else "pending"
        clearance = front_clearance(
            occupancy or perception.occupancy, self.config.robot
        )
        lines = [
            f"state={state.value} gate={'open' if gate else 'closed'}",
            f"cmd=({command.linear_x:.2f},{command.angular_z:.2f}) scan={scan_text}",
            f"device={getattr(self._perception, 'device', 'unknown')}",
            f"provider={self.config.vlm.provider}",
            f"fps={self._processing_fps:.0f} inference~{self._last_inference_ms:.0f}ms",
            f"front clear={clearance:.2f}m",
        ]
        if self._last_scan_duration_s and state != RunState.SCANNING:
            lines.append(
                f"scan={self._last_scan_duration_s:.1f}s "
                f"in={self._last_scan_usage.get('input_tokens', 0)} "
                f"out={self._last_scan_usage.get('output_tokens', 0)}"
            )
        if self.config.vlm.provider == "api":
            lines.append("REMOTE API: one current frame is uploaded per scan")
        if scan:
            lines.append(f"scan: {scan.summary[:70]}")
        if error:
            lines.append(f"error: {error[:80]}")
        debug = render_debug(
            frame,
            perception.depth_m,
            occupancy or perception.occupancy,
            perception.detections,
            lines,
        )
        message = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        if rclpy.ok():
            self._debug_pub.publish(message)

    def _publish_status(self) -> None:
        if (
            self._scanner is not None
            and self._perception is not None
            and self.config.vlm.provider == "local"
            and not self._scanner.health_check()
        ):
            self._fault("local scene scanner service is unavailable")
        with self._lock:
            state = self._state
            error = self._error
            result = self._scan_result
        status = DiagnosticStatus()
        status.name = "vision_avoid/provider"
        status.hardware_id = self.config.vlm.provider
        status.level = (
            DiagnosticStatus.OK
            if state in {RunState.READY, RunState.RUNNING}
            else DiagnosticStatus.WARN
            if state in {RunState.INIT, RunState.SCANNING, RunState.BLOCKED}
            else DiagnosticStatus.ERROR
        )
        status.message = error or state.value
        status.values = [
            self._diagnostic_pair("provider", self.config.vlm.provider),
            self._diagnostic_pair(
                "model",
                self.config.vlm.local_model
                if self.config.vlm.provider == "local"
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

    def _rescan_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if self._scanner is None or self._perception is None:
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
        if self.config.vlm.provider != "local":
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
