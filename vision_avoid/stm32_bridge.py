"""Upper-computer bridge using only the STM32 project's existing USART1 protocol."""
from __future__ import annotations

import json
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from vision_avoid.stm32_protocol import (
    CommandSequencer,
    SerialPort,
    decode_firewater,
)


class Stm32Bridge(Node):
    def __init__(self) -> None:
        super().__init__("stm32_bridge")
        self.power = bool(self.declare_parameter("drive_enabled", False).value)
        path = str(self.declare_parameter("serial_device", "").value)
        segment_m = float(self.declare_parameter("segment_m", 0.02).value)
        if not path:
            raise ValueError("serial_device is required")
        self.port = SerialPort(path, writable=self.power)
        self.sequencer = CommandSequencer(self.port.write, segment_m=segment_m)
        self.invalid_frames = 0
        self.last_serial_error = ""
        self.create_subscription(Twist, "/cmd_vel", self._command, 1)
        self.create_subscription(Bool, "/vision/scan_gate_open", self._gate, 1)
        self.create_service(SetBool, "/vehicle/arm", self._arm)
        self.status_pub = self.create_publisher(String, "/vehicle/status", 1)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"STM32 bridge on {path}; powered={self.power}; segment={segment_m * 1000:.0f} mm"
        )

    def _command(self, message: Twist) -> None:
        extra = (
            message.linear.y,
            message.linear.z,
            message.angular.x,
            message.angular.y,
        )
        if any(extra):
            self.sequencer.motion_command(float("nan"), float("nan"), time.monotonic())
        else:
            self.sequencer.motion_command(
                float(message.linear.x), float(message.angular.z), time.monotonic()
            )

    def _gate(self, message: Bool) -> None:
        self.sequencer.gate(bool(message.data), time.monotonic())

    def _arm(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        now = time.monotonic()
        if not request.data:
            self.sequencer.disarm("operator disarmed", now)
            self.sequencer.tick(now)
            response.success = True
            response.message = "disarmed"
        elif not self.power:
            response.success = False
            response.message = "drive_enabled is false"
        elif self.sequencer.arm(now):
            response.success = True
            response.message = "armed; waiting for fresh safe command"
        else:
            response.success = False
            response.message = "needs fresh idle telemetry, gate and zero-speed command"
        return response

    def _tick(self) -> None:
        now = time.monotonic()
        try:
            for line in self.port.read_lines():
                feedback = decode_firewater(line)
                if feedback is None:
                    self.invalid_frames += 1
                    if self.invalid_frames >= 5:
                        self.sequencer.disarm("invalid FireWater frames", now)
                else:
                    self.invalid_frames = 0
                    self.sequencer.feedback(feedback, now)
        except (OSError, UnicodeError) as exc:
            self.last_serial_error = str(exc)
            if self.sequencer.armed:
                self.sequencer.disarm(f"serial read failed: {exc}", now)
        self.sequencer.tick(now)
        payload = self.sequencer.status(now)
        payload.update(power=self.power, invalid_frames=self.invalid_frames)
        feedback = self.sequencer.fresh(now)
        if feedback is not None:
            payload.update(
                board_armed=None,
                left_rps=feedback.values[0],
                right_rps=feedback.values[2],
                yaw_deg=feedback.yaw_deg,
                channels=list(feedback.values),
            )
        if self.last_serial_error:
            payload["serial_error"] = self.last_serial_error
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def destroy_node(self) -> bool:
        now = time.monotonic()
        if self.sequencer.armed or self.sequencer.in_flight is not None:
            self.sequencer.disarm("bridge shutdown", now)
            self.sequencer.tick(now)
        try:
            self.port.close()
        except OSError:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Stm32Bridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
