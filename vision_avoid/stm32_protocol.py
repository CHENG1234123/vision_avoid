"""The STM32 project's existing USART1 protocol.

The controller emits 16-channel FireWater telemetry and accepts one $D action
at a time. This module never changes or reconfigures the controller.
"""
from dataclasses import dataclass
import fcntl
import math
import os
import termios
import tty

from vision_avoid.vehicle import WHEEL_DIAMETER, steering

DONE_STATES = frozenset((0, 4))


@dataclass(frozen=True)
class Feedback:
    values: tuple[float, ...]

    @property
    def state(self) -> int:
        return int(round(self.values[15])) % 10

    @property
    def done(self) -> bool:
        return self.state in DONE_STATES

    @property
    def moving(self) -> bool:
        return not self.done

    @property
    def drive_calls(self) -> int:
        return max(0, int(round(self.values[15])) // 100000)

    @property
    def positions(self) -> tuple[float, float]:
        return (round(self.values[4], 3), round(self.values[5], 3))

    @property
    def progress(self) -> tuple[int, float, float]:
        return (self.drive_calls, *self.positions)

    @property
    def yaw_deg(self) -> float:
        return self.values[8]


def decode_firewater(line: str) -> Feedback | None:
    """Parse the firmware's "channels:" FireWater frame, with no extras."""
    if not line.startswith("channels:"):
        return None
    fields = line[9:].split(",")
    if len(fields) != 16:
        return None
    try:
        values = tuple(float(field) for field in fields)
    except (ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return Feedback(values)


def drive_packet(distance_m: float, speed_mps: float, steer_deg: float) -> bytes:
    """Format the exact $D command accepted by the existing firmware."""
    if not all(math.isfinite(x) for x in (distance_m, speed_mps, steer_deg)):
        raise ValueError("command values must be finite")
    if distance_m < 0 or not 0 < speed_mps <= 1.0 or abs(steer_deg) > 18.0:
        raise ValueError("command outside forward-only limits")
    return (
        f"$D{distance_m * 1000.0:.1f},"
        f"{speed_mps / (math.pi * WHEEL_DIAMETER):.3f},"
        f"{steer_deg:.2f}\n"
    ).encode("ascii")


def stop_packet() -> bytes:
    return b"$D0,0.050,0.00\n"


class SerialPort:
    """Linux 115200 8N1 port using the same USART1 pins as the STM32 project."""

    def __init__(self, path: str, *, writable: bool = True):
        flags = os.O_RDWR if writable else os.O_RDONLY
        self.fd = os.open(path, flags | os.O_NONBLOCK | os.O_NOCTTY)
        self.buffer = b""
        try:
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            tty.setraw(self.fd)
            attrs = termios.tcgetattr(self.fd)
            attrs[4] = attrs[5] = termios.B115200
            attrs[2] &= ~termios.CRTSCTS
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            os.close(self.fd)
            raise

    def read_lines(self) -> list[str]:
        try:
            chunk = os.read(self.fd, 4096)
        except BlockingIOError:
            return []
        if not chunk:
            raise OSError("serial disconnected")
        parts = (self.buffer + chunk).split(b"\n")
        self.buffer = parts.pop()
        if len(self.buffer) > 512 or len(parts) > 8:
            self.buffer = b""
            raise OSError("serial backlog or invalid framing")
        return [x.rstrip(b"\r").decode("ascii", "strict") for x in parts]

    def write(self, data: bytes) -> None:
        if os.write(self.fd, data) != len(data):
            raise OSError("incomplete serial write")

    def close(self) -> None:
        os.close(self.fd)


@dataclass
class InFlight:
    sent_at: float
    progress: tuple[int, float, float]
    steer_deg: float
    accepted: bool = False


class CommandSequencer:
    """One short $D action at a time; a stop is queued as the next action."""

    def __init__(
        self,
        send,
        *,
        segment_m: float = 0.02,
        feedback_timeout_s: float = 0.30,
        command_timeout_s: float = 0.30,
        accept_timeout_s: float = 0.70,
        action_timeout_s: float = 2.00,
    ):
        if not 0.005 <= segment_m <= 0.05:
            raise ValueError("segment_m must be between 5 and 50 mm")
        self.send = send
        self.segment_m = segment_m
        self.feedback_timeout_s = feedback_timeout_s
        self.command_timeout_s = command_timeout_s
        self.accept_timeout_s = accept_timeout_s
        self.action_timeout_s = action_timeout_s
        self.feedback_value: Feedback | None = None
        self.feedback_at = -math.inf
        self.command = (0.0, 0.0)
        self.command_at = -math.inf
        self.gate_clear = False
        self.gate_at = -math.inf
        self.armed = False
        self.reason = "disarmed"
        self.stop_sent = False
        self.in_flight: InFlight | None = None
        self.segments_sent = 0

    def fresh(self, now: float) -> Feedback | None:
        if now - self.feedback_at > self.feedback_timeout_s:
            return None
        return self.feedback_value

    def feedback(self, value: Feedback | None, now: float) -> None:
        if value is not None:
            self.feedback_value = value
            self.feedback_at = now

    def motion_command(self, linear_x: float, angular_z: float, now: float) -> None:
        if not all(math.isfinite(x) for x in (linear_x, angular_z)):
            self.command_at = -math.inf
            return
        self.command = (linear_x, angular_z)
        self.command_at = now

    def gate(self, clear: bool, now: float) -> None:
        self.gate_clear = bool(clear)
        self.gate_at = now

    def arm(self, now: float) -> bool:
        fb = self.fresh(now)
        if (
            self.armed
            or fb is None
            or not fb.done
            or not self.gate_clear
            or now - self.gate_at > self.command_timeout_s
            or now - self.command_at > self.command_timeout_s
        ):
            return False
        self.armed = True
        self.reason = "armed; waiting for a motion segment"
        self.stop_sent = False
        self.in_flight = None
        return True

    def disarm(self, reason: str, now: float) -> None:
        if self.armed:
            self._queue_stop(reason, now)
        self.armed = False
        self.reason = reason

    def _queue_stop(self, reason: str, now: float) -> None:
        self.reason = reason
        queued_action = self.in_flight is not None
        self.in_flight = None
        if self.stop_sent:
            return
        fb = self.fresh(now)
        if fb is not None and (queued_action or not fb.done):
            try:
                self.send(stop_packet())
            except OSError as exc:
                self.reason = f"{reason}; serial write failed: {exc}"
        self.stop_sent = True

    def _fail(self, reason: str, now: float) -> None:
        self._queue_stop(reason, now)
        self.armed = False
        self.reason = reason

    def tick(self, now: float) -> None:
        fb = self.fresh(now)
        if self.armed and fb is None:
            self.in_flight = None
            self.armed = False
            self.reason = "feedback timeout; disarmed"
            return
        if not self.armed:
            return
        if now - self.command_at > self.command_timeout_s:
            self._queue_stop("command timeout", now)
            return
        if not self.gate_clear or now - self.gate_at > self.command_timeout_s:
            self._queue_stop("vision gate is closed", now)
            return
        linear_x, angular_z = self.command
        if linear_x == 0.0 and angular_z == 0.0:
            self._queue_stop("zero motion command", now)
            return

        if self.in_flight is not None:
            flight = self.in_flight
            if fb.progress != flight.progress or fb.moving:
                flight.accepted = True
            if fb.done and (flight.accepted or fb.progress != flight.progress):
                self.in_flight = None
                self.stop_sent = False
                self.reason = "segment complete"
            elif not flight.accepted and now - flight.sent_at > self.accept_timeout_s:
                self._fail("controller did not accept the $D segment", now)
            elif now - flight.sent_at > self.action_timeout_s:
                self._fail("segment timed out", now)
            return

        if not fb.done:
            return
        self.stop_sent = False
        try:
            angle = steering(linear_x, angular_z)
            packet = drive_packet(self.segment_m, linear_x, angle)
        except ValueError as exc:
            self._fail(str(exc), now)
            return
        try:
            self.send(packet)
        except OSError as exc:
            self._fail(f"serial write failed: {exc}", now)
            return
        self.in_flight = InFlight(now, fb.progress, angle)
        self.segments_sent += 1
        self.reason = "segment sent"

    def status(self, now: float) -> dict:
        fb = self.fresh(now)
        return {
            "armed": self.armed,
            "state": "disarmed" if not self.armed else (
                "segment" if self.in_flight is not None else "waiting"
            ),
            "reason": self.reason,
            "feedback_fresh": fb is not None,
            "board_state": None if fb is None else fb.state,
            "drive_calls": None if fb is None else fb.drive_calls,
            "segment_mm": self.segment_m * 1000.0,
            "segments_sent": self.segments_sent,
        }
