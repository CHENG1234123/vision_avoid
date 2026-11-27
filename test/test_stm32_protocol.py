import math

import pytest

from vision_avoid.stm32_protocol import (
    CommandSequencer,
    decode_firewater,
    drive_packet,
    stop_packet,
)


def frame(state=4, calls=0, left=0.0, right=0.0, yaw=0.0):
    values = [0.0] * 16
    values[4], values[5], values[8] = left, right, yaw
    values[15] = calls * 100000 + 10 + state
    return "channels:" + ",".join(f"{value:.3f}" for value in values)


class Sink:
    def __init__(self):
        self.packets = []

    def __call__(self, packet):
        self.packets.append(packet)


def armed_sequence(now=1.0):
    sink = Sink()
    seq = CommandSequencer(sink)
    seq.feedback(decode_firewater(frame()), now)
    seq.motion_command(0.06, 0.02, now)
    seq.gate(True, now)
    assert seq.arm(now + 0.01)
    return sink, seq, now + 0.01


def test_firewater_is_exactly_sixteen_finite_channels():
    parsed = decode_firewater(frame(state=1, calls=2, left=0.1, right=0.2, yaw=7.5))
    assert parsed is not None
    assert parsed.state == 1 and parsed.drive_calls == 2
    assert parsed.positions == (0.1, 0.2) and parsed.yaw_deg == 7.5
    for bad in (
        "channels:1,2,3",
        "channels:" + ",".join(["0"] * 15),
        "channels:" + ",".join(["0"] * 15 + ["nan"]),
        "channels:" + ",".join(["0"] * 16) + ",1",
        "other:" + ",".join(["0"] * 16),
    ):
        assert decode_firewater(bad) is None


def test_only_the_existing_dollar_d_format_is_emitted():
    assert drive_packet(0.02, 0.06, 0.10) == b"$D20.0,0.155,0.10\n"
    assert stop_packet() == b"$D0,0.050,0.00\n"
    with pytest.raises(ValueError):
        drive_packet(-0.01, 0.06, 0.0)
    with pytest.raises(ValueError):
        drive_packet(0.02, 0.06, 20.1)


def test_one_short_segment_then_wait_for_chassis_done():
    sink, seq, t = armed_sequence()
    seq.tick(t + 0.01)
    assert sink.packets == [b"$D20.0,0.155,6.18\n"]
    assert seq.in_flight is not None
    seq.tick(t + 0.05)
    assert len(sink.packets) == 1
    seq.feedback(decode_firewater(frame(state=1, calls=1, left=0.052)), t + 0.08)
    seq.tick(t + 0.09)
    assert len(sink.packets) == 1
    seq.feedback(decode_firewater(frame(state=4, calls=1, left=0.052)), t + 0.12)
    seq.tick(t + 0.13)
    seq.tick(t + 0.14)
    assert len(sink.packets) == 2
    assert sink.packets[1].startswith(b"$D20.0,")


def test_gate_close_queues_one_zero_distance_stop_and_holds():
    sink, seq, t = armed_sequence()
    seq.tick(t + 0.01)
    seq.gate(False, t + 0.05)
    seq.tick(t + 0.06)
    assert sink.packets[-1] == stop_packet()
    count = len(sink.packets)
    seq.tick(t + 0.10)
    assert len(sink.packets) == count
    seq.feedback(decode_firewater(frame(state=4, calls=1, left=0.052)), t + 0.12)
    seq.gate(True, t + 0.12)
    seq.motion_command(0.06, 0.0, t + 0.12)
    seq.tick(t + 0.13)
    assert sink.packets[-1].startswith(b"$D20.0,")


def test_disarm_queues_stop_while_action_is_in_flight():
    sink, seq, t = armed_sequence()
    seq.tick(t + 0.01)
    seq.disarm("operator", t + 0.03)
    seq.tick(t + 0.04)
    assert not seq.armed
    assert sink.packets[-1] == stop_packet()
    count = len(sink.packets)
    seq.tick(t + 0.10)
    assert len(sink.packets) == count


def test_invalid_curvature_stops_without_sending_a_motion_segment():
    sink = Sink()
    seq = CommandSequencer(sink)
    seq.feedback(decode_firewater(frame()), 1.0)
    seq.motion_command(0.06, 1.0, 1.0)
    seq.gate(True, 1.0)
    assert seq.arm(1.01)
    seq.tick(1.02)
    assert not seq.armed
    assert not any(packet.startswith(b"$D20") for packet in sink.packets)


def test_arm_requires_idle_board_and_fresh_inputs():
    sink = Sink()
    seq = CommandSequencer(sink)
    seq.feedback(decode_firewater(frame(state=1)), 1.0)
    seq.motion_command(0, 0, 1.0)
    seq.gate(True, 1.0)
    assert not seq.arm(1.01)
    seq.feedback(decode_firewater(frame()), 1.1)
    seq.gate(True, 1.1)
    seq.motion_command(0, 0, 1.1)
    assert seq.arm(1.11)


def test_stale_feedback_or_command_never_moves_vehicle():
    sink = Sink()
    seq = CommandSequencer(sink)
    seq.feedback(decode_firewater(frame()), 1.0)
    seq.motion_command(0.06, 0.0, 1.0)
    seq.gate(True, 1.0)
    assert seq.arm(1.01)
    seq.tick(1.31)
    assert not seq.armed
    assert math.isfinite(seq.feedback_at)
