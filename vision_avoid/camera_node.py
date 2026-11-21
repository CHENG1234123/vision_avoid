from __future__ import annotations

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def open_video_capture(
    path: str,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("usb_camera")
        self.declare_parameter(
            "video_device",
            "/dev/v4l/by-id/usb-LRCP_USB2.0_LRCP_USB2.0_SN0001-video-index0",
        )
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("frame_id", "camera_link")

        self._path = str(self.get_parameter("video_device").value)
        self._width = int(self.get_parameter("width").value)
        self._height = int(self.get_parameter("height").value)
        self._fps = int(self.get_parameter("fps").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._bridge = CvBridge()
        self._capture: cv2.VideoCapture | None = None
        self._last_reopen = 0.0
        self._pub = self.create_publisher(
            Image, "/camera/image_raw", qos_profile_sensor_data
        )
        self._open()
        self._timer = self.create_timer(1.0 / self._fps, self._tick)

    def _open(self) -> None:
        self._release()
        self._capture = open_video_capture(
            self._path, self._width, self._height, self._fps
        )
        if not self._capture.isOpened():
            self.get_logger().error(f"cannot open camera: {self._path}")
            self._release()
        else:
            self.get_logger().info(
                f"camera open: {self._path} {self._width}x{self._height}@{self._fps}"
            )

    def _tick(self) -> None:
        if self._capture is None:
            now = time.monotonic()
            if now - self._last_reopen >= 2.0:
                self._last_reopen = now
                self._open()
            return
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.get_logger().error("camera frame read failed; reconnecting")
            self._release()
            self._last_reopen = time.monotonic()
            return
        message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        self._pub.publish(message)

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def destroy_node(self) -> bool:
        self._release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                node._release()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
