from __future__ import annotations

import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Vision Avoid Live</title>
  <style>
    :root { color-scheme: dark; --bg:#111419; --panel:#1d222b; --text:#eef2f7;
            --muted:#9ba5b4; --ok:#37d67a; --warn:#ffad33; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; background:var(--bg); color:var(--text);
           font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }
    header { display:flex; align-items:center; justify-content:space-between;
             gap:16px; padding:12px 18px; background:var(--panel);
             border-bottom:1px solid #303745; }
    h1 { margin:0; font-size:18px; letter-spacing:.3px; }
    #status { display:flex; gap:16px; color:var(--muted); white-space:nowrap; }
    #dot { display:inline-block; width:9px; height:9px; border-radius:50%;
           background:var(--warn); margin-right:6px; }
    main { display:grid; place-items:center; padding:14px; }
    img { display:block; width:min(100%,1280px); max-height:calc(100vh - 76px);
          object-fit:contain; background:#000; border-radius:8px;
          box-shadow:0 8px 28px #0008; }
    button { border:0; border-radius:6px; padding:7px 12px; cursor:pointer;
             background:#303846; color:var(--text); }
    @media (max-width:700px) {
      #status .optional { display:none; }
      header { padding:10px 12px; }
      main { padding:8px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Vision Avoid Live</h1>
    <div id="status">
      <span><span id="dot"></span><span id="health">waiting</span></span>
      <span class="optional">STREAM FPS <b id="fps">0</b></span>
      <span class="optional">Frame age <b id="age">--</b></span>
      <button onclick="document.documentElement.requestFullscreen()">Fullscreen</button>
    </div>
  </header>
  <main><img id="stream" src="/stream.mjpg" alt="Robot debug stream"></main>
  <script>
    const health = document.getElementById("health");
    const dot = document.getElementById("dot");
    const fps = document.getElementById("fps");
    const age = document.getElementById("age");
    async function refresh() {
      try {
        const r = await fetch("/api/status", {cache:"no-store"});
        const s = await r.json();
        fps.textContent = Math.round(s.fps);
        age.textContent = s.frame_age_ms == null ? "--" : `${s.frame_age_ms.toFixed(0)} ms`;
        const live = s.frame_age_ms != null && s.frame_age_ms < 1500;
        health.textContent = live ? "LIVE" : "NO FRAME";
        dot.style.background = live ? "var(--ok)" : "var(--warn)";
      } catch (_) {
        health.textContent = "DISCONNECTED";
        dot.style.background = "var(--warn)";
      }
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class DebugStreamState:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: bytes | None = None
        self._sequence = 0
        self._frame_times: collections.deque[float] = collections.deque()
        self._received = 0
        self._clients = 0

    def publish(self, frame: bytes) -> None:
        now = time.monotonic()
        with self._condition:
            self._frame = frame
            self._sequence += 1
            self._received += 1
            self._frame_times.append(now)
            while self._frame_times and now - self._frame_times[0] > 5.0:
                self._frame_times.popleft()
            self._condition.notify_all()

    def wait_for_frame(
        self, after_sequence: int, timeout: float = 2.0
    ) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame is not None and self._sequence != after_sequence,
                timeout=timeout,
            )
            if self._frame is None or self._sequence == after_sequence:
                return None
            return self._sequence, self._frame

    def status(self) -> dict[str, float | int | None]:
        with self._condition:
            fps = 0.0
            if len(self._frame_times) >= 2:
                duration = self._frame_times[-1] - self._frame_times[0]
                if duration > 0:
                    fps = (len(self._frame_times) - 1) / duration
            age_ms = (
                (time.monotonic() - self._frame_times[-1]) * 1000.0
                if self._frame_times
                else None
            )
            return {
                "live": self._frame is not None,
                "fps": round(fps, 1),
                "frame_age_ms": None if age_ms is None else round(age_ms, 1),
                "frames_received": self._received,
                "clients": self._clients,
            }

    def client_opened(self) -> None:
        with self._condition:
            self._clients += 1

    def client_closed(self) -> None:
        with self._condition:
            self._clients = max(0, self._clients - 1)


class DebugRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        hub: DebugStreamState = self.server.hub  # type: ignore[attr-defined]
        if self.path in {"/", "/index.html"}:
            self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            body = json.dumps(hub.status()).encode("utf-8")
            self._send_bytes(body, "application/json")
        elif self.path == "/health":
            self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
        elif self.path == "/stream.mjpg":
            self._stream(hub)
        else:
            self.send_error(404)

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, hub: DebugStreamState) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_sequence = 0
        hub.client_opened()
        try:
            while True:
                item = hub.wait_for_frame(last_sequence)
                if item is None:
                    continue
                last_sequence, jpeg = item
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            hub.client_closed()

    def log_message(self, format: str, *args) -> None:
        del format, args


class DebugWebNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_debug_web")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8080)
        self.declare_parameter("jpeg_quality", 82)
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._quality = int(self.get_parameter("jpeg_quality").value)

        self._bridge = CvBridge()
        self._hub = DebugStreamState()
        self._server: ThreadingHTTPServer | None = None
        try:
            self._server = ThreadingHTTPServer((host, port), DebugRequestHandler)
            self._server.daemon_threads = True
            self._server.hub = self._hub  # type: ignore[attr-defined]
        except OSError as exc:
            self.get_logger().error(f"cannot start debug web server: {exc}")
        if self._server is not None:
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._server_thread.start()
            self.get_logger().info(f"debug dashboard: http://{host}:{port}")
        self.create_subscription(
            Image,
            "/vision/debug_image",
            self._image_callback,
            qos_profile_sensor_data,
        )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
            )
            if ok:
                self._hub.publish(encoded.tobytes())
        except Exception as exc:
            self.get_logger().error(
                f"debug frame conversion failed: {type(exc).__name__}"
            )

    def destroy_node(self) -> bool:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DebugWebNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
