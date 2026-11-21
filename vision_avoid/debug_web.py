from __future__ import annotations

import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from urllib.parse import urlparse

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import SetBool


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>小车视觉避障</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#f1f5f9; --surface:#fff; --surface-2:#f8fafc; --text:#172230;
      --muted:#637083; --line:#d8e0ea; --primary:#175fc7;
      --success:#137a4b; --warning:#a45d00; --danger:#bc3434;
      --shadow:0 8px 24px #20345112;
    }
    * { box-sizing:border-box; }
    body {
      margin:0; min-height:100vh; background:var(--bg); color:var(--text);
      font:15px/1.45 Inter,system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif;
    }
    .topbar {
      position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap;
      align-items:center; gap:14px 22px; padding:12px 22px; background:#fffffff2;
      border-bottom:1px solid var(--line); backdrop-filter:blur(12px);
    }
    h1 { margin:0; font-size:20px; letter-spacing:.2px; }
    .subtitle { color:var(--muted); font-size:12px; margin-top:2px; }
    .spacer { flex:1; }
    .badges,.health,.provider-switch { display:flex; align-items:center; gap:8px; }
    .badge,.health-chip {
      border:1px solid var(--line); background:var(--surface-2); border-radius:999px;
      padding:5px 10px; font-size:12px; font-weight:650; white-space:nowrap;
    }
    .badge[data-state="RUNNING"],.badge[data-state="READY"] {
      color:var(--success); border-color:#9bd3b6; background:#edf9f3;
    }
    .badge[data-state="SCANNING"] {
      color:var(--warning); border-color:#e7bd7c; background:#fff8eb;
    }
    .badge[data-state="BLOCKED"] {
      color:#9a5200; border-color:#e6b26c; background:#fff5e7;
    }
    .badge[data-state="FAULT"] {
      color:var(--danger); border-color:#e4a1a1; background:#fff0f0;
    }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--warning); }
    .dot.live { background:var(--success); box-shadow:0 0 0 4px #137a4b1a; }
    .provider-switch {
      padding:3px; border:1px solid var(--line); border-radius:9px;
      background:var(--surface-2);
    }
    .provider-switch button {
      border:0; border-radius:6px; padding:6px 10px; cursor:pointer;
      color:var(--muted); background:transparent; font-weight:650;
    }
    .provider-switch button.active {
      background:var(--primary); color:#fff; box-shadow:0 2px 8px #175fc72e;
    }
    .provider-switch button:disabled { cursor:not-allowed; opacity:.45; }
    button.fullscreen {
      border:1px solid var(--line); border-radius:7px; padding:6px 10px;
      background:#fff; color:var(--text); cursor:pointer;
    }
    main { max-width:1600px; margin:auto; padding:16px 20px 28px; }
    .layout {
      display:grid; grid-template-columns:minmax(0,2fr) minmax(320px,1fr);
      gap:16px; align-items:start;
    }
    .side { display:grid; gap:16px; }
    .card {
      background:var(--surface); border:1px solid var(--line); border-radius:12px;
      box-shadow:var(--shadow); overflow:hidden;
    }
    .card-head {
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      min-height:42px; padding:9px 13px; border-bottom:1px solid var(--line);
    }
    .card-title { font-size:14px; font-weight:720; }
    .card-note { color:var(--muted); font-size:12px; }
    .media {
      display:block; width:100%; background:#e5eaf0; object-fit:contain;
    }
    .media.paused { visibility:hidden; }
    .camera-view { aspect-ratio:16/9; }
    .small-view { aspect-ratio:16/9; }
    .detections {
      display:flex; flex-wrap:wrap; gap:7px; padding:10px 13px;
      min-height:42px; border-top:1px solid var(--line);
    }
    .detection {
      padding:4px 8px; border-radius:999px; background:#edf5ff;
      border:1px solid #c6ddfb; color:#164c96; font-size:12px; font-weight:650;
    }
    .depth-meta {
      display:flex; align-items:center; gap:10px; padding:9px 12px;
      border-top:1px solid var(--line); color:var(--muted); font-size:12px;
    }
    .gradient {
      flex:1; height:8px; border-radius:999px;
      background:linear-gradient(90deg,#7a0000,#ff9d00,#f4f100,#21c7bb,#11149c);
    }
    .metrics {
      display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      gap:12px; margin-top:16px;
    }
    .metric {
      background:var(--surface); border:1px solid var(--line); border-radius:10px;
      padding:12px 14px; box-shadow:var(--shadow);
    }
    .metric-label { color:var(--muted); font-size:12px; margin-bottom:4px; }
    .metric-value { font-size:23px; line-height:1.1; font-weight:750; }
    .metric-unit { color:var(--muted); font-size:12px; margin-left:3px; }
    .detail-grid {
      display:grid; grid-template-columns:3fr 2fr; gap:12px; margin-top:16px;
    }
    .detail { padding:13px 15px; min-height:106px; }
    .detail p { margin:6px 0 0; white-space:pre-wrap; }
    .alert {
      display:none; margin-top:12px; padding:11px 14px; border-radius:9px;
      background:#fff0f0; border:1px solid #e3a0a0; color:#982a2a;
    }
    .alert.show { display:block; }
    .pending {
      display:none; margin-top:10px; padding:9px 12px; border-radius:8px;
      background:#fff7e8; border:1px solid #ebc982; color:#8a5100; font-size:13px;
    }
    .pending.show { display:block; }
    dialog {
      border:0; border-radius:12px; padding:0; width:min(420px,calc(100vw - 32px));
      box-shadow:0 24px 70px #16243b45;
    }
    dialog::backdrop { background:#18223080; }
    .dialog-body { padding:20px; }
    .dialog-title { margin:0 0 8px; font-size:18px; }
    .dialog-actions {
      display:flex; justify-content:flex-end; gap:8px; padding:12px 20px 18px;
    }
    .dialog-actions button {
      border:1px solid var(--line); border-radius:7px; padding:7px 12px;
      background:#fff; cursor:pointer;
    }
    .dialog-actions .primary { background:var(--primary); color:#fff; border-color:var(--primary); }
    .toast {
      position:fixed; right:18px; bottom:18px; max-width:360px; padding:10px 14px;
      color:#fff; background:#253247; border-radius:8px; opacity:0;
      transform:translateY(8px); transition:.18s ease; pointer-events:none;
    }
    .toast.show { opacity:1; transform:none; }
    @media (max-width:900px) {
      .topbar { padding:10px 12px; gap:10px; }
      .spacer { display:none; }
      .health { order:3; width:100%; flex-wrap:wrap; }
      main { padding:10px 10px 24px; }
      .layout { grid-template-columns:1fr; }
      .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .detail-grid { grid-template-columns:1fr; }
      h1 { font-size:18px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>小车视觉避障</h1>
      <div class="subtitle">单摄像头 · 本地实时感知 · 新场景语义扫描</div>
    </div>
    <div class="provider-switch" aria-label="视觉模型提供方">
      <button id="providerLocal" type="button">本地 Qwen</button>
      <button id="providerApi" type="button">云端 API</button>
    </div>
    <div class="badges">
      <span id="stateBadge" class="badge" data-state="INIT">INIT</span>
      <span id="gateBadge" class="badge">门禁关闭</span>
    </div>
    <div class="spacer"></div>
    <div class="health">
      <span class="health-chip">
        <span id="liveDot" class="dot"></span><span id="liveText">等待画面</span>
      </span>
      <span class="health-chip">感知 <b id="perceptionFps">--</b> FPS</span>
      <span class="health-chip">帧龄 <b id="frameAge">--</b></span>
      <button class="fullscreen" type="button" id="fullscreen">全屏</button>
    </div>
  </header>
  <main>
    <div id="pending" class="pending" aria-live="polite"></div>
    <section class="layout">
      <article class="card">
        <div class="card-head">
          <span class="card-title">摄像头与目标识别</span>
          <span class="card-note" id="cameraNote">等待图像</span>
        </div>
        <img id="camera" class="media camera-view" data-src="/stream/camera.mjpg" alt="摄像头与检测结果">
        <div id="detections" class="detections"></div>
      </article>
      <div class="side">
        <article class="card">
          <div class="card-head">
            <span class="card-title">单目深度</span><span class="card-note">GT 色阶</span>
          </div>
          <img id="depth" class="media small-view" data-src="/stream/depth.mjpg" alt="单目深度图">
          <div class="depth-meta">
            <span>近处 红</span><span class="gradient"></span><span>远处 蓝</span>
            <span id="depthRange">--</span>
          </div>
        </article>
        <article class="card">
          <div class="card-head">
            <span class="card-title">障碍俯视图</span><span class="card-note">前方 ↑</span>
          </div>
          <img id="bev" class="media small-view" data-src="/stream/bev.mjpg" alt="障碍俯视图">
        </article>
      </div>
    </section>
    <section class="metrics" aria-live="polite">
      <div class="metric"><div class="metric-label">线速度</div><div>
        <span id="linearSpeed" class="metric-value">--</span><span class="metric-unit">m/s</span>
      </div></div>
      <div class="metric"><div class="metric-label">角速度</div><div>
        <span id="angularSpeed" class="metric-value">--</span><span class="metric-unit">rad/s</span>
      </div></div>
      <div class="metric"><div class="metric-label">前向净空</div><div>
        <span id="clearance" class="metric-value">--</span><span class="metric-unit">m</span>
      </div></div>
      <div class="metric"><div class="metric-label">推理耗时</div><div>
        <span id="inference" class="metric-value">--</span><span class="metric-unit">ms</span>
      </div></div>
      <div class="metric">
        <div class="metric-label">Qwen 结论</div>
        <div id="scanResult" class="metric-value">--</div>
      </div>
      <div class="metric"><div class="metric-label">扫描耗时</div><div>
        <span id="scanTime" class="metric-value">--</span><span class="metric-unit">s</span>
      </div></div>
      <div class="metric">
        <div class="metric-label">Token</div><div id="tokens" class="metric-value">--</div>
      </div>
      <div class="metric">
        <div class="metric-label">计算设备</div><div id="device" class="metric-value">--</div>
      </div>
    </section>
    <div id="alert" class="alert" role="alert"></div>
    <section class="detail-grid">
      <article class="card detail">
        <div class="card-title">Qwen 场景摘要</div>
        <p id="scanSummary">尚无扫描结果</p>
      </article>
      <article class="card detail">
        <div class="card-title">当前模型</div>
        <p><b id="activeProvider">--</b></p>
        <div class="card-note" id="providerHint"></div>
      </article>
    </section>
  </main>
  <dialog id="apiDialog">
    <div class="dialog-body">
      <h2 class="dialog-title">切换到云端 API？</h2>
      <p>下一次新环境扫描会上传当前帧，并可能产生 API 费用。切换将在下一次扫描时生效。</p>
    </div>
    <div class="dialog-actions">
      <button type="button" id="apiCancel">取消</button>
      <button type="button" class="primary" id="apiConfirm">确认切换</button>
    </div>
  </dialog>
  <div id="toast" class="toast" aria-live="polite"></div>
  <script>
    const $ = (id) => document.getElementById(id);
    let latest = null;
    let apiConfirming = false;
    const streamIds = ["camera", "depth", "bev"];
    const streamVisible = Object.fromEntries(streamIds.map((id) => [id, false]));
    const streamActive = Object.fromEntries(streamIds.map((id) => [id, false]));

    function syncStreams() {
      for (const id of streamIds) {
        const image = $(id);
        const active = streamVisible[id] && !document.hidden;
        if (active === streamActive[id]) continue;
        streamActive[id] = active;
        image.classList.toggle("paused", !active);
        if (active) {
          image.src = `${image.dataset.src}?t=${Date.now()}`;
        } else {
          image.removeAttribute("src");
        }
      }
    }

    const streamObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) streamVisible[entry.target.id] = entry.isIntersecting;
      syncStreams();
    }, {rootMargin:"160px 0px", threshold:0.01});
    for (const id of streamIds) streamObserver.observe($(id));
    document.addEventListener("visibilitychange", syncStreams);

    function setText(id, value) {
      $(id).textContent = value == null || value === "" ? "--" : String(value);
    }

    function updateProviderButtons(status) {
      const active = status.active_provider || "local";
      const pending = status.pending_provider || null;
      const control = Boolean(status.control_enabled);
      $("providerLocal").classList.toggle("active", active === "local");
      $("providerApi").classList.toggle("active", active === "api");
      $("providerLocal").disabled = !control;
      $("providerApi").disabled = !control || !status.api_available;
      $("providerLocal").setAttribute("aria-pressed", active === "local");
      $("providerApi").setAttribute("aria-pressed", active === "api");
      const note = $("pending");
      if (pending) {
        const label = pending === "api" ? "云端 API" : "本地 Qwen";
        note.textContent = `待切换到${label}，下一次扫描生效`;
        note.classList.add("show");
      } else if (!status.api_available) {
        note.textContent = "云端 API 未配置，只可使用本地 Qwen";
        note.classList.add("show");
      } else if (!control) {
        note.textContent = "页面不是通过本机回环地址打开，Provider 切换已锁定";
        note.classList.add("show");
      } else {
        note.classList.remove("show");
      }
      $("activeProvider").textContent =
        active === "api" ? "云端 API" : "本地 Qwen 4B";
      $("providerHint").textContent = pending
        ? "新 Provider 将在下一次扫描时启用"
        : "当前 Provider 正在使用";
    }

    function update(status) {
      latest = status;
      $("stateBadge").textContent = status.state || "UNKNOWN";
      $("stateBadge").dataset.state = status.state || "UNKNOWN";
      $("gateBadge").textContent = status.gate_open ? "门禁开启" : "门禁关闭";
      setText("linearSpeed", status.linear_x?.toFixed(2));
      setText("angularSpeed", status.angular_z?.toFixed(2));
      setText("clearance", status.front_clearance_m?.toFixed(2));
      setText("perceptionFps", status.fps?.toFixed(0));
      setText("inference", status.inference_ms?.toFixed(0));
      setText("scanResult", status.scan_traversability || "pending");
      setText("scanTime", status.scan_seconds?.toFixed(1));
      setText(
        "tokens",
        status.input_tokens == null ? "--" : `${status.input_tokens}/${status.output_tokens}`
      );
      setText("device", status.device);
      setText("scanSummary", status.scan_summary || "尚无扫描结果");
      setText(
        "depthRange",
        status.depth_min_m == null ? "--" :
          `${status.depth_min_m.toFixed(1)}-${status.depth_max_m.toFixed(1)} m`
      );
      const alert = $("alert");
      alert.textContent = status.error || "";
      alert.classList.toggle("show", Boolean(status.error));
      const detections = status.detections || [];
      const list = $("detections");
      list.replaceChildren(...detections.map((d) => {
        const chip = document.createElement("span");
        chip.className = "detection";
        chip.textContent = `${d.label} ${Number(d.confidence).toFixed(2)}`;
        return chip;
      }));
      if (!detections.length) {
        const empty = document.createElement("span");
        empty.className = "card-note";
        empty.textContent = "当前帧未识别到目标";
        list.append(empty);
      }
      updateProviderButtons(status);
    }

    function updateStreams(streams, state) {
      const camera = streams.camera || {};
      const live = camera.frame_age_ms != null && camera.frame_age_ms < 1500;
      $("liveDot").classList.toggle("live", live && state !== "FAULT");
      $("liveText").textContent = state === "FAULT" ? "FAULT" : live ? "LIVE" : "PAUSED";
      const fps = camera.fps == null ? "--" : Math.round(camera.fps);
      $("cameraNote").textContent = `视频 ${fps} FPS`;
      $("frameAge").textContent =
        camera.frame_age_ms == null ? "--" : `${Math.round(camera.frame_age_ms)} ms`;
    }

    async function refresh() {
      try {
        const response = await fetch("/api/status", {cache:"no-store"});
        const data = await response.json();
        updateStreams(data.streams || {}, data.state);
        update(data);
      } catch (_) {
        $("liveText").textContent = "连接断开";
        $("liveDot").classList.remove("live");
      }
    }

    function toast(message) {
      const element = $("toast");
      element.textContent = message;
      element.classList.add("show");
      clearTimeout(element._timer);
      element._timer = setTimeout(() => element.classList.remove("show"), 3200);
    }

    async function switchProvider(provider) {
      try {
        const response = await fetch("/api/provider", {
          method:"POST",
          headers:{"Content-Type":"application/json", "X-Vision-Control":"1"},
          body:JSON.stringify({provider})
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
          throw new Error(result.message || "切换请求被拒绝");
        }
        toast(result.message);
        await refresh();
      } catch (error) {
        toast(`切换失败：${error.message}`);
      }
    }

    $("providerLocal").onclick = () => switchProvider("local");
    $("providerApi").onclick = () => $("apiDialog").showModal();
    $("apiCancel").onclick = () => $("apiDialog").close();
    $("apiConfirm").onclick = () => {
      if (apiConfirming) return;
      apiConfirming = true;
      $("apiDialog").close();
      switchProvider("api").finally(() => { apiConfirming = false; });
    };
    $("fullscreen").onclick = () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    };
    for (const id of streamIds) {
      $(id).addEventListener("error", () => {
        if (!streamActive[id]) return;
        setTimeout(() => {
          streamActive[id] = false;
          syncStreams();
        }, 1200);
      });
    }
    refresh();
    setInterval(refresh, 750);
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

    def client_count(self) -> int:
        with self._condition:
            return self._clients

    def status(self) -> dict[str, float | int | bool | None]:
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
        path = urlparse(self.path).path
        streams: dict[str, DebugStreamState] = self.server.streams  # type: ignore
        stream_paths = {
            "/stream/camera.mjpg": "camera",
            "/stream/depth.mjpg": "depth",
            "/stream/bev.mjpg": "bev",
            "/stream.mjpg": "legacy",
        }
        if path in {"/", "/index.html"}:
            self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            node: DebugWebNode = self.server.node  # type: ignore
            self._send_json(node.status_payload())
        elif path == "/health":
            self._send_bytes(b"ok\n", "text/plain; charset=utf-8")
        elif path in stream_paths:
            name = stream_paths[path]
            self._stream(name, streams[name])
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/provider":
            self.send_error(404)
            return
        node: DebugWebNode = self.server.node  # type: ignore
        if not self.server.control_enabled:  # type: ignore
            self._send_json(
                {"success": False, "message": "provider control is disabled"},
                status=403,
            )
            return
        if not self._local_client():
            self._send_json(
                {"success": False, "message": "provider control is local only"},
                status=403,
            )
            return
        if self.headers.get("X-Vision-Control") != "1":
            self._send_json(
                {"success": False, "message": "missing control header"}, status=403
            )
            return
        if not self._same_origin():
            self._send_json(
                {"success": False, "message": "cross-origin request rejected"},
                status=403,
            )
            return
        if "application/json" not in self.headers.get("Content-Type", "").lower():
            self._send_json(
                {"success": False, "message": "JSON content type required"}, status=415
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 1 or length > 128:
            self._send_json(
                {"success": False, "message": "invalid request size"}, status=413
            )
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                {"success": False, "message": "invalid JSON"}, status=400
            )
            return
        if set(body) != {"provider"} or body["provider"] not in {"local", "api"}:
            self._send_json(
                {"success": False, "message": "invalid provider"}, status=400
            )
            return
        success, message = node.switch_provider(body["provider"])
        self._send_json(
            {"success": success, "message": message},
            status=200 if success else 409,
        )

    def _stream(self, name: str, hub: DebugStreamState) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_sequence = 0
        hub.client_opened()
        node: DebugWebNode = self.server.node  # type: ignore
        node.stream_client_changed(name)
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
            node.stream_client_changed(name)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _local_client(self) -> bool:
        try:
            return ip_address(self.client_address[0]).is_loopback
        except ValueError:
            return False

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed = urlparse(origin)
        if parsed.port != self.server.server_port:  # type: ignore
            return False
        try:
            return ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            return parsed.hostname == "localhost"

    def log_message(self, format: str, *args) -> None:
        del format, args


class DebugWebNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_debug_web")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8080)
        self.declare_parameter("jpeg_quality", 70)
        self.declare_parameter("max_stream_fps", 10)
        self.declare_parameter("camera_max_fps", 15)
        self.declare_parameter("depth_max_fps", 6)
        self.declare_parameter("bev_max_fps", 6)
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        self._quality = int(self.get_parameter("jpeg_quality").value)
        self._periods = {
            "camera": 1.0
            / max(1, int(self.get_parameter("camera_max_fps").value)),
            "depth": 1.0
            / max(1, int(self.get_parameter("depth_max_fps").value)),
            "bev": 1.0 / max(1, int(self.get_parameter("bev_max_fps").value)),
            "legacy": 1.0
            / max(1, int(self.get_parameter("max_stream_fps").value)),
        }
        self._control_enabled = self._is_loopback_host(host)
        self._bridge = CvBridge()
        self._streams = {
            name: DebugStreamState()
            for name in ("camera", "depth", "bev", "legacy")
        }
        self._last_encoded: dict[str, float] = {}
        self._status_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._stream_clients = {
            name: False for name in ("camera", "depth", "bev", "legacy")
        }
        self._provider_client = self.create_client(
            SetBool, "/vision/set_api_provider"
        )
        self._debug_clients_pub = self.create_publisher(
            String, "/vision/debug_clients", 10
        )
        self._debug_status: dict = {}
        self._server: ThreadingHTTPServer | None = None
        try:
            self._server = ThreadingHTTPServer((host, port), DebugRequestHandler)
            self._server.daemon_threads = True
            self._server.streams = self._streams  # type: ignore
            self._server.node = self  # type: ignore
            self._server.control_enabled = self._control_enabled  # type: ignore
        except OSError as exc:
            self.get_logger().error(f"cannot start debug web server: {exc}")
        if self._server is not None:
            self._server_thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._server_thread.start()
            self.get_logger().info(f"debug dashboard: http://{host}:{port}")

        jpeg_topics = {
            "/vision/camera_jpeg": "camera",
            "/vision/depth_jpeg": "depth",
            "/vision/bev_jpeg": "bev",
        }
        for topic, name in jpeg_topics.items():
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, stream_name=name: self._compressed_callback(
                    stream_name, message
                ),
                qos_profile_sensor_data,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
        self.create_subscription(
            Image,
            "/vision/debug_image",
            lambda message: self._image_callback("legacy", message),
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_subscription(
            String,
            "/vision/debug_status",
            self._status_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if host == "localhost":
            return True
        try:
            return ip_address(host).is_loopback
        except ValueError:
            return False

    def _image_callback(self, stream_name: str, message: Image) -> None:
        hub = self._streams[stream_name]
        if hub.client_count() == 0:
            return
        now = time.monotonic()
        if (
            now - self._last_encoded.get(stream_name, 0.0)
            < self._periods[stream_name] * 0.75
        ):
            return
        self._last_encoded[stream_name] = now
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
            )
            if ok:
                hub.publish(encoded.tobytes())
        except Exception as exc:
            self.get_logger().error(
                f"debug frame conversion failed: {type(exc).__name__}"
            )

    def _compressed_callback(
        self, stream_name: str, message: CompressedImage
    ) -> None:
        hub = self._streams[stream_name]
        if hub.client_count() == 0:
            return
        now = time.monotonic()
        if (
            now - self._last_encoded.get(stream_name, 0.0)
            < self._periods[stream_name] * 0.75
        ):
            return
        self._last_encoded[stream_name] = now
        hub.publish(bytes(message.data))

    def stream_client_changed(self, stream_name: str) -> None:
        hub = self._streams[stream_name]
        with self._client_lock:
            self._stream_clients[stream_name] = hub.client_count() > 0
            self._debug_clients_pub.publish(
                String(
                    data=json.dumps(
                        self._stream_clients, separators=(",", ":")
                    )
                )
            )

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
            if not isinstance(status, dict):
                return
            with self._status_lock:
                self._debug_status = status
        except json.JSONDecodeError:
            self.get_logger().warning("invalid debug status JSON")

    def status_payload(self) -> dict:
        with self._status_lock:
            status = dict(self._debug_status)
        streams = {name: hub.status() for name, hub in self._streams.items()}
        status.update(
            {
                "streams": streams,
                "control_enabled": self._control_enabled,
            }
        )
        return status

    def switch_provider(self, provider: str) -> tuple[bool, str]:
        if not self._provider_client.wait_for_service(timeout_sec=2.0):
            return False, "provider service is unavailable"
        request = SetBool.Request()
        request.data = provider == "api"
        event = threading.Event()
        future = self._provider_client.call_async(request)
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=3.0):
            future.cancel()
            return False, "provider switch timed out"
        try:
            response = future.result()
            return bool(response.success), str(response.message)
        except Exception as exc:
            return False, f"provider switch failed: {type(exc).__name__}"

    def destroy_node(self) -> bool:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = DebugWebNode()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
