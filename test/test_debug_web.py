import threading
import urllib.request
from http.server import ThreadingHTTPServer

from vision_avoid.debug_web import (
    HTML_PAGE,
    DebugRequestHandler,
    DebugStreamState,
)


def test_stream_state_publishes_frame_and_status():
    stream = DebugStreamState()
    stream.publish(b"jpeg")
    sequence, frame = stream.wait_for_frame(0, timeout=0.1)
    assert sequence == 1
    assert frame == b"jpeg"
    status = stream.status()
    assert status["live"]
    assert status["frames_received"] == 1


def test_html_dashboard_and_status_endpoint():
    stream = DebugStreamState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), DebugRequestHandler)
    server.hub = stream  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/", timeout=2) as response:
            page = response.read().decode()
        assert "Vision Avoid Live" in page
        assert "/stream.mjpg" in page
        assert "Vision Avoid Live" in HTML_PAGE
        with urllib.request.urlopen(f"{base}/api/status", timeout=2) as response:
            assert response.headers["Content-Type"] == "application/json"
            assert b"frames_received" in response.read()
    finally:
        server.shutdown()
        server.server_close()
