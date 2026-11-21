import threading
import json
import urllib.error
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
    streams = {
        name: DebugStreamState() for name in ("camera", "depth", "bev", "legacy")
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), DebugRequestHandler)
    server.streams = streams  # type: ignore[attr-defined]
    server.node = type(
        "FakeNode",
        (),
        {
            "status_payload": lambda self: {
                "api_available": False,
                "active_provider": "local",
                "control_enabled": True,
                "streams": {
                    name: stream.status() for name, stream in streams.items()
                },
            },
            "switch_provider": lambda self, provider: (
                True,
                f"{provider} pending",
            ),
        },
    )()  # type: ignore[attr-defined]
    server.control_enabled = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(f"{base}/", timeout=2) as response:
            page = response.read().decode()
        assert "小车视觉避障" in page
        assert "/stream/camera.mjpg" in page
        assert "/stream/depth.mjpg" in page
        assert "/stream/bev.mjpg" in page
        assert 'data-src="/stream/camera.mjpg"' in page
        assert "new IntersectionObserver" in page
        assert "document.hidden" in page
        assert "小车视觉避障" in HTML_PAGE
        with urllib.request.urlopen(f"{base}/api/status", timeout=2) as response:
            assert response.headers["Content-Type"].startswith("application/json")
            assert b"frames_received" in response.read()

        request = urllib.request.Request(
            f"{base}/api/provider",
            data=json.dumps({"provider": "api"}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Vision-Control": "1",
                "Origin": base,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.loads(response.read())
        assert result["success"]

        missing_header = urllib.request.Request(
            f"{base}/api/provider",
            data=b'{"provider":"api"}',
            headers={"Content-Type": "application/json", "Origin": base},
            method="POST",
        )
        try:
            urllib.request.urlopen(missing_header, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403
        else:
            raise AssertionError("control request without header was accepted")

        cases = [
            (b'{"provider":"remote"}', base, 400),
            (
                b'{"provider":"api"}',
                "http://192.0.2.10:8080",
                403,
            ),
            (b"x" * 129, base, 413),
        ]
        for body, origin, expected_status in cases:
            request = urllib.request.Request(
                f"{base}/api/provider",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Vision-Control": "1",
                    "Origin": origin,
                },
                method="POST",
            )
            try:
                urllib.request.urlopen(request, timeout=2)
            except urllib.error.HTTPError as exc:
                assert exc.code == expected_status
            else:
                raise AssertionError(
                    f"invalid control request was accepted: {body!r}"
                )
    finally:
        server.shutdown()
        server.server_close()
