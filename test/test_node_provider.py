import threading
from types import SimpleNamespace

from vision_avoid.node import AvoidanceNode
from std_msgs.msg import String
from std_srvs.srv import SetBool


class FakeLocalScanner:
    def __init__(self):
        self.warmups = 0

    def warmup(self):
        self.warmups += 1


def test_local_scan_warms_qwen_after_an_api_pending_unload():
    scanner = FakeLocalScanner()
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _scanner=scanner,
        _active_provider="local",
        _local_qwen_unload_thread=None,
        _local_qwen_unloaded=True,
        _check_gpu_headroom=lambda: None,
    )

    assert AvoidanceNode._scanner_for_scan(node, "local") is scanner
    assert scanner.warmups == 1
    assert not node._local_qwen_unloaded


def test_provider_switch_is_pending_and_rejects_unconfigured_api():
    unloaded = []
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _api_available=False,
        _active_provider="local",
        _scanner=object(),
        _pending_provider=None,
        _scanning=False,
        _unload_qwen_after_scan=False,
        _provider_change_id=0,
        _unload_local_qwen_async=lambda: unloaded.append(True),
    )

    response = AvoidanceNode._set_api_provider_callback(
        node, SetBool.Request(data=True), SetBool.Response()
    )
    assert not response.success
    assert node._pending_provider is None

    node._api_available = True
    response = AvoidanceNode._set_api_provider_callback(
        node, SetBool.Request(data=True), SetBool.Response()
    )
    assert response.success
    assert node._active_provider == "local"
    assert node._pending_provider == "api"
    assert unloaded


def test_api_switch_during_scan_waits_for_scan_before_unload():
    unloaded = []
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _api_available=True,
        _active_provider="local",
        _scanner=object(),
        _pending_provider=None,
        _scanning=True,
        _unload_qwen_after_scan=False,
        _provider_change_id=0,
        _unload_local_qwen_async=lambda: unloaded.append(True),
    )

    response = AvoidanceNode._set_api_provider_callback(
        node, SetBool.Request(data=True), SetBool.Response()
    )
    assert response.success
    assert node._active_provider == "local"
    assert node._pending_provider == "api"
    assert node._unload_qwen_after_scan
    assert not unloaded


def test_debug_client_state_only_accepts_known_stream_names():
    node = SimpleNamespace(
        _lock=threading.RLock(),
        _debug_stream_active={
            "camera": False,
            "depth": False,
            "bev": False,
        },
    )
    AvoidanceNode._debug_clients_callback(
        node,
        String(data='{"camera":true,"depth":false,"bev":true,"other":true}'),
    )
    assert node._debug_stream_active == {
        "camera": True,
        "depth": False,
        "bev": True,
    }
