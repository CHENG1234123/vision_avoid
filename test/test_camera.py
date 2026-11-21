from vision_avoid.camera_node import open_video_capture


class FakeCapture:
    def __init__(self, path, backend):
        self.path = path
        self.backend = backend
        self.settings = {}

    def set(self, key, value):
        self.settings[key] = value
        return True


def test_camera_open_configures_mjpg_and_size(monkeypatch):
    import vision_avoid.camera_node as camera_node

    monkeypatch.setattr(camera_node.cv2, "VideoCapture", FakeCapture)
    capture = open_video_capture("/dev/v4l/by-id/test", 640, 480, 30)
    assert capture.path == "/dev/v4l/by-id/test"
    assert capture.settings[camera_node.cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert capture.settings[camera_node.cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert capture.settings[camera_node.cv2.CAP_PROP_FPS] == 30
