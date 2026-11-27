from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def detect_views(
    images: list[np.ndarray], pattern: tuple[int, int]
) -> tuple[list[np.ndarray], tuple[int, int]]:
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        image_points.append(corners.reshape(-1, 2))
    if image_size is None:
        raise ValueError("no images supplied")
    return image_points, image_size


def load_folder(folder: Path) -> list[np.ndarray]:
    files = sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    images = [cv2.imread(str(path)) for path in files]
    return [image for image in images if image is not None]


def capture_images(
    device: int, width: int = 640, height: int = 480
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera /dev/video{device}")
    images: list[np.ndarray] = []
    print("SPACE captures a checkerboard view, C calibrates, Q quits")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                print(
                    f"warning: camera returned {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}"
                )
            cv2.imshow("camera calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                images.append(frame.copy())
                print(f"captured {len(images)}")
            if key == ord("c"):
                return images
            if key == ord("q"):
                raise KeyboardInterrupt
    finally:
        capture.release()
        cv2.destroyAllWindows()


def calibrate(
    images: list[np.ndarray],
    pattern: tuple[int, int],
    square_size_m: float,
) -> dict:
    image_points, image_size = detect_views(images, pattern)
    if len(image_points) < 5:
        raise ValueError(f"need at least 5 usable views, found {len(image_points)}")

    grid = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    grid[:, :2] = (
        np.mgrid[0 : pattern[0], 0 : pattern[1]]
        .T.reshape(-1, 2)
        .astype(np.float32)
        * square_size_m
    )
    object_points = [grid.copy() for _ in image_points]
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
        "distortion": distortion.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "views": len(image_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, help="folder of checkerboard images")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cols", type=int, default=9, help="internal corner columns")
    parser.add_argument("--rows", type=int, default=6, help="internal corner rows")
    parser.add_argument("--square-size", type=float, default=0.025, help="metres")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("camera_calibration.yaml"),
    )
    args = parser.parse_args()

    images = (
        load_folder(args.images.expanduser().resolve())
        if args.images
        else capture_images(args.camera, args.width, args.height)
    )
    result = calibrate(images, (args.cols, args.rows), args.square_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(f"wrote {args.output} with RMS {result['rms_reprojection_error']:.3f}px")


if __name__ == "__main__":
    main()
