"""
RecoG - Step 1: camera check.
Opens the Logitech C920, shows a mirrored live feed with FPS.
Press Q to quit.
"""
import time
import cv2

CAMERA_INDEX = 0          # try 1 if the C920 isn't your first camera
WIDTH, HEIGHT, FPS = 1280, 720, 30


def open_camera(index: int) -> cv2.VideoCapture:
    # DirectShow is the most reliable backend for the C920 on Windows
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    # MJPG lets the C920 deliver 720p at a full 30 fps over USB
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


def main() -> None:
    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {CAMERA_INDEX}. "
            "Check Windows camera privacy settings, close Teams/Zoom/Logi Tune, "
            "or try CAMERA_INDEX = 1."
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened at {w}x{h}. Press Q to quit.")

    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        frame = cv2.flip(frame, 1)  # mirror so it feels natural

        now = time.perf_counter()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RecoG - camera test", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
