"""
RecoG - Step 2: hand tracking.
Detects up to 2 hands with MediaPipe and draws a neon skeleton.
Press Q to quit.
"""
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from camera_test import open_camera, CAMERA_INDEX
from drawing import draw_hands, draw_hud
from fingers import fingers_up

MODEL_PATH = os.path.join("models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def ensure_model() -> None:
    """Download the hand model once; it's gitignored."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading hand model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def real_handedness(mp_label: str) -> str:
    """We mirror the frame before detection, so MediaPipe's label is swapped."""
    return "Right" if mp_label == "Left" else "Left"


def main() -> None:
    ensure_model()
    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {CAMERA_INDEX}.")

    start = time.perf_counter()
    prev = start
    last_ts = -1

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs strictly increasing timestamps (ms)
            ts = max(int((time.perf_counter() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            hands = [
                (lms, real_handedness(hd[0].category_name), fingers_up(world))
                for lms, world, hd in zip(result.hand_landmarks,
                                          result.hand_world_landmarks,
                                          result.handedness)
            ]
            draw_hands(frame, hands)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            draw_hud(frame, fps, len(hands))

            cv2.imshow("RecoG", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()