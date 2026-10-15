"""
RecoG - drawing helpers: neon hand skeleton + minimal HUD.
"""
import cv2
import numpy as np

# Landmark chains per finger (wrist=0, fingertips=4,8,12,16,20)
FINGERS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}
PALM = [0, 5, 9, 13, 17, 0]
FINGERTIPS = [4, 8, 12, 16, 20]

# BGR colors - one neon color per finger
COLORS = {
    "thumb":  (180, 105, 255),   # pink
    "index":  (240, 220, 60),    # cyan
    "middle": (255, 120, 160),   # violet
    "ring":   (255, 160, 60),    # blue
    "pinky":  (170, 255, 120),   # mint
}
PALM_COLOR = (210, 210, 210)
WHITE = (255, 255, 255)
AA = cv2.LINE_AA  # anti-aliased = smooth edges
FONT = cv2.FONT_HERSHEY_DUPLEX


def to_pixels(landmarks, w: int, h: int) -> list[tuple[int, int]]:
    """MediaPipe gives 0..1 coordinates -> convert to pixels."""
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def _polyline(img, points, chain, color, thickness):
    pts = np.array([points[i] for i in chain], dtype=np.int32)
    cv2.polylines(img, [pts], False, color, thickness, AA)


def _add_glow(frame, layer):
    """Blur a layer at half resolution and add it on top = soft neon glow."""
    h, w = frame.shape[:2]
    small = cv2.resize(layer, (w // 2, h // 2))
    small = cv2.GaussianBlur(small, (0, 0), 5)
    glow = cv2.resize(small, (w, h))
    cv2.add(frame, glow, dst=frame)


def pill(frame, text: str, org: tuple[int, int], scale: float = 0.55,
         alpha: float = 0.55, pad: int = 8, center: bool = False) -> None:
    """Semi-transparent dark label with white text.
    org = top-left corner, or top-center if center=True."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    if center:
        x -= tw // 2 + pad
    x1, y1 = max(x, 0), max(y, 0)
    x2 = min(x + tw + pad * 2, frame.shape[1])
    y2 = min(y + th + base + pad * 2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    roi[:] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)
    cv2.putText(frame, text, (x + pad, y + pad + th), FONT, scale, WHITE, 1, AA)


def draw_hands(frame, hands) -> None:
    """hands = list of (landmarks, label). Draws glow + crisp skeleton."""
    h, w = frame.shape[:2]
    all_points = [(to_pixels(lms, w, h), label) for lms, label in hands]
    if not all_points:
        return

    # 1) glow layer: thick lines on black, blurred, added to the frame
    glow = np.zeros_like(frame)
    for points, _ in all_points:
        for name, chain in FINGERS.items():
            _polyline(glow, points, chain, COLORS[name], 8)
    _add_glow(frame, glow)

    # 2) crisp thin lines + joints on top
    for points, label in all_points:
        _polyline(frame, points, PALM, PALM_COLOR, 1)
        for name, chain in FINGERS.items():
            _polyline(frame, points, chain, COLORS[name], 2)
            for i in chain:
                if i in FINGERTIPS:
                    cv2.circle(frame, points[i], 5, COLORS[name], -1, AA)
                    cv2.circle(frame, points[i], 9, WHITE, 1, AA)
                else:
                    cv2.circle(frame, points[i], 3, WHITE, -1, AA)

        wx, wy = points[0]
        pill(frame, label, (wx, wy + 14), scale=0.5, center=True)


def draw_hud(frame, fps: float, hand_count: int) -> None:
    pill(frame, f"{fps:4.0f} FPS   |   {hand_count} hand{'s' if hand_count != 1 else ''}",
         (16, 16))