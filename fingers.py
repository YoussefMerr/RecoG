"""
RecoG - finger state detection.
Decides which fingers are up using MediaPipe's 3D *world* landmarks
(in meters, centered on the hand), so it works at any distance or angle.
"""
import math

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# (tip, pip) joint indices for the four long fingers
LONG_FINGERS = {
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
WRIST = 0
THUMB_TIP, THUMB_MCP, PINKY_MCP = 4, 2, 17

# Tuning knobs: raise if fingers count as "up" too easily, lower if too hard
FINGER_MARGIN = 1.05
THUMB_MARGIN = 1.10


def _dist(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def fingers_up(world) -> dict[str, bool]:
    """world = 21 world landmarks of one hand -> {"thumb": True, ...}"""
    wrist = world[WRIST]
    state = {}

    # Long finger is up if its tip is clearly farther from the wrist
    # than its middle joint (a curled finger folds the tip back in).
    for name, (tip, pip) in LONG_FINGERS.items():
        state[name] = _dist(world[tip], wrist) > _dist(world[pip], wrist) * FINGER_MARGIN

    # Thumb is up if its tip is farther from the pinky base than its own base
    # (a folded thumb crosses the palm toward the pinky).
    pinky_base = world[PINKY_MCP]
    state["thumb"] = (_dist(world[THUMB_TIP], pinky_base)
                      > _dist(world[THUMB_MCP], pinky_base) * THUMB_MARGIN)

    return {name: state[name] for name in FINGER_NAMES}  # keep a fixed order


def count_up(state: dict[str, bool]) -> int:
    return sum(state.values())  # True counts as 1, False as 0