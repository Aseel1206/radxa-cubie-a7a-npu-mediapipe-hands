import math
import cv2
import numpy as np


def roi_from_det(box, kp, r, l, t, scale=2.6, shift_y=-0.5):
    """Rotated square hand ROI (cx, cy, size, rotation) in ORIGINAL image pixels, the way MediaPipe does it.
    box, kp: palm detector output (normalised to the 192x192 letterboxed input); r, l, t: letterbox scale/left/top.
    The rotation makes the wrist -> middle-finger-base direction vertical; the box is shifted along that axis and enlarged by `scale`."""
    px = (kp[:, 0] * 192 - l) / r; py = (kp[:, 1] * 192 - t) / r
    rot = math.pi / 2 - math.atan2(-(py[2] - py[0]), px[2] - px[0]); rot = rot - 2 * math.pi * math.floor((rot + math.pi) / (2 * math.pi))
    cx = ((box[0] + box[2]) / 2 * 192 - l) / r; cy = ((box[1] + box[3]) / 2 * 192 - t) / r
    w = (box[2] - box[0]) * 192 / r; h = (box[3] - box[1]) * 192 / r
    cx += -h * shift_y * math.sin(rot); cy += h * shift_y * math.cos(rot)
    return cx, cy, max(w, h) * scale, rot


def crop(img, cx, cy, size, rot, out=224):
    """Warp the rotated ROI to out x out. Returns (patch, M); M maps patch pixels -> image pixels."""
    c, s = math.cos(rot), math.sin(rot); k = size / out
    M = np.array([[c * k, -s * k, cx - (c * k * out / 2 - s * k * out / 2)], [s * k, c * k, cy - (s * k * out / 2 + c * k * out / 2)]], np.float32)
    return cv2.warpAffine(img, M, (out, out), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), M
