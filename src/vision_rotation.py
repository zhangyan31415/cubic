import numpy as np
import cv2


def rotation_k_from_rvec(rvec: np.ndarray) -> int:
    """
    Convert PnP rvec to a roll quantized in k*90° around the face normal.
    Returns k in {0,1,2,3}; rotate grid clockwise by k.
    """
    if rvec is None or np.asarray(rvec).size != 3:
        return 0
    R, _ = cv2.Rodrigues(rvec)
    face_x = R @ np.array([1.0, 0.0, 0.0], dtype=float)
    angle = np.arctan2(face_x[1], face_x[0])
    k = int(np.round(angle / (np.pi / 2.0))) % 4
    return k


def rotate_grid_labels(labels9, k: int):
    """
    Rotate a 3x3 face label list (len=9, row-major) clockwise k times.
    """
    g = np.array(labels9, dtype=object).reshape(3, 3)
    g_rot = np.rot90(g, -k)
    return g_rot.reshape(-1).tolist()

