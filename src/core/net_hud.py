import cv2
import numpy as np
from typing import Dict, Tuple


LETTER_TO_BGR: Dict[str, Tuple[int, int, int]] = {
    'U': (255, 255, 255),   # white
    'D': (0, 255, 255),     # yellow
    'R': (0, 0, 255),       # red
    'L': (0, 128, 255),     # orange
    'F': (0, 255, 0),       # green
    'B': (255, 0, 0),       # blue
    '?': (40, 40, 40),
}


class NetHUD:
    """Render a 2D unfolded cube net in the bottom-right corner.

    Layout (faces):
        [  -,  U,  -,  - ]
        [  L,  F,  R,  B ]
        [  -,  D,  -,  - ]
    Each face is 3x3.
    """

    def __init__(self, margin: int = 10, tile_gap: int = 2, label: bool = False):
        self.margin = int(margin)
        self.tile_gap = int(tile_gap)
        self.show_label = bool(label)

    def _ensure_tile_size(self, W: int, H: int, scale: float) -> int:
        # Net is 12 tiles wide (4 faces * 3 tiles), 9 tiles tall (3 faces * 3 tiles)
        max_w = int(W * float(scale))
        max_h = int(H * float(scale))
        tw_by_w = max(6, (max_w - 3 * self.tile_gap) // 12)  # account for gaps between faces
        tw_by_h = max(6, (max_h - 2 * self.tile_gap) // 9)
        return int(max(6, min(tw_by_w, tw_by_h)))

    def render(self, img: np.ndarray, state_manager, scale: float = 0.28) -> np.ndarray:
        if img is None or img.size == 0:
            return img
        H, W = img.shape[:2]
        tile = self._ensure_tile_size(W, H, scale)
        gap = self.tile_gap

        net_w = 12 * tile + 3 * gap  # three gaps between four faces in a row
        net_h = 9 * tile + 2 * gap   # two gaps between three faces in a column

        # Bottom-right anchor
        x0 = max(self.margin, W - net_w - self.margin)
        y0 = max(self.margin, H - net_h - self.margin)

        # Face layout positions in face-grid units (face index cells), each face 3x3
        layout = {
            'U': (3, 0),  # start column, start row in tile units of faces (faces columns: 0..3, rows: 0..2)
            'L': (0, 1),
            'F': (3, 1),
            'R': (6, 1),
            'B': (9, 1),
            'D': (3, 2),
        }

        def draw_face(face_letter: str):
            if face_letter not in layout:
                return
            fc, fr = layout[face_letter]
            colors = state_manager.get_face_colors(face_letter)
            if colors is None:
                # Unknown face: draw grid with '?' color
                colors = [["?"] * 3 for _ in range(3)]
            for i in range(3):
                for j in range(3):
                    letter = colors[i][j]
                    bgr = LETTER_TO_BGR.get(letter if letter in LETTER_TO_BGR else face_letter, (60, 60, 60))
                    # Compute pixel rect
                    cx = x0 + (fc + j) * tile + (fc // 3) * gap
                    cy = y0 + (fr + i) * tile + (fr // 3) * gap
                    x1, y1 = cx, cy
                    x2, y2 = cx + tile - 1, cy + tile - 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, -1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (10, 10, 10), 1)
                    if self.show_label and letter in LETTER_TO_BGR:
                        cv2.putText(img, letter, (x1 + 3, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2,
                                    cv2.LINE_AA)
                        cv2.putText(img, letter, (x1 + 3, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                    cv2.LINE_AA)

        for face in ['U', 'L', 'F', 'R', 'B', 'D']:
            draw_face(face)
        return img

