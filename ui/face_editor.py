import cv2
import numpy as np
from typing import List, Tuple

PALETTE = {
    "White": (255, 255, 255),
    "Yellow": (0, 255, 255),
    "Red": (0, 0, 255),
    "Orange": (0, 128, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "?": (32, 32, 32),
}
ORDER = ["White", "Yellow", "Red", "Orange", "Green", "Blue", "?"]


class FaceEditor:
    """
    Lightweight face correction panel using OpenCV window.
    - Left click: cycle the clicked cell color.
    - Enter/Space: confirm current face.
    - U/Backspace: undo to previous snapshot.
    - P: pause/resume flag (outer loop can observe).
    - Q/ESC: cancel.
    """

    def __init__(self, win_name: str = "Face Editor", cell=64, margin=12):
        self.win = win_name
        self.cell = int(cell)
        self.margin = int(margin)
        self.grid: List[str] = ["?"] * 9
        self.paused: bool = False
        self.history: List[List[str]] = []
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def set_grid(self, labels9: List[str]):
        assert len(labels9) == 9
        self.grid = labels9.copy()

    def push_history(self, snapshot: List[str]):
        self.history.append(snapshot.copy())

    def pop_history(self) -> List[str] | None:
        if self.history:
            self.grid = self.history.pop()
            return self.grid
        return None

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        i, j = self._xy_to_ij(x, y)
        if 0 <= i < 3 and 0 <= j < 3:
            idx = i * 3 + j
            curr = self.grid[idx]
            try:
                k = (ORDER.index(curr) + 1) % len(ORDER)
            except ValueError:
                k = 0
            self.grid[idx] = ORDER[k]

    def _xy_to_ij(self, x, y) -> Tuple[int, int]:
        M = self.margin
        C = self.cell
        gx, gy = (x - M) // C, (y - M) // C
        return int(gy), int(gx)

    def _render(self) -> np.ndarray:
        M, C = self.margin, self.cell
        H, W = M * 2 + C * 3, M * 2 + C * 3
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (24, 24, 24)
        for i in range(3):
            for j in range(3):
                x1, y1 = M + j * C, M + i * C
                x2, y2 = x1 + C - 2, y1 + C - 2
                color = PALETTE.get(self.grid[i * 3 + j], (64, 64, 64))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        tip = "[Click]改色  [Enter/Space]确认  [U/Backspace]回退  [P]暂停  [Q/ESC]退出"
        cv2.rectangle(img, (0, H - 26), (W, H), (0, 0, 0), -1)
        cv2.putText(img, tip, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if self.paused:
            cv2.putText(img, "PAUSED", (W - 90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        return img

    def edit_until_confirm(self, init_labels9: List[str]) -> Tuple[bool, List[str]]:
        self.set_grid(init_labels9)
        self.push_history(init_labels9)
        while True:
            cv2.imshow(self.win, self._render())
            key = cv2.waitKey(15) & 0xFF
            if key in (13, 32):  # Enter / Space
                return True, self.grid.copy()
            elif key in (ord('u'), 8):  # U / Backspace
                _ = self.pop_history()
            elif key in (ord('p'),):
                self.paused = not self.paused
            elif key in (ord('q'), 27):
                return False, self.grid.copy()

