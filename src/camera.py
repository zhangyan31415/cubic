import cv2


class Camera:
    """Simple camera wrapper for consistent lifecycle management."""

    def __init__(self, cam_id: int = 0):
        self.cam_id = cam_id
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        return self

    def set_resolution(self, width: int, height: int):
        """Attempt to set camera capture resolution."""
        if self.cap is None:
            return
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        except Exception:
            pass

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("读取摄像头帧失败")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
