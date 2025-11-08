"""
检测模块：魔方检测
采用多线索融合（边缘/颜色/色彩度）进行鲁棒检测，可选YOLO增强
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover - 运行环境缺少scikit-learn时告警
    DBSCAN = None


class RubikDetector:
    """魔方检测器，优先使用传统CV方法，在无训练模型时保持高成功率"""

    def __init__(
        self,
        use_yolo: bool = False,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        debug: bool = False,
    ):
        """
        Args:
            use_yolo: 是否启用YOLO（需要训练好的魔方模型）
            model_path: YOLO模型路径
            conf_threshold: YOLO置信度阈值
            debug: 是否记录调试信息
        """
        self.use_yolo = use_yolo
        self.conf_threshold = conf_threshold
        self.debug = debug

        self.yolo_model = None
        if use_yolo:
            try:
                from ultralytics import YOLO  # 延迟导入

                self.yolo_model = YOLO(model_path)
                print(f"✓ YOLO模型已加载: {model_path}")
            except Exception as exc:
                print(f"⚠ YOLO加载失败，将使用CV方式检测: {exc}")
                self.use_yolo = False

        # 调参：候选色块的最小/最大面积占比（相对于短边平方）
        self.area_ratio_low = 0.0004
        self.area_ratio_high = 0.12

        # 调参：宽高比允许范围
        self.aspect_low = 0.55
        self.aspect_high = 1.6

        # 调参：紧凑度（面积 / 外接矩形面积）
        self.compactness_threshold = 0.65

        # 调试信息缓存
        self.debug_last_masks: List[np.ndarray] = []
        self.debug_last_candidates: List[dict] = []

    # ------------------------------------------------------------------ #
    # 内部工具函数
    # ------------------------------------------------------------------ #
    def _generate_masks(self, frame: np.ndarray) -> List[np.ndarray]:
        """生成多种候选掩膜，融合不同视觉线索"""
        masks = []
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1) 自适应阈值（适应不同光照）
        mask_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        masks.append(mask_adapt)

        # 2) Canny 边缘
        mask_canny = cv2.Canny(blurred, 40, 140)
        masks.append(mask_canny)

        # 3) HSV 饱和度阈值（高饱和度区域通常为魔方色块）
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, mask_sat = cv2.threshold(hsv[:, :, 1], 60, 255, cv2.THRESH_BINARY)
        masks.append(mask_sat)

        # 4) Lab 颜色差异（a/b通道差异较大的区域）
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ab = cv2.addWeighted(
            cv2.absdiff(lab[:, :, 1], 128),
            0.5,
            cv2.absdiff(lab[:, :, 2], 128),
            0.5,
            0,
        )
        ab = cv2.GaussianBlur(ab, (5, 5), 0)
        _, mask_lab = cv2.threshold(ab, np.mean(ab) + np.std(ab) * 0.3, 255, cv2.THRESH_BINARY)
        masks.append(mask_lab.astype(np.uint8))

        # 5) 色彩度（Colorfulness）指数
        b, g, r = cv2.split(frame.astype(np.float32))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = np.sqrt(rg**2 + yb**2)
        colorfulness = cv2.normalize(colorfulness, None, 0, 255, cv2.NORM_MINMAX)
        colorfulness = cv2.GaussianBlur(colorfulness, (7, 7), 0)
        _, mask_color = cv2.threshold(
            colorfulness.astype(np.uint8), np.mean(colorfulness) + np.std(colorfulness), 255, cv2.THRESH_BINARY
        )
        masks.append(mask_color)

        # 6) 形态学增强版本（防止断裂）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for m in list(masks):
            enhanced = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            masks.append(enhanced)

        return masks

    def _collect_candidates(self, frame: np.ndarray, mask: np.ndarray) -> List[dict]:
        """从单个掩膜中提取候选色块"""
        if mask is None:
            return []

        mask_u8 = mask if mask.dtype == np.uint8 else cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        h, w = frame.shape[:2]
        ref = float(min(h, w) ** 2)
        min_area = max(60.0, ref * self.area_ratio_low)
        max_area = ref * self.area_ratio_high

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) not in (4, 5):  # 允许轻微变形
                continue

            x, y, w_box, h_box = cv2.boundingRect(approx)
            if w_box < 8 or h_box < 8:
                continue

            aspect = float(w_box) / h_box
            if not (self.aspect_low <= aspect <= self.aspect_high):
                continue

            rect_area = float(w_box * h_box)
            compactness = area / rect_area if rect_area > 0 else 0.0
            if compactness < self.compactness_threshold:
                continue

            # 计算凸包充填度（solidity）
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 1e-6 else 0.0
            if solidity < 0.8:
                continue

            cx = x + w_box / 2.0
            cy = y + h_box / 2.0
            size = (w_box + h_box) / 2.0

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": w_box,
                    "h": h_box,
                    "area": area,
                    "aspect": aspect,
                    "compactness": compactness,
                    "solidity": solidity,
                    "cx": cx,
                    "cy": cy,
                    "size": size,
                }
            )

        return candidates

    def _cluster_candidates(self, frame: np.ndarray, candidates: List[dict]) -> List[dict]:
        """使用DBSCAN聚类候选色块，挑选最符合3x3结构的集合"""
        if not candidates:
            return []

        centers = np.array([[c["cx"], c["cy"]] for c in candidates], dtype=np.float32)
        sizes = np.array([c["size"] for c in candidates], dtype=np.float32)
        avg_size = float(np.median(sizes)) if len(sizes) else 20.0
        eps = max(avg_size * 0.8, 15.0)

        labels = None
        if DBSCAN is not None and len(candidates) >= 3:
            try:
                labels = DBSCAN(eps=eps, min_samples=4).fit(centers).labels_
            except Exception:
                labels = None

        if labels is None:
            labels = np.zeros(len(candidates), dtype=int)

        best_cluster: List[dict] = []
        best_score = -1e9

        unique_labels = [l for l in set(labels) if l != -1]
        if not unique_labels:
            unique_labels = [0]  # 全部作为一类

        for label in unique_labels:
            cluster = [candidates[i] for i in range(len(candidates)) if labels[i] == label]
            if len(cluster) < 5:
                continue

            xs = np.array([c["x"] for c in cluster])
            ys = np.array([c["y"] for c in cluster])
            ws = np.array([c["w"] for c in cluster])
            hs = np.array([c["h"] for c in cluster])

            span_w = float(np.max(xs + ws) - np.min(xs))
            span_h = float(np.max(ys + hs) - np.min(ys))
            if span_w < 20 or span_h < 20:
                continue

            aspect = span_w / span_h if span_h > 1e-6 else 1.0
            size_std = float(np.std([c["size"] for c in cluster])) if len(cluster) > 1 else 0.0
            compact_mean = float(np.mean([c["compactness"] for c in cluster]))

            # 分数：方块数量 + 紧凑度 + 结构性 - 尺寸差异 - 纵横比偏差
            score = (
                len(cluster) * 1.5
                + compact_mean * 2.0
                - 0.4 * abs(aspect - 1.0)
                - 0.15 * (size_std / (avg_size + 1e-6))
            )

            if score > best_score:
                best_score = score
                best_cluster = cluster

        if not best_cluster and candidates:
            # 如果聚类失败，退回到面积居中的前9个
            sorted_by_size = sorted(candidates, key=lambda c: c["area"])
            start = max(0, len(sorted_by_size) - 9) // 2
            best_cluster = sorted_by_size[start : start + 9]

        return best_cluster

    # ------------------------------------------------------------------ #
    # 主检测流程
    # ------------------------------------------------------------------ #
    def detect_by_contours(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        masks = self._generate_masks(frame)
        candidates: List[dict] = []
        for mask in masks:
            candidates.extend(self._collect_candidates(frame, mask))

        if self.debug:
            self.debug_last_masks = masks
            self.debug_last_candidates = candidates

        if len(candidates) < 5:
            return []

        cluster = self._cluster_candidates(frame, candidates)
        if not cluster:
            return []

        xs = [c["x"] for c in cluster]
        ys = [c["y"] for c in cluster]
        ws = [c["w"] for c in cluster]
        hs = [c["h"] for c in cluster]
        sizes = [c["size"] for c in cluster]

        avg_size = float(np.median(sizes)) if sizes else 20.0
        margin = max(20.0, avg_size * 0.8)

        x1 = max(0, int(min(xs) - margin))
        y1 = max(0, int(min(ys) - margin))
        x2 = min(frame.shape[1], int(max(x + w for x, w in zip(xs, ws)) + margin))
        y2 = min(frame.shape[0], int(max(y + h for y, h in zip(ys, hs)) + margin))

        if x2 - x1 < 40 or y2 - y1 < 40:
            return []

        completeness = min(1.0, len(cluster) / 9.0)
        size_uniform = 1.0 - min(1.0, np.std(sizes) / (avg_size + 1e-6))
        confidence = max(0.35, 0.55 * completeness + 0.45 * size_uniform)

        return [(x1, y1, x2, y2, confidence)]

    def detect_by_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if not self.use_yolo or self.yolo_model is None:
            return []
        try:
            results = self.yolo_model.predict(frame, imgsz=640, conf=self.conf_threshold, verbose=False)
            boxes = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
            boxes.sort(key=lambda item: item[4], reverse=True)
            return boxes
        except Exception as exc:
            print(f"YOLO检测出错: {exc}")
            return []

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if self.use_yolo:
            boxes = self.detect_by_yolo(frame)
            if boxes:
                return boxes
        return self.detect_by_contours(frame)

    def get_best_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        boxes = self.detect(frame)
        if boxes:
            x1, y1, x2, y2, _ = boxes[0]
            return int(x1), int(y1), int(x2), int(y2)
        return None

