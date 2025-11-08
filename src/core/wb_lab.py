"""
白平衡 + Lab 色彩空间 + CIEDE2000 颜色识别模块
优化版颜色识别，使用工业级色差公式
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math

class AdvancedColorRecognizer:
    """高级颜色识别器：白平衡 + Lab + ΔE + 会话调色板（中心色自适应）"""

    # 备用标准色（Lab空间，作为调色板未初始化时的回退）
    STANDARD_COLORS_LAB = {
        'U': np.array([95, 0, 0], dtype=np.float32),     # 白
        'R': np.array([50, 70, 50], dtype=np.float32),   # 红
        'F': np.array([60, -60, 50], dtype=np.float32),  # 绿
        'D': np.array([90, -5, 80], dtype=np.float32),   # 黄
        'L': np.array([60, 40, 60], dtype=np.float32),   # 橙
        'B': np.array([40, 20, -70], dtype=np.float32),  # 蓝
    }

    def __init__(self, use_xphoto: bool = True):
        self.use_xphoto = use_xphoto
        self.wb_algo = None
        # 会话调色板：在运行时由每面的中心贴纸更新 {'U':Lab_u, ...}
        self.palette_lab: Dict[str, np.ndarray] = {}
        try:
            if use_xphoto and hasattr(cv2, 'xphoto'):
                self.wb_algo = cv2.xphoto.createGrayworldWB()
                # 某些实现支持阈值设置，以减弱极端高饱和的影响
                if hasattr(self.wb_algo, 'setSaturationThreshold'):
                    self.wb_algo.setSaturationThreshold(0.98)
                print("✓ 使用 xphoto GrayworldWB")
        except Exception:
            self.wb_algo = None
    
    def white_balance(self, img: np.ndarray) -> np.ndarray:
        if self.wb_algo is not None:
            try:
                return self.wb_algo.balanceWhite(img)
            except Exception:
                pass
        # 简易 Gray-World 兜底
        result = img.astype(np.float32)
        avg = np.mean(result, axis=(0, 1)) + 1e-6
        gray = np.mean(avg)
        scale = gray / avg
        result *= scale
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def remove_highlights(self, img: np.ndarray, threshold: int = 240) -> np.ndarray:
        """去除高光（饱和像素）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray < threshold
        
        # 对高光区域做中值滤波
        result = img.copy()
        if not mask.all():
            result = cv2.inpaint(img, (~mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        
        return result
    
    def ciede2000(self, lab1: np.ndarray, lab2: np.ndarray) -> float:
        """CIEDE2000 实现（kL=kC=kH=1）。"""
        L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
        L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])
        avg_L = (L1 + L2) / 2.0
        C1 = math.hypot(a1, b1)
        C2 = math.hypot(a2, b2)
        avg_C = (C1 + C2) / 2.0
        G = 0.5 * (1 - math.sqrt((avg_C ** 7) / (avg_C ** 7 + 25 ** 7))) if avg_C != 0 else 0.0
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = math.hypot(a1p, b1)
        C2p = math.hypot(a2p, b2)
        avg_Cp = (C1p + C2p) / 2.0
        def atan2d(y, x):
            ang = math.degrees(math.atan2(y, x))
            return ang + 360 if ang < 0 else ang
        h1p = 0.0 if C1p == 0 else atan2d(b1, a1p)
        h2p = 0.0 if C2p == 0 else atan2d(b2, a2p)
        dLp = L2 - L1
        dCp = C2p - C1p
        if C1p * C2p == 0:
            dhp = 0.0
        else:
            dh = h2p - h1p
            if dh > 180:
                dh -= 360
            elif dh < -180:
                dh += 360
            dhp = dh
        dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))
        avg_hp = h1p + h2p
        if C1p * C2p == 0:
            avg_hp = h1p + h2p
        else:
            if abs(h1p - h2p) > 180:
                avg_hp = (h1p + h2p + 360) / 2.0 if (h1p + h2p) < 360 else (h1p + h2p - 360) / 2.0
            else:
                avg_hp = (h1p + h2p) / 2.0
        T = (
            1
            - 0.17 * math.cos(math.radians(avg_hp - 30))
            + 0.24 * math.cos(math.radians(2 * avg_hp))
            + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
            - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
        )
        Sl = 1 + (0.015 * ((avg_L - 50) ** 2)) / math.sqrt(20 + ((avg_L - 50) ** 2))
        Sc = 1 + 0.045 * avg_Cp
        Sh = 1 + 0.015 * avg_Cp * T
        delta_ro = 30 * math.exp(-(((avg_hp - 275) / 25) ** 2))
        Rc = 2 * math.sqrt((avg_Cp ** 7) / (avg_Cp ** 7 + 25 ** 7))
        Rt = -math.sin(math.radians(2 * delta_ro)) * Rc
        dE = math.sqrt(
            (dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh)
        )
        return float(dE)
    
    def _robust_lab_mean_from_tile(self, tile_bgr: np.ndarray) -> np.ndarray:
        """对一个小格子（BGR）进行稳健采样，返回 Lab 的中位值。"""
        if tile_bgr.size == 0:
            return np.array([50, 0, 0], dtype=np.float32)
        h, w = tile_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        r = int(min(w, h) * 0.33)
        Y, X = np.ogrid[:h, :w]
        mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r * r
        pixels = tile_bgr[mask]
        if len(pixels) == 0:
            pixels = tile_bgr.reshape(-1, 3)
        # 去掉亮度极端 10% + 10%
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)[:, 0, :]
        v = hsv[:, 2]
        if len(v) >= 20:
            lo, hi = np.percentile(v, 10), np.percentile(v, 90)
            keep = (v >= lo) & (v <= hi)
            pixels = pixels[keep]
        lab8 = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)[:, 0, :].astype(np.float32)
        # OpenCV 8位Lab需转换到CIE标准：L∈[0,100]，a/b≈[-128,127]
        lab_std = np.empty_like(lab8, dtype=np.float32)
        lab_std[..., 0] = lab8[..., 0] * (100.0 / 255.0)
        lab_std[..., 1] = lab8[..., 1] - 128.0
        lab_std[..., 2] = lab8[..., 2] - 128.0
        return np.median(lab_std, axis=0)

    def extract_face_labs_from_warped(self, warped_face_bgr: np.ndarray, rect_quads: List[np.ndarray]) -> List[np.ndarray]:
        """从透视展开的面提取9个小格的 Lab 值。
        rect_quads: 每个元素为4点 TL,TR,BR,BL（但实际是轴对齐矩形）。
        """
        balanced = self.white_balance(warped_face_bgr)
        labs: List[np.ndarray] = []
        for quad in rect_quads:
            x1, y1 = int(min(quad[:, 0])), int(min(quad[:, 1]))
            x2, y2 = int(max(quad[:, 0])), int(max(quad[:, 1]))
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(balanced.shape[1], x2); y2 = min(balanced.shape[0], y2)
            tile = balanced[y1:y2, x1:x2]
            labs.append(self._robust_lab_mean_from_tile(tile))
        return labs

    def update_palette(self, face_label: str, center_lab: np.ndarray, momentum: float = 0.7):
        """用中心格的 Lab 更新会话调色板（指数滑动平均）。"""
        if center_lab is None:
            return
        if face_label not in self.palette_lab:
            self.palette_lab[face_label] = center_lab.astype(np.float32)
        else:
            self.palette_lab[face_label] = (
                float(momentum) * self.palette_lab[face_label].astype(np.float32)
                + (1.0 - float(momentum)) * center_lab.astype(np.float32)
            )

    def classify_labs(self, labs: List[np.ndarray]) -> List[str]:
        """基于会话调色板（优先）或备选标准色，对每个 Lab 贴纸赋面字母标签。"""
        labels: List[str] = []
        palette = self.palette_lab if len(self.palette_lab) > 0 else self.STANDARD_COLORS_LAB
        for lab in labs:
            best_lab = None
            best_de = 1e9
            for face, proto in palette.items():
                de = self.ciede2000(lab, proto)
                if de < best_de:
                    best_de = de
                    best_lab = face
            labels.append(best_lab if best_lab is not None else 'U')
        return labels
    
    # 兼容旧接口（不再推荐）：从原图+四边形蒙版直接识别
    def recognize_face(self, img: np.ndarray, grid_quads: List[np.ndarray]) -> List[str]:
        if not grid_quads or len(grid_quads) != 9:
            return ['U'] * 9
        # 简单兜底：在多边形掩膜内部采 Lab 的均值
        img_wb = self.white_balance(img)
        labs: List[np.ndarray] = []
        for quad in grid_quads:
            mask = np.zeros(img_wb.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
            pixels = img_wb[mask > 0]
            if len(pixels) == 0:
                labs.append(np.array([50, 0, 0], dtype=np.float32))
                continue
            lab8 = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)[:, 0, :].astype(np.float32)
            lab_std = np.empty_like(lab8, dtype=np.float32)
            lab_std[..., 0] = lab8[..., 0] * (100.0 / 255.0)
            lab_std[..., 1] = lab8[..., 1] - 128.0
            lab_std[..., 2] = lab8[..., 2] - 128.0
            labs.append(np.median(lab_std, axis=0).astype(np.float32))
        return self.classify_labs(labs)
    
    def calibrate_colors(self, face_samples: Dict[str, List[np.ndarray]]):
        """
        根据实际魔方校准标准颜色
        Args:
            face_samples: {face_label: [Lab颜色样本列表]}
        """
        print("校准标准颜色...")
        for label, samples in face_samples.items():
            if samples:
                mean_color = np.mean(samples, axis=0)
                self.STANDARD_COLORS_LAB[label] = mean_color
                print(f"  {label}: Lab({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})")
