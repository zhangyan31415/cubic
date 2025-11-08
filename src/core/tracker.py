"""
跟踪模块：SORT/ByteTrack 多目标跟踪
简化版：单目标场景使用卡尔曼滤波
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from filterpy.kalman import KalmanFilter

class SimpleTracker:
    """简化的单目标跟踪器（基于卡尔曼滤波）"""
    
    def __init__(self):
        # 8维状态：[x, y, w, h, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.initialized = False
        self.hit_streak = 0
        self.time_since_update = 0
        self.age = 0
    
    def init_kf(self, bbox: Tuple[int, int, int, int]):
        """初始化卡尔曼滤波器"""
        x1, y1, x2, y2 = bbox
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        
        # 状态转移矩阵（匀速模型）
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差
        self.kf.Q = np.eye(8) * 0.1
        
        # 观测噪声协方差
        self.kf.R = np.eye(4) * 10
        
        # 初始状态
        self.kf.x = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # 初始协方差
        self.kf.P = np.eye(8) * 1000
        
        self.initialized = True
        self.hit_streak = 1
        self.time_since_update = 0
    
    def update(self, bbox: Optional[Tuple[int, int, int, int]]):
        """更新跟踪器"""
        self.age += 1
        
        if bbox is None:
            self.time_since_update += 1
            # 预测
            self.kf.predict()
        else:
            self.time_since_update = 0
            self.hit_streak += 1
            
            x1, y1, x2, y2 = bbox
            z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)
            
            if not self.initialized:
                self.init_kf(bbox)
            else:
                self.kf.update(z)
    
    def predict(self) -> Tuple[int, int, int, int]:
        """预测下一帧的位置"""
        if not self.initialized:
            return None
        
        self.kf.predict()
        x, y, w, h = self.kf.x[:4]
        
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        return (x1, y1, x2, y2)
    
    def get_state(self) -> Optional[Tuple[int, int, int, int]]:
        """获取当前状态"""
        if not self.initialized:
            return None
        
        x, y, w, h = self.kf.x[:4]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        return (x1, y1, x2, y2)
    
    def is_tracked(self) -> bool:
        """判断是否正在跟踪"""
        return self.initialized and self.time_since_update < 5

