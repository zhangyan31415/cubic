"""
状态管理模块：时序缓存 + 状态校验 + 可解性检查
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import collections

class StateManager:
    """魔方状态管理器"""
    
    # 面顺序：U, R, F, D, L, B
    FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']
    
    def __init__(self):
        # 状态缓存：facelet[face][row][col] = label
        self.facelet_state = np.full((6, 3, 3), None, dtype=object)
        
        # 每个面的置信度（0-1）
        self.face_confidence = np.zeros(6)
        
        # 每个面的最后更新时间
        self.face_update_time = np.zeros(6)
        
        # 稳定状态计数器
        self.stable_count = 0
        self.last_state_string = None
    
    def update_face(self, face_idx: int, labels: List[str], confidence: float = 1.0):
        """
        更新一个面的状态
        Args:
            face_idx: 面索引 (0=U, 1=R, 2=F, 3=D, 4=L, 5=B)
            labels: 9个标签的列表
            confidence: 置信度 (0-1)
        """
        if len(labels) != 9:
            return False
        
        # 更新状态
        for i in range(3):
            for j in range(3):
                self.facelet_state[face_idx, i, j] = labels[i * 3 + j]
        
        self.face_confidence[face_idx] = confidence
        self.face_update_time[face_idx] += 1
        
        return True
    
    def build_state_string(self) -> Optional[str]:
        """
        构建54字符的状态串
        Returns:
            状态串或None（如果不完整）
        """
        state_list = []
        
        for face_idx in range(6):
            for i in range(3):
                for j in range(3):
                    label = self.facelet_state[face_idx, i, j]
                    if label is None:
                        return None  # 状态不完整
                    state_list.append(label)
        
        return ''.join(state_list)
    
    def check_counts(self, state_string: str) -> Tuple[bool, Dict[str, int]]:
        """
        检查状态串的计数是否正确（每色9个）
        """
        if len(state_string) != 54:
            return False, {}
        
        cnt = collections.Counter(state_string)
        valid = all(cnt.get(ch, 0) == 9 for ch in "URFDLB")
        
        return valid, dict(cnt)
    
    def check_solvability(self, state_string: str) -> bool:
        """
        检查魔方是否可解：先计数，再用kociemba做一致性预检。
        """
        valid, _ = self.check_counts(state_string)
        if not valid:
            return False
        try:
            import kociemba
            _ = kociemba.solve(state_string)
            return True
        except Exception:
            return False
    
    def get_completeness(self) -> float:
        """获取状态完整度（0-1），按面计算"""
        # 计算完整的面数（每个面9个格子都有值才算完整）
        complete_faces = 0
        for face_idx in range(6):
            face_data = self.facelet_state[face_idx]
            if np.all(face_data != None):
                complete_faces += 1
        
        # 返回完整面的比例（每完成一个面 = 16.67%）
        return complete_faces / 6.0
    
    def get_face_colors(self, face_label: str) -> Optional[list]:
        """
        获取指定面的3x3颜色矩阵
        Args:
            face_label: 面标签 ('U','R','F','D','L','B')
        Returns:
            3x3矩阵，每个元素是颜色标签，None表示未知
        """
        face_idx = self.FACE_ORDER.index(face_label) if face_label in self.FACE_ORDER else None
        if face_idx is None:
            return None
        
        face_data = self.facelet_state[face_idx]
        colors = []
        for i in range(3):
            row = []
            for j in range(3):
                label = face_data[i, j]
                row.append(label if label is not None else '?')
            colors.append(row)
        return colors
    
    def get_face_completeness(self) -> List[float]:
        """获取每个面的完整度"""
        completeness = []
        for face_idx in range(6):
            filled = np.sum(self.facelet_state[face_idx] != None)
            completeness.append(filled / 9.0)
        return completeness
    
    def reset(self):
        """重置状态"""
        self.facelet_state = np.full((6, 3, 3), None, dtype=object)
        self.face_confidence = np.zeros(6)
        self.face_update_time = np.zeros(6)
        self.stable_count = 0
        self.last_state_string = None
    
    def get_stable_state(self) -> Optional[str]:
        """
        获取稳定的状态（如果所有面都已更新且通过校验）
        """
        # 检查是否所有面都已填充
        if self.get_completeness() < 1.0:
            return None
        
        state_string = self.build_state_string()
        if state_string is None:
            return None
        
        # 检查计数
        if not self.check_solvability(state_string):
            return None
        
        # 如果状态与上次相同，增加稳定计数
        if state_string == self.last_state_string:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_state_string = state_string
        
        # 需要稳定至少3帧
        if self.stable_count >= 3:
            return state_string
        
        return None

    # --- 新增：中心唯一性快速检查（基于当前字母状态） ---
    def centers_ok(self) -> bool:
        """检查六个中心贴（位置[1,1]）是否各不相同，且与面序一致（可选）。"""
        centers = []
        for idx, face in enumerate(self.FACE_ORDER):
            c = self.facelet_state[idx, 1, 1]
            if c is None:
                return False
            centers.append(c)
        return len(set(centers)) == 6

    # --- 新增：完整可解性预检（计数 + Kociemba 预检） ---
    def check_solvability_full(self) -> bool:
        state_str = self.build_state_string()
        if state_str is None:
            return False
        ok, _ = self.check_counts(state_str)
        if not ok:
            return False
        try:
            import kociemba
            _ = kociemba.solve(state_str)
            return True
        except Exception:
            return False
