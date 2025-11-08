"""
3D叠加和箭头指引模块
在视频上叠加3D魔方模型和操作箭头
"""
import cv2
import numpy as np
from typing import Optional, Tuple

class OverlayRenderer:
    """3D叠加渲染器"""
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.K = camera_matrix
        self.dist = dist_coeffs
        
        # 魔方3D模型参数
        self.cube_size = 1.0
        self.face_size = self.cube_size / 3.0
    
    def draw_cube_wireframe(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
        """绘制魔方线框"""
        # 定义立方体的8个顶点
        vertices_3d = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ], dtype=np.float32) * self.cube_size
        
        # 投影到2D
        vertices_2d, _ = cv2.projectPoints(vertices_3d, rvec, tvec, self.K, self.dist)
        vertices_2d = vertices_2d.reshape(-1, 2).astype(int)
        
        # 定义12条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 前面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 后面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 连接
        ]
        
        # 绘制边
        for edge in edges:
            pt1 = tuple(vertices_2d[edge[0]])
            pt2 = tuple(vertices_2d[edge[1]])
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    
    def draw_face_outline(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                          face: str):
        """绘制指定面的轮廓"""
        # 面的法向量和中心
        face_configs = {
            'U': ([0, 0.5, 0], [0, 1, 0], [1, 0, 0]),
            'D': ([0, -0.5, 0], [0, -1, 0], [1, 0, 0]),
            'F': ([0, 0, 0.5], [0, 0, 1], [1, 0, 0]),
            'B': ([0, 0, -0.5], [0, 0, -1], [1, 0, 0]),
            'L': ([-0.5, 0, 0], [-1, 0, 0], [0, 0, 1]),
            'R': ([0.5, 0, 0], [1, 0, 0], [0, 0, 1]),
        }
        
        if face not in face_configs:
            return
        
        center, normal, right = face_configs[face]
        center = np.array(center, dtype=np.float32) * self.cube_size
        
        # 计算面的4个角点
        up = np.cross(normal, right)
        up = up / np.linalg.norm(up)
        right = np.array(right) / np.linalg.norm(right)
        
        half_size = self.face_size * 1.5  # 整个面的尺寸
        corners_3d = [
            center + (-half_size) * right + (-half_size) * up,
            center + (half_size) * right + (-half_size) * up,
            center + (half_size) * right + (half_size) * up,
            center + (-half_size) * right + (half_size) * up,
        ]
        
        corners_3d = np.array(corners_3d, dtype=np.float32)
        corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, self.K, self.dist)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)
        
        # 绘制轮廓
        cv2.polylines(frame, [corners_2d], True, (0, 255, 0), 3)
    
    def draw_rotation_arrow(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                           move: str):
        """
        绘制旋转箭头（用户视角）
        Args:
            move: 移动指令，如 'R', "U'", 'F2'
        """
        if not move:
            return
        
        face = move[0]
        is_prime = len(move) > 1 and move[1] == "'"
        is_double = len(move) > 1 and move[1] == '2'
        
        # 绘制面的轮廓
        self.draw_face_outline(frame, rvec, tvec, face)
        
        # 计算箭头位置（在面的中心）
        face_centers = {
            'U': np.array([0, 0.5, 0], dtype=np.float32),
            'D': np.array([0, -0.5, 0], dtype=np.float32),
            'F': np.array([0, 0, 0.5], dtype=np.float32),
            'B': np.array([0, 0, -0.5], dtype=np.float32),
            'L': np.array([-0.5, 0, 0], dtype=np.float32),
            'R': np.array([0.5, 0, 0], dtype=np.float32),
        }
        
        if face not in face_centers:
            return
        
        center_3d = face_centers[face] * self.cube_size
        
        # 投影到2D
        center_2d, _ = cv2.projectPoints(
            center_3d.reshape(1, -1), rvec, tvec, self.K, self.dist
        )
        center_2d = center_2d[0][0].astype(int)
        
        # 计算箭头方向（根据旋转方向）
        # 这里简化处理：根据面的法向量和旋转方向计算
        arrow_length = 50
        arrow_angle = 0  # 默认角度
        
        # 根据旋转方向调整角度
        if is_prime:
            arrow_angle = 180  # 逆时针
        elif is_double:
            arrow_angle = 90  # 180度
        
        # 绘制箭头
        arrow_color = (0, 255, 255) if is_double else (0, 255, 0)
        thickness = 3
        
        # 绘制圆形箭头（表示旋转）
        radius = 30
        cv2.circle(frame, tuple(center_2d), radius, arrow_color, thickness)
        
        # 绘制箭头指示方向
        import math
        angle_rad = math.radians(arrow_angle)
        end_x = int(center_2d[0] + radius * math.cos(angle_rad))
        end_y = int(center_2d[1] + radius * math.sin(angle_rad))
        
        cv2.arrowedLine(
            frame, tuple(center_2d), (end_x, end_y),
            arrow_color, thickness, tipLength=0.3
        )
        
        # 添加文字说明
        move_text = move
        if is_double:
            move_text = f"{face}2 (180°)"
        elif is_prime:
            move_text = f"{face}' (逆时针)"
        else:
            move_text = f"{face} (顺时针)"
        
        text_pos = (center_2d[0] - 40, center_2d[1] + 60)
        cv2.putText(
            frame, move_text, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2
        )
    
    def render_overlay(self, frame: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                      next_move: Optional[str] = None):
        """
        渲染完整的叠加层
        """
        # 绘制魔方线框
        self.draw_cube_wireframe(frame, rvec, tvec)
        
        # 如果有下一步操作，绘制箭头
        if next_move:
            self.draw_rotation_arrow(frame, rvec, tvec, next_move)
        
        return frame

