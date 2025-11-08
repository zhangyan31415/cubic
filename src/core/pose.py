"""
位姿估计模块：solvePnP + 网格检测
支持无标记位姿估计（基于魔方几何特征）
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

class PoseEstimator:
    """魔方位姿估计器"""
    
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        初始化位姿估计器
        Args:
            camera_matrix: 相机内参矩阵 K (3x3)
            dist_coeffs: 畸变系数
        """
        self.K = camera_matrix
        self.dist = dist_coeffs
        
        # 魔方3D模型（单位立方体，边长=1）
        # 定义6个面的角点（相对于魔方中心）
        self.cube_size = 1.0
        self.face_size = self.cube_size / 3.0  # 每个小方块的大小
        
        # 构建3D点：6个面的中心点（用于粗略位姿估计）
        self.face_centers_3d = np.array([
            [0, 0, 0.5],      # 前面 (F)
            [0, 0, -0.5],     # 后面 (B)
            [0, 0.5, 0],      # 上面 (U)
            [0, -0.5, 0],     # 下面 (D)
            [-0.5, 0, 0],     # 左面 (L)
            [0.5, 0, 0],      # 右面 (R)
        ], dtype=np.float32)
    
    def detect_face_grid(self, roi: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        检测ROI中的3x3网格（一个面的9个色块）
        Returns:
            List of 9 quadrilateral corners (4 points each)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 形态学操作清理
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选出接近正方形的轮廓
        squares = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < roi.shape[0] * roi.shape[1] * 0.01:  # 太小
                continue
            if area > roi.shape[0] * roi.shape[1] * 0.3:  # 太大
                continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.7 < aspect_ratio < 1.3:  # 接近正方形
                    squares.append(approx)
        
        # 如果找到9个或更多，选择最合适的9个
        if len(squares) >= 9:
            # 按面积排序，选择中等大小的9个
            squares.sort(key=cv2.contourArea)
            # 取中间9个
            start = (len(squares) - 9) // 2
            squares = squares[start:start+9]
            
            # 按位置排序（从上到下，从左到右）
            def get_center(approx):
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
                return (0, 0)
            
            centers = [get_center(sq) for sq in squares]
            # 按y坐标分组，然后每组内按x坐标排序
            rows = {}
            for i, (cx, cy) in enumerate(centers):
                row = cy // (roi.shape[0] // 3)
                if row not in rows:
                    rows[row] = []
                rows[row].append((i, cx, cy))
            
            sorted_indices = []
            for row in sorted(rows.keys()):
                row_items = sorted(rows[row], key=lambda x: x[1])
                sorted_indices.extend([i for i, _, _ in row_items])
            
            return [squares[i] for i in sorted_indices]
        
        return None
    
    def estimate_pose_from_grid(self, roi: np.ndarray, grid_quads: List[np.ndarray], 
                                 roi_offset: Tuple[int, int] = (0, 0)) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从检测到的网格估计位姿
        Args:
            roi: ROI图像
            grid_quads: 9个四边形的角点
            roi_offset: ROI在原始图像中的偏移 (x, y)
        Returns:
            (rvec, tvec) 或 None
        """
        if len(grid_quads) != 9:
            return None
        
        # 构建2D-3D对应点
        # 假设这是前面（F面），3D坐标在z=0.5平面上
        obj_points = []
        img_points = []
        
        # 计算每个小方块的中心点（2D和3D）
        for i in range(3):
            for j in range(3):
                # 3D点（相对于魔方中心）
                x_3d = (j - 1) * self.face_size
                y_3d = (1 - i) * self.face_size  # 注意y轴方向
                z_3d = self.cube_size / 2
                obj_points.append([x_3d, y_3d, z_3d])
                
                # 2D点（四边形中心）
                quad = grid_quads[i * 3 + j]
                M = cv2.moments(quad)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + roi_offset[0]
                    cy = int(M["m01"] / M["m00"]) + roi_offset[1]
                    img_points.append([cx, cy])
        
        if len(obj_points) < 6:  # 至少需要6个点
            return None
        
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        
        # 使用solvePnP估计位姿
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            return (rvec, tvec)
        return None
    
    def project_points(self, points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """将3D点投影到2D图像平面"""
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        return points_2d.reshape(-1, 2)
    
    def get_default_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """获取默认相机内参（如果未标定）"""
        fx = fy = width * 0.8
        cx = width / 2
        cy = height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

