"""
颜色识别模块：自动白平衡 + KMeans聚类 + 匈牙利匹配
无需手动校准，自动识别6种颜色
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

class ColorRecognizer:
    """自动颜色识别器"""
    
    # 标准魔方6色（RGB）
    STANDARD_COLORS = {
        'U': np.array([255, 255, 255]),  # 白色
        'R': np.array([255, 0, 0]),      # 红色
        'F': np.array([0, 255, 0]),     # 绿色
        'D': np.array([255, 255, 0]),   # 黄色
        'L': np.array([255, 165, 0]),   # 橙色
        'B': np.array([0, 0, 255]),      # 蓝色
    }
    
    def __init__(self):
        self.n_clusters = 6
    
    def gray_world_awb(self, img: np.ndarray) -> np.ndarray:
        """
        Gray-World 自动白平衡
        假设整幅图像的平均反射率是灰色的
        """
        img_float = img.astype(np.float32)
        
        # 计算每个通道的平均值
        avg_b = np.mean(img_float[:, :, 0])
        avg_g = np.mean(img_float[:, :, 1])
        avg_r = np.mean(img_float[:, :, 2])
        
        # 计算全局平均
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        # 避免除零
        if avg_b > 0:
            img_float[:, :, 0] = img_float[:, :, 0] * (avg_gray / avg_b)
        if avg_g > 0:
            img_float[:, :, 1] = img_float[:, :, 1] * (avg_gray / avg_g)
        if avg_r > 0:
            img_float[:, :, 2] = img_float[:, :, 2] * (avg_gray / avg_r)
        
        # 限制到[0, 255]
        img_balanced = np.clip(img_float, 0, 255).astype(np.uint8)
        return img_balanced
    
    def extract_facelet_colors(self, roi: np.ndarray, grid_quads: List[np.ndarray]) -> List[np.ndarray]:
        """
        从9个网格中提取颜色
        Returns:
            List of RGB colors (9个)
        """
        colors = []
        
        for quad in grid_quads:
            # 获取四边形区域
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [quad], 255)
            
            # 提取区域内的颜色（取中心区域，避免边缘干扰）
            masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
            
            # 计算非零像素的平均颜色
            pixels = masked_roi[mask > 0]
            if len(pixels) > 0:
                # 取中心50%的像素（排除边缘）
                if len(pixels) > 10:
                    pixels = pixels[len(pixels)//4:-len(pixels)//4]
                color = np.mean(pixels, axis=0)
                colors.append(color)
            else:
                colors.append(np.array([128, 128, 128]))  # 默认灰色
        
        return colors
    
    def cluster_colors(self, all_colors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对所有颜色进行KMeans聚类
        Args:
            all_colors: List of RGB colors (可能来自多个面)
        Returns:
            (cluster_centers, labels)
        """
        if len(all_colors) < self.n_clusters:
            return None, None
        
        colors_array = np.array(all_colors, dtype=np.float32)
        
        # 转换到Lab色彩空间（更好的聚类效果）
        colors_lab = cv2.cvtColor(colors_array.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)
        colors_lab = colors_lab.reshape(-1, 3)
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors_lab)
        centers_lab = kmeans.cluster_centers_
        
        # 转换回RGB
        centers_lab_reshaped = centers_lab.reshape(1, -1, 3)
        centers_rgb = cv2.cvtColor(centers_lab_reshaped.astype(np.uint8), cv2.COLOR_LAB2RGB)
        centers_rgb = centers_rgb.reshape(-1, 3)
        
        return centers_rgb, labels
    
    def match_to_standard(self, cluster_centers: np.ndarray) -> Dict[int, str]:
        """
        使用匈牙利算法将聚类中心匹配到标准颜色
        Returns:
            Dict mapping cluster_index -> label ('U', 'R', 'F', 'D', 'L', 'B')
        """
        if cluster_centers is None or len(cluster_centers) != self.n_clusters:
            return {}
        
        # 构建成本矩阵（颜色距离）
        cost_matrix = np.zeros((self.n_clusters, self.n_clusters))
        standard_list = list(self.STANDARD_COLORS.values())
        
        for i, cluster_color in enumerate(cluster_centers):
            for j, std_color in enumerate(standard_list):
                # 计算欧氏距离
                cost_matrix[i, j] = np.linalg.norm(cluster_color - std_color)
        
        # 匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 构建映射
        label_names = list(self.STANDARD_COLORS.keys())
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            mapping[i] = label_names[j]
        
        return mapping
    
    def recognize_face(self, roi: np.ndarray, grid_quads: List[np.ndarray]) -> List[str]:
        """
        识别一个面的9个色块
        Returns:
            List of 9 labels ('U', 'R', 'F', 'D', 'L', 'B')
        """
        # 自动白平衡
        roi_balanced = self.gray_world_awb(roi)
        
        # 提取颜色
        colors = self.extract_facelet_colors(roi_balanced, grid_quads)
        
        # 如果颜色太少，返回默认值
        if len(colors) != 9:
            return ['U'] * 9
        
        # 简单方法：直接与标准颜色匹配（每个色块）
        labels = []
        for color in colors:
            best_label = 'U'
            best_dist = float('inf')
            
            for label, std_color in self.STANDARD_COLORS.items():
                dist = np.linalg.norm(color - std_color)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
            
            labels.append(best_label)
        
        return labels
    
    def recognize_multiple_faces(self, face_colors_list: List[List[np.ndarray]]) -> List[List[str]]:
        """
        识别多个面（使用全局聚类）
        Args:
            face_colors_list: List of face colors, each face has 9 colors
        Returns:
            List of face labels, each face has 9 labels
        """
        # 收集所有颜色
        all_colors = []
        for face_colors in face_colors_list:
            all_colors.extend(face_colors)
        
        if len(all_colors) < self.n_clusters:
            # 如果颜色太少，使用单面识别
            return [self.recognize_face(None, None) for _ in face_colors_list]
        
        # 全局聚类
        cluster_centers, labels = self.cluster_colors(all_colors)
        
        if cluster_centers is None:
            return [self.recognize_face(None, None) for _ in face_colors_list]
        
        # 匹配到标准颜色
        mapping = self.match_to_standard(cluster_centers)
        
        # 分配标签
        result = []
        idx = 0
        for face_colors in face_colors_list:
            face_labels = []
            for _ in range(9):
                cluster_idx = labels[idx]
                label = mapping.get(cluster_idx, 'U')
                face_labels.append(label)
                idx += 1
            result.append(face_labels)
        
        return result
    
    def recognize_face_from_warped(self, warped_face: np.ndarray) -> List[str]:
        """
        从透视展开的标准正方形面识别颜色
        Args:
            warped_face: 300x300的标准正方形图像
        Returns:
            List of 9 labels ('U', 'R', 'F', 'D', 'L', 'B')
        """
        # 自动白平衡
        warped_balanced = self.gray_world_awb(warped_face)
        
        # 将300x300分成3x3网格，每格100x100
        cell_size = 100
        labels = []
        
        for i in range(3):
            for j in range(3):
                # 提取中心区域（避免边缘）
                y1, y2 = i * cell_size + 20, (i + 1) * cell_size - 20
                x1, x2 = j * cell_size + 20, (j + 1) * cell_size - 20
                
                cell = warped_balanced[y1:y2, x1:x2]
                
                if cell.size > 0:
                    # 计算平均颜色
                    color = np.mean(cell.reshape(-1, 3), axis=0)
                    
                    # 匹配到标准颜色
                    best_label = 'U'
                    best_dist = float('inf')
                    
                    for label, std_color in self.STANDARD_COLORS.items():
                        dist = np.linalg.norm(color - std_color)
                        if dist < best_dist:
                            best_dist = dist
                            best_label = label
                    
                    labels.append(best_label)
                else:
                    labels.append('U')
        
        return labels

