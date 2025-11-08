"""
主程序：全自动魔方识别与引导系统
YOLO11分割模型 → 精确检测 → 识别状态 → 求解 → 3D箭头指引
"""
import cv2
import numpy as np
import time
import subprocess
import threading
from typing import Optional

from src.detectors.roboflow_detector import RoboflowRubikDetector
from src.core.tracker import SimpleTracker
from src.core.pose import PoseEstimator
from src.core.state import StateManager
from src.core.solver import RubikSolver
from src.core.overlay import OverlayRenderer
from src.core.mini_cube_hud import MiniCubeHUD

# 中文绘字工具
from PIL import Image, ImageDraw, ImageFont

def speak(text):
    """异步语音播报（带防抖，避免回音）"""
    # 防抖：同一句话2秒内不重复
    now = time.time()
    
    if not hasattr(speak, '_last_text'):
        speak._last_text = None
        speak._last_time = 0
    
    # 如果是同一句话且时间间隔太短，跳过
    if speak._last_text == text and (now - speak._last_time) < 2.0:
        return
    
    speak._last_text = text
    speak._last_time = now
    
    def _speak():
        try:
            subprocess.run(['say', '-v', 'Ting-Ting', text], check=False, timeout=3)
        except Exception:
            pass
    
    threading.Thread(target=_speak, daemon=True).start()

def put_text_cn(img_bgr, text, xy=(20, 40), font_size=24, color=(0, 255, 0)):
    """中文文字绘制"""
    font_candidates = (
        "PingFang.ttc", "STHeiti Medium.ttc", "Hiragino Sans GB.ttc",
        "NotoSansCJK-Regular.ttc", "SimSun.ttc", "Microsoft YaHei.ttf"
    )
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            continue
    
    if font is None:
        cv2.putText(img_bgr, text.encode('ascii', 'ignore').decode('ascii'),
                   xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img_bgr
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def move_to_cn(move: str) -> str:
    """移动指令转中文"""
    if not move:
        return ""
    
    face_map = {'U': '上', 'D': '下', 'L': '左', 'R': '右', 'F': '前', 'B': '后'}
    face = face_map.get(move[0], move[0])
    
    if len(move) > 1:
        if move[1] == "'":
            return f"{face}面逆时针90°"
        elif move[1] == '2':
            return f"{face}面180°"
    
    return f"{face}面顺时针90°"

class RubikAutoGuide:
    """全自动魔方引导系统"""
    
    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        
        # 获取画面尺寸
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("无法打开摄像头")
        
        h, w = frame.shape[:2]
        
        # 初始化相机参数（改进版：区分fx和fy）
        # TODO: 使用棋盘格标定获得真实内参以提高PnP精度
        fx = w * 0.9  # 横向焦距
        fy = h * 0.9  # 纵向焦距
        cx = w / 2.0  # 主点x
        cy = h / 2.0  # 主点y
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        print(f"📷 相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # 初始化各个模块
        # 使用YOLO11分割模型（精确轮廓检测 + 高精度识别）
        from src.detectors.roboflow_detector import LocalYOLODetector
        import torch
        
        # 自动选择设备
        if torch.backends.mps.is_available():
            device = 'mps'
            half = False  # MPS不支持FP16
        elif torch.cuda.is_available():
            device = '0'
            half = True
        else:
            device = 'cpu'
            half = False
        
        self.detector = LocalYOLODetector(
            model_path="models/rubik_cube_yolo11_seg.pt",
            device=device,
            half=half
        )
        # YOLO11-seg 模型特性：
        # ✓ mAP50: 97.9% (所有颜色识别率 > 96%)
        # ✓ 推理速度: 0.9ms/张 (比YOLOv8快30%)
        # ✓ 支持精确分割：返回轮廓点而非简单边界框
        # ✓ 模型大小: 45.2MB (轻量高效)
        
        # 色块类别映射（根据训练数据）
        self.facelet_classes = {
            0: 'Blue',
            1: 'Center', 
            2: 'Face',
            3: 'Green',
            4: 'Orange',
            5: 'Red',
            6: 'White',
            7: 'Yellow'
        }
        
        # 面名称到索引的映射
        self.FACE_TO_IDX = {'U':0, 'R':1, 'F':2, 'D':3, 'L':4, 'B':5}
        
        self.current_facelets = []  # 当前检测到的色块
        self.frame_id = 0  # 帧计数器（用于检测分频）
        
        self.tracker = SimpleTracker()
        self.pose_estimator = PoseEstimator(self.camera_matrix, self.dist_coeffs)
        # YOLO11直接返回颜色，不需要额外的颜色识别器
        self.state_manager = StateManager()
        self.solver = RubikSolver()
        self.overlay = OverlayRenderer(self.camera_matrix, self.dist_coeffs)
        self.hud = MiniCubeHUD(size=220)  # 3D迷你魔方HUD
        
        # 状态变量
        self.current_bbox = None
        self.current_rvec = None
        self.current_tvec = None
        self.last_rvec = None  # 保存上一帧的rvec
        self.last_tvec = None
        self._last_completeness = 0.0  # 上次的完整度（用于检测变化）
        self.last_solve_time = 0
        self.last_move_time = 0
        self.move_index = 0
        self.solution_moves = []
        self.last_state_string = None
        self._last_move_spoken = None  # 上次播报的步骤（防重复）
        self._last_color_info = None  # 上次识别的颜色信息（用于显示）
        self.stable_frames = 0
        
        # 性能统计
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # 颜色-面的双向映射（通过“中心块”在线自校准）
        # face_to_color: 'U'|'D'|'L'|'R'|'F'|'B' -> 'White'|'Yellow'|...
        # color_to_face: 颜色 -> 面（当六色都确定后，分类将依赖此映射以避免抖动）
        self.face_to_color = {}
        self.color_to_face = {}
        
        # 位姿平滑（减小3D HUD 抖动）
        self.rvec_smooth = None
        self.rvec_smooth_alpha = 0.25  # 低通滤波权重（0~1，越大越快跟随）
    
    def _select_face_cluster(self, detections):
        """从色块检测结果中选出最可能属于同一面的聚类（最多9个）。
        detections: List[dict] - 每个dict包含 x1,y1,x2,y2,conf,cls,points(可选)
        返回: 该聚类的子列表（保持原字段），或 None
        """
        if not detections:
            return None
        # 计算中心和尺寸
        items = []
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            size = 0.5 * ((x2 - x1) + (y2 - y1))
            items.append((cx, cy, size))
        if not items:
            return None
        sizes = [s for _, _, s in items]
        med_size = max(8.0, float(np.median(sizes)))
        eps = med_size * 1.8
        n = len(items)
        # 构建基于距离的连通分量
        visited = [False] * n
        components = []
        for i in range(n):
            if visited[i]:
                continue
            # BFS
            queue = [i]
            visited[i] = True
            comp = [i]
            while queue:
                u = queue.pop(0)
                ux, uy, us = items[u]
                for v in range(n):
                    if visited[v]:
                        continue
                    vx, vy, vs = items[v]
                    # 阈值使用两者尺寸的较大值
                    th = 1.8 * max(us, vs)
                    if (ux - vx) ** 2 + (uy - vy) ** 2 <= th ** 2:
                        visited[v] = True
                        queue.append(v)
                        comp.append(v)
            components.append(comp)
        # 选择最大且形状更接近3x3的组件
        best_comp = None
        best_score = -1e9
        for comp in components:
            if len(comp) < 5:
                continue
            # 计算边界框与长宽比、尺寸方差
            xs = [detections[i]['x1'] for i in comp] + [detections[i]['x2'] for i in comp]
            ys = [detections[i]['y1'] for i in comp] + [detections[i]['y2'] for i in comp]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            aspect = (w + 1e-6) / (h + 1e-6)
            size_std = float(np.std([items[i][2] for i in comp]))
            # 评分：数量优先，其次形状接近正方与尺寸一致性
            score = len(comp) * 2.0 - 0.6 * abs(aspect - 1.0) - 0.02 * size_std
            if score > best_score:
                best_score = score
                best_comp = comp
        if not best_comp:
            return None
        # 只取最多9个，优先靠近中心的
        cx_all = float(np.mean([items[i][0] for i in best_comp]))
        cy_all = float(np.mean([items[i][1] for i in best_comp]))
        best_comp_sorted = sorted(best_comp, key=lambda i: (items[i][0]-cx_all)**2 + (items[i][1]-cy_all)**2)
        keep_idx = best_comp_sorted[:min(9, len(best_comp_sorted))]
        return [detections[i] for i in keep_idx]

    def _order_tl_tr_br_bl(self, pts):
        """将点排序为 TL, TR, BR, BL。pts: Nx2 或列表。"""
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
            return None
        if pts.shape[0] > 4:
            rect = cv2.minAreaRect(pts)
            pts = cv2.boxPoints(rect).astype(np.float32)
        s = pts.sum(1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
        return np.stack([tl, tr, br, bl], axis=0)

    def _outer_quad_from_facelets(self, detections, expand=1.06):
        if not detections:
            return None
        corners = []
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            corners.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        quad = self._order_tl_tr_br_bl(corners)
        if quad is None:
            return None
        c = quad.mean(0)
        quad = c + (quad - c) * float(expand)
        return quad

    def _warp_face_to_square(self, roi, quad_xy, out_size=300):
        if isinstance(quad_xy, np.ndarray) and quad_xy.shape[0] == 4:
            quad = quad_xy.astype(np.float32)
        else:
            quad = np.array(self._order_tl_tr_br_bl(quad_xy), dtype=np.float32)
        dst = np.float32([[0,0], [out_size-1,0], [out_size-1,out_size-1], [0,out_size-1]])
        H = cv2.getPerspectiveTransform(quad, dst)
        face = cv2.warpPerspective(roi, H, (out_size, out_size), flags=cv2.INTER_LINEAR)
        return face, H

    def _build_uniform_quads(self, out_size=300, margin=20):
        """在 out_size×out_size 空间内构造 3×3 等分网格的 9 个四边形（TL,TR,BR,BL）。"""
        cs = out_size // 3
        quads = []
        for i in range(3):
            for j in range(3):
                x1 = j * cs + margin
                y1 = i * cs + margin
                x2 = (j + 1) * cs - margin
                y2 = (i + 1) * cs - margin
                quad = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32)
                quads.append(quad)
        return quads

    def detect_and_track(self, frame: np.ndarray) -> Optional[tuple]:
        """检测和跟踪魔方（YOLO11很快，每帧检测）"""
        self.frame_id += 1
        
        # YOLO11推理只需0.9ms，每帧都检测！
        detections = self.detector.detect(frame, conf_threshold=0.5)
        
        if not detections:
            self.tracker.update(None)
            self.current_facelets = []
            self.current_bbox = None
            return None
        
        # 保存检测到的色块（用于状态识别）
        self.current_facelets = detections
        
        # 先选出最可能的同一面聚类
        cluster = self._select_face_cluster(detections)
        src_for_bbox = cluster if cluster else detections
        x1 = min(det['x1'] for det in src_for_bbox)
        y1 = min(det['y1'] for det in src_for_bbox)
        x2 = max(det['x2'] for det in src_for_bbox)
        y2 = max(det['y2'] for det in src_for_bbox)
        
        # 扩展边界
        margin = 30
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        bbox = (x1, y1, x2, y2)
        self.tracker.update(bbox)
        self.current_bbox = bbox
        return bbox
    
    def _visible_face_from_pose(self, rvec) -> Optional[str]:
        """根据位姿判断当前可见的面"""
        if rvec is None:
            return None
        
        R, _ = cv2.Rodrigues(rvec)
        
        # 魔方物体坐标系的6个面法向（与PnP模型一致）
        FACE_NORMALS_OBJ = {
            'F': np.array([0, 0,  1.0]),   # 正 Z
            'B': np.array([0, 0, -1.0]),
            'U': np.array([0, 1.0, 0]),   # 正 Y
            'D': np.array([0,-1.0, 0]),
            'R': np.array([1.0, 0, 0]),   # 正 X
            'L': np.array([-1.0, 0, 0]),
        }
        
        # 映射到相机坐标，z分量最大的就是正对摄像头的面
        best_face, best_z = None, -1e9
        for label, normal in FACE_NORMALS_OBJ.items():
            n_cam = R @ normal
            z = n_cam[2]
            if z > best_z:
                best_face, best_z = label, z
        
        return best_face
    
    def _draw_axes_debug(self, img, rvec, tvec, K, dist):
        """绘制坐标轴用于调试（X红, Y绿, Z蓝）"""
        axis = np.float32([[0,0,0], [0.03,0,0], [0,0.03,0], [0,0,0.05]])
        pts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
        o, x, y, z = [tuple(map(int, p.ravel())) for p in pts]
        cv2.line(img, o, x, (0, 0, 255), 2)  # X 红
        cv2.line(img, o, y, (0, 255, 0), 2)  # Y 绿
        cv2.line(img, o, z, (255, 0, 0), 2)  # Z 蓝
    
    # 上面重写了排序/外接四角/warp 辅助函数
    
    def _order_facelets_3x3(self, dets):
        """将9个检测框按3×3空间位置排序（TL→BR）"""
        if len(dets) < 9:
            return None
        
        # 计算中心点
        centers = np.array([[(d['x1']+d['x2'])/2., (d['y1']+d['y2'])/2.] for d in dets[:9]], np.float32)
        mu = centers.mean(0)
        
        # PCA主轴和次轴
        _, _, Vt = np.linalg.svd(centers - mu)
        ax = Vt[0]  # 主轴（通常是横向）
        ay = Vt[1]  # 次轴（通常是纵向）
        
        # 保证右手坐标系（图像y轴向下）
        if np.cross(np.append(ax, 0), np.append(ay, 0))[2] < 0:
            ay = -ay
        
        # 投影到主轴和次轴
        u = (centers - mu) @ ax  # 横向坐标
        v = (centers - mu) @ ay  # 纵向坐标
        
        # 分成3行3列
        row_bounds = np.percentile(v, [33.33, 66.67])
        col_bounds = np.percentile(u, [33.33, 66.67])
        
        rows = np.digitize(v, row_bounds)
        cols = np.digitize(u, col_bounds)
        
        # 构建3×3网格
        grid = [[None]*3 for _ in range(3)]
        for i, (r, c) in enumerate(zip(rows, cols)):
            r, c = int(r), int(c)
            if 0 <= r < 3 and 0 <= c < 3:
                if grid[r][c] is None:  # 避免重复
                    grid[r][c] = dets[i]
        
        # 检查是否9个格子都有
        if any(grid[r][c] is None for r in range(3) for c in range(3)):
            return None
        
        # 展平成TL→BR顺序
        return [grid[r][c] for r in range(3) for c in range(3)]
    
    def _rotate_labels_3x3(self, labels, k):
        """旋转3×3标签（k×90°，k=0/1/2/3）"""
        g = np.array(labels).reshape(3, 3)
        if k == 0:
            return labels
        elif k == 1:
            g = np.rot90(g, k=-1)  # 顺时针90°
        elif k == 2:
            g = np.rot90(g, 2)     # 180°
        else:
            g = np.rot90(g, 1)      # 逆时针90°
        return g.reshape(-1).tolist()
    
    def recognize_current_state(self, frame: np.ndarray, bbox: tuple) -> bool:
        """识别当前魔方状态（完整PnP链路：ROI→网格→位姿→颜色）"""
        x1, y1, x2, y2 = bbox
        
        # 提取ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 50 or roi.shape[1] < 50:
            return False
        
        # ===== 关键修复：先用PoseEstimator提取3×3网格 =====
        # 这是PnP的必要输入！之前一直缺失导致rvec=None
        grid_quads = self.pose_estimator.detect_face_grid(roi)
        
        if grid_quads is None or len(grid_quads) != 9:
            # 调试信息：为什么没找到9个格子
            if grid_quads is None:
                print(f"⚠️ 未检测到网格 (grid_quads=None)")
            else:
                print(f"⚠️ 检测到{len(grid_quads)}个格子，需要9个")
            return False
        
        # ===== 步骤1：PnP位姿估计 =====
        pose_result = self.pose_estimator.estimate_pose_from_grid(
            roi, grid_quads, roi_offset=(x1, y1)
        )
        
        if pose_result is None:
            print("⚠️ PnP失败")
            return False
        
        rvec_raw, tvec_raw = pose_result
        
        # ===== 步骤2：位姿平滑（关键！避免HUD抖动） =====
        if self.last_rvec is not None and self.last_tvec is not None:
            # 指数移动平均
            alpha = 0.3  # 平滑系数
            rvec_raw = (1 - alpha) * self.last_rvec + alpha * rvec_raw
            tvec_raw = (1 - alpha) * self.last_tvec + alpha * tvec_raw
        
        # 更新当前位姿
        self.current_rvec = rvec_raw
        self.current_tvec = tvec_raw
        self.last_rvec = rvec_raw
        self.last_tvec = tvec_raw
        
        # 更新平滑位姿（用于HUD渲染）
        if self.rvec_smooth is None:
            self.rvec_smooth = rvec_raw.copy()
        else:
            self.rvec_smooth = (1.0 - self.rvec_smooth_alpha) * self.rvec_smooth + self.rvec_smooth_alpha * rvec_raw
        
        print(f"✓ PnP成功: rvec norm={np.linalg.norm(rvec_raw):.3f}, tvec[2]={tvec_raw[2,0]:.3f}")
        
        # ===== 步骤3：颜色识别（使用YOLO检测结果） =====
        cluster = self._select_face_cluster(self.current_facelets)
        if not cluster or len(cluster) < 7:
            # 位姿有了，但颜色信息不足，至少位姿可以更新HUD
            return True  # 改为True，因为位姿已经更新了
        
        # 按3×3空间位置排序
        nine = self._order_facelets_3x3(cluster)
        if nine is None:
            return True  # 位姿有效，颜色排序失败，仍返回True
        
        # ===== 步骤4：提取YOLO颜色名，转换成颜色字母 =====
        COLOR_TO_LETTER = {
            'White': 'U',   'Yellow': 'D',  'Red': 'R',
            'Orange': 'L',  'Green': 'F',   'Blue': 'B'
        }
        
        raw_letters = []
        raw_confs = []
        for det in nine:
            cls_id = det.get('cls', -1)
            if 0 <= cls_id < len(self.detector.model.names):
                color_name = self.detector.model.names[cls_id]
                if color_name in COLOR_TO_LETTER:
                    raw_letters.append(COLOR_TO_LETTER[color_name])
                    raw_confs.append(det['conf'])
                else:
                    raw_letters.append('?')
                    raw_confs.append(det['conf'])
            else:
                raw_letters.append('?')
                raw_confs.append(0.0)
        
        # ===== 步骤5：找几何中心最近的格子（中心块） =====
        centers = np.array([[(d['x1']+d['x2'])/2., (d['y1']+d['y2'])/2.] for d in nine], np.float32)
        geo_center = centers.mean(0)
        dists = ((centers - geo_center)**2).sum(1)
        center_idx = np.argmin(dists)
        center_letter = raw_letters[center_idx]
        
        # 如果中心块是'?'，用众数填充
        if center_letter == '?':
            from collections import Counter
            counts = Counter([l for l in raw_letters if l != '?'])
            if counts:
                center_letter = counts.most_common(1)[0][0]
        
        # 用中心块颜色填充所有'?'
        raw_letters = [center_letter if l == '?' else l for l in raw_letters]
        
        # ===== 步骤6：用rvec判断可见面（F/B/U/D/L/R） =====
        # 由于PnP已经在前面完成，这里用rvec推断物理面
        def get_visible_face_from_rvec(rvec):
            """根据rvec判断哪个面朝向相机"""
            R, _ = cv2.Rodrigues(rvec)
            # 定义6个面的法向量（物体坐标系）
            face_normals = {
                'F': np.array([0, 0, 1.0]),
                'B': np.array([0, 0, -1.0]),
                'U': np.array([0, 1.0, 0]),
                'D': np.array([0, -1.0, 0]),
                'R': np.array([1.0, 0, 0]),
                'L': np.array([-1.0, 0, 0]),
            }
            # 找到z分量最大的（朝向相机）
            best_face = None
            best_z = -9999
            for face_label, normal in face_normals.items():
                n_cam = R @ normal
                if n_cam[2] > best_z:
                    best_z = n_cam[2]
                    best_face = face_label
            return best_face
        
        visible_face = get_visible_face_from_rvec(self.current_rvec)
        if visible_face is None:
            visible_face = 'F'  # 默认前面
        
        # ===== 步骤7：k×90°朝向校正（简化版） =====
        # TODO: 根据实际投影方向估计k
        k = 0  # 暂时简化，后续可根据投影校正
        labels = self._rotate_labels_3x3(raw_letters, k)
        
        # ===== 步骤8：保存显示信息 =====
        self._last_color_info = {
            'labels': labels,
            'percentages': [(lbl, conf*100) for lbl, conf in zip(labels, raw_confs)],
            'quads': None,
            'H': None,
            'roi_offset': (x1, y1),
            'cluster': nine
        }
        
        # ===== 步骤9：更新StateManager（按物理面） =====
        face_idx = self.FACE_TO_IDX[visible_face]
        success = self.state_manager.update_face(face_idx, labels, confidence=0.9)
        
        if success:
            # 显示提示（只在完整度变化时播报，避免重复）
            completeness = self.state_manager.get_completeness()
            
            # 只在完整度跨越阈值时播报一次
            if not hasattr(self, '_last_completeness'):
                self._last_completeness = 0.0
            
            if completeness >= 1.0/6 and completeness < 2.0/6 and self._last_completeness < 1.0/6:
                speak("已识别第一面，请转动魔方扫描其他面")
            elif completeness >= 0.5 and completeness < 0.6 and self._last_completeness < 0.5:
                speak("已识别三个面，继续")
            elif completeness >= 0.9 and self._last_completeness < 0.9:
                speak("识别完成，正在计算解法")
            
            self._last_completeness = completeness
        
        return success
    
    def check_move_completion(self) -> bool:
        """检查是否完成了当前步骤"""
        # 获取当前状态
        current_state = self.state_manager.get_stable_state()
        
        if current_state is None:
            return False
        
        # 如果状态发生变化，可能完成了操作
        if current_state != self.last_state_string:
            self.stable_frames = 0
            self.last_state_string = current_state
            return False
        
        self.stable_frames += 1
        
        # 如果状态稳定了一段时间，认为完成了操作
        if self.stable_frames > 10:  # 约0.3秒（30fps）
            return True
        
        return False
    
    def run(self):
        """主循环"""
        print("=" * 60)
        print("全自动魔方识别与引导系统 (YOLO11-seg)")
        print("=" * 60)
        print("✓ 模型: YOLO11分割模型 (mAP50: 97.9%)")
        print("✓ 推理速度: ~0.9ms/张")
        print("提示：将魔方放在摄像头前，系统会自动识别并引导")
        print("按 ESC 退出")
        print()
        
        speak("系统启动，请将魔方放在摄像头前")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检测和跟踪
            bbox = self.detect_and_track(frame)
            
            # 创建显示画面
            display = frame.copy()
            
            if bbox:
                x1, y1, x2, y2 = bbox
                
                # 绘制整体检测框
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制每个色块（支持分割轮廓）
                for det in self.current_facelets:
                    bx1, by1, bx2, by2 = det['x1'], det['y1'], det['x2'], det['y2']
                    conf = det['conf']
                    cls_id = det.get('cls', -1)
                    points = det.get('points')
                    
                    # 如果有轮廓点，绘制精确轮廓；否则绘制矩形
                    if points and len(points) > 2:
                        # 绘制轮廓（蓝色）
                        pts = np.array(points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display, [pts], True, (255, 0, 0), 2)
                        # 填充半透明
                        overlay = display.copy()
                        cv2.fillPoly(overlay, [pts], (255, 0, 0))
                        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
                    else:
                        # 绘制矩形框（蓝色）
                        cv2.rectangle(display, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 0, 0), 2)
                    
                    # 显示置信度
                    txt = f"{conf:.2f}"
                    if cls_id >= 0:
                        txt += f"/{cls_id}"
                    cv2.putText(display, txt, (int(bx1), int(by1)-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # 显示检测信息（检测是否有分割轮廓）
                has_segmentation = any(det.get('points') is not None for det in self.current_facelets)
                seg_info = " (Segmentation)" if has_segmentation else ""
                cv2.putText(display, f"Detected {len(self.current_facelets)} facelets{seg_info}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 识别状态（每5帧更新一次，降低CPU负担）
                recognized = False
                if self.frame_id % 5 == 0:
                    recognized = self.recognize_current_state(frame, bbox)
                else:
                    # 使用上一帧的识别结果
                    recognized = hasattr(self, '_last_color_info') and self._last_color_info is not None
                
                # 直接在每个检测框上显示YOLO的原始颜色（简化逻辑）
                for det in self.current_facelets:
                    cls_id = det.get('cls', -1)
                    if cls_id >= 0 and cls_id < len(self.detector.model.names):
                        color_name = self.detector.model.names[cls_id]
                        conf = det['conf']
                        
                        # 获取检测框中心
                        bx1, by1, bx2, by2 = det['x1'], det['y1'], det['x2'], det['y2']
                        center_x = int((bx1 + bx2) / 2)
                        center_y = int((by1 + by2) / 2)
                        
                        # 绘制标签（简单文字，不要背景框）
                        text = f"{color_name}"
                        cv2.putText(display, text, (center_x - 20, center_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(display, text, (center_x - 20, center_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
                
                # 绘制坐标轴用于调试
                if self.current_rvec is not None and self.current_tvec is not None:
                    # 在ROI上绘制坐标轴
                    roi_display = display[y1:y2, x1:x2].copy()
                    cv2.drawFrameAxes(
                        roi_display,
                        self.camera_matrix,
                        self.dist_coeffs,
                        self.current_rvec,
                        self.current_tvec,
                        0.05,  # 轴长度
                        thickness=2
                    )
                    display[y1:y2, x1:x2] = roi_display
                
                if recognized and self.current_rvec is not None:
                    # 获取下一步操作
                    next_move = self.solver.get_next_move()
                    
                    # 如果有解，显示3D叠加和箭头
                    if next_move:
                        display = self.overlay.render_overlay(
                            display, self.current_rvec, self.current_tvec, next_move
                        )
                        
                        # 显示文字提示
                        move_cn = move_to_cn(next_move)
                        tip = f"下一步: {next_move} - {move_cn}"
                        display = put_text_cn(display, tip, (20, 40), 
                                             font_size=28, color=(0, 255, 255))
                    else:
                        # 尝试求解
                        stable_state = self.state_manager.get_stable_state()
                        if stable_state and stable_state != self.last_state_string:
                            print(f"检测到新状态，尝试求解...")
                            moves = self.solver.solve(stable_state)
                            if moves:
                                self.solution_moves = moves
                                self.move_index = 0
                                self.last_state_string = stable_state
                                self._last_move_spoken = None  # 重置
                                print(f"求解成功！共 {len(moves)} 步")
                                speak(f"求解成功，共{len(moves)}步")
                                
                                # 播报第一步（防抖机制会自动处理间隔）
                                if moves:
                                    speak(f"第一步，{move_to_cn(moves[0])}")
                                    self._last_move_spoken = 0
                            else:
                                display = put_text_cn(display, "状态不可解，请重新摆放", 
                                                     (20, 40), color=(0, 0, 255))
                    
                    # 检查是否完成当前步骤
                    if self.check_move_completion() and self.solver.get_next_move():
                        self.solver.advance_move()
                        self.move_index += 1
                        self.stable_frames = 0
                        
                        # 播报下一步（只在步骤变化时播报）
                        next_move = self.solver.get_next_move()
                        if next_move:
                            if self._last_move_spoken != self.move_index:
                                speak(f"第{self.move_index+1}步，{move_to_cn(next_move)}")
                                self._last_move_spoken = self.move_index
                        else:
                            # 只在第一次完成时播报
                            if self._last_move_spoken != -1:
                                speak("完成！魔方已还原")
                                self._last_move_spoken = -1
                            print("✓ 魔方已还原！")
                else:
                    display = put_text_cn(display, "正在识别魔方状态...", 
                                         (20, 40), color=(255, 255, 0))
            else:
                display = put_text_cn(display, "未检测到魔方，请将魔方放在摄像头前", 
                                     (20, 40), color=(0, 0, 255))
            
            # 显示状态信息
            completeness = self.state_manager.get_completeness()
            info_text = f"状态完整度: {completeness*100:.1f}%"
            if self.solution_moves:
                info_text += f" | 剩余步骤: {len(self.solver.get_remaining_moves())}"
            display = put_text_cn(display, info_text, (20, 80), 
                                 font_size=20, color=(255, 255, 255))
            
            # 计算FPS（简化显示，不用慢的中文绘字）
            self.fps_counter += 1
            if time.time() - self.fps_time > 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = time.time()
            
            # 显示FPS（用简单的cv2.putText）
            if hasattr(self, 'current_fps'):
                fps_text = f"FPS: {self.current_fps}"
                cv2.putText(display, fps_text, (display.shape[1]-120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 渲染3D迷你魔方HUD（使用平滑后的位姿）
            r_for_hud = self.rvec_smooth if self.rvec_smooth is not None else self.current_rvec
            display = self.hud.render(display, r_for_hud, self.state_manager)
            
            cv2.imshow("AI Rubik Auto Guide", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n程序结束，感谢使用！")

if __name__ == "__main__":
    try:
        app = RubikAutoGuide(camera_id=0)
        app.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
