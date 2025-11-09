"""
Roboflow 预训练模型检测器
使用现成的魔方检测模型，无需自己训练
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import requests
import base64

class RoboflowRubikDetector:
    """
    使用 Roboflow 预训练魔方检测模型
    支持两种模式：
    1. 整块魔方检测（用于 ROI 定位）
    2. 面片检测（用于 3x3 色块识别）
    """
    
    def __init__(self, api_key: str = None, mode: str = "cube"):
        """
        初始化
        Args:
            api_key: Roboflow API key (从 https://app.roboflow.com/ 获取)
            mode: "cube" (整块检测) 或 "facelet" (面片检测)
        """
        self.api_key = api_key
        self.mode = mode
        
        # Roboflow 模型端点
        if mode == "cube":
            # 整块魔方检测
            self.model_endpoint = "rubik-s-cube-detector/1"
            self.project = "maskdetection-06po4"
        else:
            # 面片检测
            self.model_endpoint = "rubik-s-cube-face-detection/1"
            self.project = "rm-22-yolov5"
        
        self.base_url = f"https://detect.roboflow.com/{self.model_endpoint}"
        
        print(f"✓ Roboflow 检测器初始化 (模式: {mode})")
        if not api_key:
            print("⚠ 未提供 API key，请访问 https://app.roboflow.com/ 获取")
    
    def detect_online(self, frame: np.ndarray, confidence: float = 40) -> List[dict]:
        """
        在线推理（需要 API key）
        """
        if not self.api_key:
            print("❌ 需要 API key 才能使用在线推理")
            return []
        
        # 编码图像
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # 调用 API
        url = f"{self.base_url}?api_key={self.api_key}&confidence={confidence}"
        
        try:
            response = requests.post(
                url,
                json=img_base64,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('predictions', [])
            else:
                print(f"API 错误: {response.status_code}")
                return []
        except Exception as e:
            print(f"请求失败: {e}")
            return []
    
    def detect_local(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        本地推理（需要下载模型）
        这里使用备用的轮廓检测
        """
        # 如果没有本地模型，使用轮廓检测作为备选
        from detector import RubikDetector
        detector = RubikDetector(use_yolo=False)
        return detector.detect(frame)
    
    def detect(self, frame: np.ndarray, use_online: bool = False) -> List[Tuple[int, int, int, int, float]]:
        """
        检测魔方
        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        if use_online and self.api_key:
            predictions = self.detect_online(frame)
            boxes = []
            for pred in predictions:
                x = pred['x']
                y = pred['y']
                w = pred['width']
                h = pred['height']
                conf = pred['confidence']
                
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                boxes.append((x1, y1, x2, y2, conf))
            
            return boxes
        else:
            return self.detect_local(frame)
    
    def get_best_box(self, frame: np.ndarray, use_online: bool = False) -> Optional[Tuple[int, int, int, int]]:
        """获取最佳检测框"""
        boxes = self.detect(frame, use_online)
        if boxes:
            x1, y1, x2, y2, _ = boxes[0]
            return (x1, y1, x2, y2)
        return None


class LocalYOLODetector:
    """
    本地 YOLO 检测器（使用下载的预训练模型）
    """
    
    def __init__(self, model_path: str = "models/rubik_roboflow.pt", device: str = '0', half: bool = True,
                 allowed_names: Optional[set] = None, default_conf: float = 0.5):
        """
        初始化
        Args:
            model_path: 从 Roboflow 下载的模型路径
            device: 设备 ('0' for GPU, 'cpu' for CPU)
            half: 是否使用 FP16 半精度（GPU加速）
        """
        self.model_path = model_path
        self.model = None
        self.device = device
        self.half = half
        self.allowed_names = set(allowed_names) if allowed_names else None
        self.default_conf = float(default_conf)
        
        try:
            from ultralytics import YOLO
            import os
            import torch
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                
                # 尝试使用 GPU
                if device != 'cpu' and torch.cuda.is_available():
                    self.model.to(device)
                    if half:
                        self.model.model.half()
                    print(f"✓ 加载模型: {model_path} (GPU + FP16)")
                elif device == 'mps' or (device == '0' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    # Mac MPS
                    try:
                        self.model.to('mps')
                        print(f"✓ 加载模型: {model_path} (MPS)")
                    except:
                        self.device = 'cpu'
                        self.half = False
                        print(f"✓ 加载模型: {model_path} (CPU)")
                else:
                    self.device = 'cpu'
                    self.half = False
                    print(f"✓ 加载模型: {model_path} (CPU)")
            else:
                print(f"⚠ 模型文件不存在: {model_path}")
                print("请从 Roboflow 下载模型并放置到 models/ 目录")
                print("下载地址: https://universe.roboflow.com/maskdetection-06po4/rubik-s-cube-detector")
        except ImportError:
            print("⚠ ultralytics 未安装，请运行: pip install ultralytics")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = None):
        """检测魔方（GPU + FP16 加速）
        Returns:
            List of detection dicts with keys: x1, y1, x2, y2, conf, cls, points (if segmentation)
        """
        if self.model is None:
            return []
        if conf_threshold is None:
            conf_threshold = self.default_conf
        
        # 使用 GPU + FP16
        results = self.model.predict(
            frame, 
            conf=conf_threshold, 
            verbose=False,
            device=self.device,
            half=self.half
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                try:
                    cls_id = int(box.cls[0].item())
                except Exception:
                    cls_id = -1
                
                # Class filtering by allowed_names (if provided)
                if self.allowed_names is not None and hasattr(self.model, 'names'):
                    try:
                        class_name = self.model.names[cls_id]
                        if class_name not in self.allowed_names:
                            continue
                    except Exception:
                        # If cannot resolve class name, keep the detection
                        pass

                det = {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'conf': float(conf),
                    'cls': cls_id,
                    'points': None
                }
                
                # 如果是分割模型，提取轮廓点
                if masks is not None:
                    try:
                        # 获取mask
                        mask = masks.data[idx].cpu().numpy()
                        # 将mask resize到原图尺寸
                        h, w = frame.shape[:2]
                        mask_h, mask_w = mask.shape
                        if mask_h != h or mask_w != w:
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # 转换为二值图
                        mask_binary = (mask > 0.5).astype(np.uint8) * 255
                        
                        # 提取轮廓
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # 取最大的轮廓
                            contour = max(contours, key=cv2.contourArea)
                            # 简化轮廓（减少点数）
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            # 转换为点列表
                            points = approx.reshape(-1, 2).tolist()
                            det['points'] = points
                    except Exception:
                        pass
                
                detections.append(det)
        
        # 按置信度排序
        detections.sort(key=lambda x: x['conf'], reverse=True)
        return detections
    
    def get_best_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """获取最佳检测框"""
        detections = self.detect(frame)
        if detections:
            det = detections[0]
            return (det['x1'], det['y1'], det['x2'], det['y2'])
        return None
