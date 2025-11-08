"""
配置文件：相机标定参数和系统配置
"""
import numpy as np

# 相机内参矩阵 K (3x3)
# 如果已标定，替换为实际值
# 格式：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
CAMERA_MATRIX = None  # 将在运行时根据画面尺寸自动生成

# 畸变系数 (5个)
DIST_COEFFS = np.zeros(5, dtype=np.float32)

# YOLO模型路径
# 选项：
# - "yolov8n.pt" (nano, 最快)
# - "yolov8s.pt" (small, 平衡)
# - "yolov8m.pt" (medium, 更准确)
# - 自定义训练的魔方检测模型路径
YOLO_MODEL_PATH = "yolov8n.pt"

# 检测置信度阈值
DETECTION_CONF_THRESHOLD = 0.3

# 状态稳定阈值（帧数）
STABLE_FRAMES_THRESHOLD = 10

# 求解器配置
SOLVER_TIMEOUT = 5.0  # 秒

# 显示配置
SHOW_FPS = True
SHOW_DEBUG_INFO = True

# 语音配置
ENABLE_VOICE = True
VOICE_LANG = "zh_CN"

# 性能配置
TARGET_FPS = 30
MAX_FRAME_SKIP = 2

