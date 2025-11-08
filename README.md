# 🎲 AI Rubik's Cube Solver

基于YOLO11分割模型的魔方自动识别与求解系统。

## ✨ 特性

- 🤖 **YOLO11-seg** 实时检测魔方色块（mAP50: 97.9%）
- 🎯 **自动识别** 魔方6个面的状态
- 🧩 **Kociemba算法** 快速求解（<1秒）
- 🎨 **3D可视化** 实时显示魔方状态和求解步骤
- 🗣️ **语音提示** 指导用户操作
- 📊 **实时HUD** 显示识别进度和解法

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
python app.py
```

### 3. 操作说明

1. 将魔方对准摄像头
2. 系统自动检测并识别色块
3. 转动魔方，让系统扫描所有6个面
4. 识别完成后，按照3D提示完成求解

**快捷键：**
- `空格` - 开始/暂停识别
- `r` - 重置状态
- `ESC` - 退出

## 🧪 测试工具

### 测试完整流程

```bash
python test_color_detect/test_full_pipeline.py
```

### 测试Roboflow API

```bash
python test_color_detect/test_roboflow_api.py
```

## 🎓 训练自己的模型

### 训练YOLO11分割模型

```bash
python train_segmentation.py
```

**支持多GPU训练**：脚本会自动询问使用的GPU数量（1/2/4/8），并自动调整batch size和image size。

### 训练配置

- **数据集**: Roboflow Rubik's Cube Dataset (1136张图像)
- **模型**: YOLO11m-seg
- **类别**: 8个 (Blue, Center, Face, Green, Orange, Red, White, Yellow)
- **性能**: mAP50=97.9%, 推理速度=0.9ms/帧 (H200 GPU)

## 📁 项目结构

```
cubic/
├── app.py                          # 主程序
├── train_segmentation.py           # 训练脚本
├── models/                         # 模型文件
│   └── rubik_cube_yolo11_seg.pt   # 训练好的YOLO11模型
├── src/                           # 核心代码
│   ├── core/                      # 核心功能模块
│   │   ├── state.py              # 状态管理
│   │   ├── solver.py             # Kociemba求解器
│   │   ├── pose.py               # 位姿估计
│   │   ├── mini_cube_hud.py      # 3D可视化
│   │   └── ...
│   └── detectors/                # 检测器
│       └── roboflow_detector.py  # YOLO检测器
└── test_color_detect/            # 测试工具
    ├── test_full_pipeline.py     # 完整流程测试
    └── test_roboflow_api.py      # API测试
```

## 🔧 技术栈

- **检测**: YOLO11-seg (Ultralytics)
- **求解**: Kociemba Algorithm
- **位姿估计**: OpenCV solvePnP (IPPE_SQUARE)
- **3D渲染**: OpenCV + NumPy
- **深度学习框架**: PyTorch

## 📊 模型性能

| 指标 | 值 |
|------|-----|
| mAP50 | 97.9% |
| mAP50-95 | 73.2% |
| 推理速度 | 0.9ms/帧 (H200) |
| 模型大小 | 49.8MB |
| 参数量 | 25.5M |

## 🤝 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO11模型
- [Roboflow](https://roboflow.com/) - 数据集托管
- [kociemba](https://github.com/muodov/kociemba) - 魔方求解算法

## 📄 许可

MIT License

## 🐛 问题反馈

如有问题，请提交Issue。
