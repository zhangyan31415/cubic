# 🎲 AI Rubik's Cube Solver

基于 YOLO11 分割的魔方自动识别与求解系统，已模块化重构为“分割掩膜 → 聚类拼 3×3 网格 → PnP 位姿 → 状态构建 → Kociemba 求解 → AR 指引”的稳定管线。

## ✨ 特性

- 🤖 **YOLO11‑seg** 实时检测魔方色块（mAP50: 97.9%）
- 🧩 **分割拼网格** 直接用分割轮廓聚类+PCA 构造 3×3 网格（更稳）
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
- `R` - 重置状态
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

## 📁 项目结构（已模块化重构）

```
cubic/
├── app.py                           # 主程序（新：模块化调度）
├── models/
│   └── rubik_cube_yolo11_seg.pt     # YOLO11 分割模型
├── src/
│   ├── camera.py                    # 相机封装
│   ├── vision.py                    # 视觉管线：分割→聚类→网格→PnP
│   ├── cube_state.py                # 状态封装（基于 StateManager）
│   ├── solver_wrap.py               # 求解封装 + 中文步骤
│   ├── core/
│   │   ├── pose.py                  # 位姿估计（OpenCV solvePnP）
│   │   ├── state.py                 # 状态管理
│   │   ├── solver.py                # Kociemba 求解器
│   │   ├── mini_cube_hud.py         # 3D 迷你魔方 HUD
│   │   └── overlay.py               # 叠加箭头/辅助线
│   └── detectors/
│       └── roboflow_detector.py     # YOLO 检测器（分割轮廓点）
└── test_color_detect/
    ├── test_full_pipeline.py        # 完整流程测试
    └── test_roboflow_api.py         # API 测试
```

## 🔧 技术栈

- **检测**: YOLO11-seg (Ultralytics)
- **求解**: Kociemba Algorithm
- **位姿估计**: OpenCV solvePnP（平面目标，含位姿平滑）
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

如有问题，请提交 Issue。

## 🆕 重构说明（要点）

- 由“ROI 内边缘/九宫格检测”改为“分割轮廓→聚类→PCA→等距网格线”直接拼 3×3 网格，显著降低光照/反光/无黑边环境下的失败率。
- PnP 基于全图坐标的 9 个小格中心点，配合指数平滑，HUD 更稳。
- 主程序解耦为 `camera / vision / cube_state / solver_wrap` 四个模块，便于替换检测器或扩展到其它阶数。
- 键位调整：移除空格键，保留 `R`（重置）、`ESC`（退出）。
