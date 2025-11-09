# 🎲 AI Rubik's Cube Solver

基于 YOLO11 分割的魔方自动识别与求解系统，当前版本已完整重构为：

> **分割掩膜 → 聚类拼 3×3 网格 → PnP + RefineLM 位姿 → rvec 量化旋转 → 中心锚定状态构建 → Kociemba 预检 + 求解 → HUD/语音/编辑器交互**

整条链路在几何、颜色、可解性和交互上都增加了防错逻辑，适应不同光照与机位。

## ✨ 特性

- 🤖 **YOLO11‑seg + 类别过滤**：仅保留六种颜色类，避免 “Face/Center” 掩膜干扰聚类。
- 🧩 **分割→聚类→PCA→3×3 网格**：不足 9 块时自动启用“轮廓+透视”兜底，确保 PnP 有效输入。
- 🎯 **PnP + solvePnPRefineLM**：全图坐标求位姿并做 LM 精修，输出 rvec/tvec 供 HUD 与旋转对齐使用。
- 🔄 **rvec 量化 & 中心锚定**：按 rvec 滚转角将九宫格顺时针量化旋转，六个中心颜色锁定面字母映射。
- 🧮 **Kociemba 预检 + 求解**：计数 + 预检+可解才入最终求解器，避免错误状态。
- 🧊 **FaceEditor（可选）**：OpenCV 面板支持逐面校对/回退/暂停，识别失败时人工兜底。
- 🗣️ **语音 / HUD / Overlay**：实时提示 + 3D 迷你魔方 HUD + 箭头指引。

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

1. 将魔方对准摄像头，等待系统锁定 9 个色块（HUD 左上角显示当前面）。
2. 按提示转动魔方依次扫描 6 个面；启用 FaceEditor 时会弹出九宫格窗口，可点击修改颜色再确认。
3. 识别完成后 HUD/语音会给出求解步骤，按箭头或文字指示操作。

**默认快捷键：**
- `R`：重置识别/求解状态
- `ESC`：退出程序
- `P`（FaceEditor 内）：暂停/恢复视频流
- `U`（FaceEditor 内）：回退当前面编辑记录

## ⚙️ 配置文件 `config.yaml`

运行前可通过 `config.yaml` 精细调整各环节：

| 字段 | 说明 |
| --- | --- |
| `model_path` | YOLO11 分割模型路径 |
| `detector.conf` | 推理置信度阈值 |
| `detector.allowed_names` | 允许的类别（默认六种颜色） |
| `smoothing.ema_alpha` | rvec/tvec EMA 平滑系数 |
| `camera.matrix/dist` | 相机内参与畸变，留空则按首帧尺寸推算 |
| `camera.use_refine_lm` | 是否在 PnP 后调用 `solvePnPRefineLM` |
| `keys.reset/quit` | 快捷键自定义 |
| `tts.enabled` | 语音提示开关（macOS 默认用 `say`） |
| `ui.face_editor` | 是否启用面校对/暂停/回退窗口 |

示例：

```yaml
model_path: models/rubik_cube_yolo11_seg.pt
detector:
  conf: 0.45
  allowed_names: [Blue, Green, Red, Orange, White, Yellow]
camera:
  matrix: null    # 置入标定矩阵即可替换
  dist: [0, 0, 0, 0, 0]
  use_refine_lm: true
ui:
  face_editor: false
```

## 🧠 识别流程与关键策略

1. **检测**：YOLO11‑seg 输出所有色块 + 分割多边形；立即按 `allowed_names` 过滤，只保留六种颜色。
2. **聚类/网格**：
   - 主路径：按质心 + PCA 聚类出一个 3×3 面，构造均匀网格；
   - 兜底：不足 9 块时对最大轮廓做透视拉正，按固定 3×3 切块。
3. **PnP**：使用全图坐标的格子中心做 `solvePnP`，随后 `solvePnPRefineLM` 微调，输出 rvec/tvec（并做 EMA 平滑）。
4. **rvec 量化**：根据 rvec 的面内 x 轴投影计算滚转角，量化成 k×90°，对九宫格做顺时针旋转以统一朝向。
5. **颜色锚定**：
   - 6 个中心块颜色必须互不相同，并与面字母建立一一映射；
   - 每一贴纸都被映射成其中心面的字母（U/R/F/D/L/B），符合 Kociemba 输入格式。
6. **可解性预检**：构造 54 字符串后先做计数检测，再调用 `kociemba.solve()` 进行奇偶/角棱朝向预检（不会执行真实求解，只做验证）。
7. **求解与提示**：预检通过后才送入 Kociemba 求解器，HUD、语音、Overlay 会逐步提示操作；启用 FaceEditor 时可随时暂停/回退。

> 建议在日志中开启以下打印：`facelets_detected / cluster_size / rotation_k / centers_ok / preflight_msg / reproj_error`，方便排障。

## 🧪 测试与排障

- `python test_color_detect/test_full_pipeline.py`：使用图片验证 YOLO11 分割输出及中文可视化。
- `python test_color_detect/test_roboflow_api.py`：对接 Roboflow 远程 API。
- **调参建议**：
  - 光照极端时降低 `detector.conf` 并配合 FaceEditor；
  - 若 HUD 抖动，校准真实 K/dist 或增加 `smoothing.ema_alpha`。
  - 若状态不可解，查看 `centers_ok`/`check_solvability_full` 的提示文本，通常是中心冲突或奇偶错误。

### 测试完整流程

```bash
python test_color_detect/test_full_pipeline.py
```

### 测试Roboflow API

```bash
python test_color_detect/test_roboflow_api.py
```

## 🎓 训练/数据说明

- 当前仓库提供的是推理与交互代码，旧版 `train_segmentation.py` 已归档至 `_archive/`，如需重训请参考 Roboflow 项目或在 `_archive` 中查阅脚本。
- 数据集：Roboflow Rubik's Cube Dataset（1136 张图片，8 类标签）。训练时建议移除 `Center/Face` 类或单独建 head，以避免推理阶段的类别过滤。
- 增强策略建议：亮度/色温随机、贴纸磨损/缺角合成、背景替换；推理端配合中心 HSV 最近邻重标可显著降低红↔橙、蓝↔绿误判。

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
├── src/vision_rotation.py          # rvec 量化 & 3×3 顺序对齐
├── ui/
│   └── face_editor.py              # 逐面校对/暂停/回退面板
└── test_color_detect/
    ├── test_full_pipeline.py       # 完整流程测试
    └── test_roboflow_api.py        # API 测试
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
