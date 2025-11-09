"""
主程序：AI 魔方识别与引导（模块化重构）
YOLO 分割 → 聚类拼网格 → PnP → 状态构建 → Kociemba → AR 指引
"""
import time
import threading
import subprocess
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.camera import Camera
from src.vision import CubeVision
from src.cube_state import CubeState
from src.solver_wrap import Solver, move_to_cn
from src.core.overlay import OverlayRenderer
from src.core.mini_cube_hud import MiniCubeHUD
import yaml
from src.vision_rotation import rotation_k_from_rvec, rotate_grid_labels

try:
    from ui.face_editor import FaceEditor
except Exception:
    FaceEditor = None


def speak(text: str):
    now = time.time()
    if not hasattr(speak, '_last_text'):
        speak._last_text = None
        speak._last_time = 0
    if speak._last_text == text and (now - speak._last_time) < 2.0:
        return
    speak._last_text = text
    speak._last_time = now

    def _s():
        try:
            subprocess.run(['say', '-v', 'Ting-Ting', text], check=False, timeout=3)
        except Exception:
            pass
    threading.Thread(target=_s, daemon=True).start()


def put_text_cn(img_bgr, text, xy=(20, 40), font_size=24, color=(0, 255, 0)):
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


class App:
    def __init__(self, cam_id: int = 0):
        # Load config
        with open("config.yaml", "r") as f:
            self.cfg = yaml.safe_load(f) or {}

        cam = Camera(cam_id).open()
        # 可选设置摄像头分辨率（来自配置）
        cam_cfg = (self.cfg or {}).get('camera', {}) or {}
        res = cam_cfg.get('resolution')
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                cam.set_resolution(int(res[0]), int(res[1]))
            except Exception:
                pass
        frame = cam.read()
        h, w = frame.shape[:2]
        # Camera params from config if present
        cam_cfg = (self.cfg or {}).get('camera', {}) or {}
        if cam_cfg.get('matrix'):
            K = np.array(cam_cfg['matrix'], dtype=np.float32)
            cx, cy = float(K[0][2]), float(K[1][2])
            fx, fy = float(K[0][0]), float(K[1][1])
        else:
            fx, fy = w * 0.9, h * 0.9
            cx, cy = w / 2.0, h / 2.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist = np.array(cam_cfg.get('dist', [0,0,0,0,0]), dtype=np.float32)
        print(f"📷 相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}; refineLM={cam_cfg.get('use_refine_lm', True)}")

        # device auto selection
        import torch
        if torch.backends.mps.is_available():
            device, half = 'mps', False
        elif torch.cuda.is_available():
            device, half = '0', True
        else:
            device, half = 'cpu', False

        self.camera = cam
        model_path = (self.cfg.get('model_path') or "models/rubik_cube_yolo11_seg.pt")
        det_cfg = self.cfg.get('detector', {}) or {}
        allowed = det_cfg.get('allowed_names', None)
        default_conf = float(det_cfg.get('conf', 0.45))
        ema_alpha = float((self.cfg.get('smoothing', {}) or {}).get('ema_alpha', 0.25))
        refine_lm = bool(cam_cfg.get('use_refine_lm', True))
        yolo_cfg = self.cfg.get('yolo', {}) or {}
        self.vision = CubeVision(K, dist, model_path=model_path, device=device, half=half,
                                 allowed_names=allowed, default_conf=default_conf,
                                 ema_alpha=ema_alpha, refine_lm=refine_lm,
                                 imgsz=int(yolo_cfg.get('imgsz', 640)),
                                 iou=float(yolo_cfg.get('iou', 0.5)),
                                 max_det=int(yolo_cfg.get('max_det', 100)))
        self.cube = CubeState()
        self.solver = Solver()
        self.overlay = OverlayRenderer(K, dist)
        self.hud = MiniCubeHUD(size=220)
        self.use_face_editor = bool((self.cfg.get('ui', {}) or {}).get('face_editor', False)) and FaceEditor is not None
        self._editor = FaceEditor() if self.use_face_editor else None

        # runtime
        self.frame_id = 0
        self.current_rvec = None
        self.current_tvec = None
        self.last_state_string = None
        self._last_completeness = 0.0
        self._last_move_spoken: Optional[int] = None
        self.stable_frames = 0
        self.current_fps = 0
        self._fps_counter = 0
        self._fps_time = time.time()

    def _draw_detections(self, display, facelets):
        for det in facelets:
            bx1, by1, bx2, by2 = det['x1'], det['y1'], det['x2'], det['y2']
            conf = det['conf']
            points = det.get('points')
            if points and len(points) > 2:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (255, 0, 0), 2)
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            else:
                cv2.rectangle(display, (int(bx1), int(by1)), (int(bx2), int(by2)), (255, 0, 0), 2)
            cv2.putText(display, f"{conf:.2f}", (int(bx1), int(by1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return display

    def _order_labels(self, cluster: list):
        """返回 (labels9, center_letter)。labels9 为 3×3 展平的面字母序列。"""
        nine = self.vision.order_facelets_3x3(cluster)
        if nine is None:
            return None, None
        names = getattr(self.vision.detector.model, 'names', {})
        letters = []
        for d in nine:
            cls_id = d.get('cls', -1)
            # names 可能是 list 或 dict
            if isinstance(names, dict):
                color_name = names.get(cls_id, '?')
            else:
                color_name = names[cls_id] if 0 <= cls_id < len(names) else '?'
            letters.append(self.vision.COLOR_TO_LETTER.get(color_name, '?'))
        # 估计几何中心作为中心块索引
        centers = np.array([[(d['x1']+d['x2'])/2., (d['y1']+d['y2'])/2.] for d in nine], np.float32)
        geo = centers.mean(0)
        idx = int(np.argmin(((centers-geo)**2).sum(1)))
        center_letter = letters[idx]
        if center_letter == '?':
            from collections import Counter
            c = Counter([x for x in letters if x != '?'])
            center_letter = c.most_common(1)[0][0] if c else '?'
        letters = [center_letter if x == '?' else x for x in letters]
        return letters, center_letter

    def _check_move_completion(self) -> bool:
        current_state = self.cube.get_stable_state()
        if current_state is None:
            return False
        if current_state != self.last_state_string:
            self.stable_frames = 0
            self.last_state_string = current_state
            return False
        self.stable_frames += 1
        return self.stable_frames > 10

    def run(self):
        print("="*60)
        print("AI Rubik's Cube (YOLO-seg)")
        print("按 ESC 退出，按 R 重置")
        print("="*60)
        speak("系统启动，请将魔方放在摄像头前")

        while True:
            frame = self.camera.read()
            self.frame_id += 1
            display = frame.copy()

            facelets = self.vision.detect_facelets(frame)
            bbox = None
            if facelets:
                cluster = self.vision.select_face_cluster(facelets)
                src = cluster if cluster else facelets
                x1 = min(d['x1'] for d in src); y1 = min(d['y1'] for d in src)
                x2 = max(d['x2'] for d in src); y2 = max(d['y2'] for d in src)
                m=30
                x1=max(0,x1-m); y1=max(0,y1-m)
                x2=min(frame.shape[1],x2+m); y2=min(frame.shape[0],y2+m)
                bbox=(x1,y1,x2,y2)
                display = self._draw_detections(display, facelets)
                cv2.rectangle(display, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(display, f"Detected {len(facelets)} facelets", (x1, max(0,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

                # Build grid and pose every 5 frames to reduce load
                if self.frame_id % 5 == 0:
                    quads = self.vision.build_grid_from_segmentation(facelets)
                    if quads is None or len(quads) != 9:
                        # Fallback via contour/warp
                        quads = self.vision.fallback_quads_via_contour(frame, facelets)
                    if quads is None or len(quads) != 9:
                        print("⚠️ 未检测到网格 (分割与兜底均失败)")
                    else:
                        pose = self.vision.estimate_pose_from_quads(frame, quads)
                        if pose is not None:
                            self.current_rvec, self.current_tvec = pose
                            print(f"✓ PnP成功: |r|={np.linalg.norm(self.current_rvec):.3f}, tz={self.current_tvec[2,0]:.3f}")

                if self.current_rvec is not None:
                    vis_face = self.vision.visible_face_from_rvec(self.current_rvec) or 'F'
                    if cluster and len(cluster) >= 7:
                        labels, center_letter = self._order_labels(cluster)
                        if labels and center_letter and '?' not in labels and center_letter != '?':
                            # 面内旋转仅由 rvec 量化，用于统一九宫格方向
                            k = rotation_k_from_rvec(self.current_rvec)
                            labels = rotate_grid_labels(labels, k)
                            # 可选人工校对
                            if self.use_face_editor:
                                ok_ed, labels_ed = self._editor.edit_until_confirm(labels)
                                if ok_ed:
                                    labels = labels_ed
                            # 用中心颜色锚定的面字母写入状态（不再用 rvec 判定）
                            face_letter = center_letter
                            ok = self.cube.update_face_by_label(face_letter, labels, confidence=0.9)
                            if ok:
                                comp = self.cube.completeness()
                                if comp >= 1.0/6 and self._last_completeness < 1.0/6:
                                    speak("已识别第一面，请转动魔方扫描其他面")
                                elif comp >= 0.5 and self._last_completeness < 0.5:
                                    speak("已识别三个面，继续")
                                elif comp >= 0.9 and self._last_completeness < 0.9:
                                    speak("识别完成，正在计算解法")
                                self._last_completeness = comp

                # guidance / solving
                if self.current_rvec is not None:
                    next_move = self.solver.next_move()
                    if next_move:
                        display = self.overlay.render_overlay(display, self.current_rvec, self.current_tvec, next_move)
                        tip = f"下一步: {next_move} - {move_to_cn(next_move)}"
                        display = put_text_cn(display, tip, (20, 40), font_size=28, color=(0,255,255))
                        if self._check_move_completion():
                            self.solver.advance()
                            step_idx = (self._last_move_spoken or -1) + 1
                            nm = self.solver.next_move()
                            if nm:
                                if self._last_move_spoken != step_idx:
                                    speak(f"第{step_idx+1}步，{move_to_cn(nm)}")
                                    self._last_move_spoken = step_idx
                            else:
                                if self._last_move_spoken != -1:
                                    speak("完成！魔方已还原")
                                    self._last_move_spoken = -1
                    else:
                        stable_state = self.cube.get_stable_state()
                        if stable_state and stable_state != self.last_state_string:
                            print("检测到新状态，尝试求解...")
                            moves = self.solver.solve(stable_state)
                            if moves:
                                self._last_move_spoken = None
                                self.last_state_string = stable_state
                                speak(f"求解成功，共{len(moves)}步")
                                if moves:
                                    speak(f"第一步，{move_to_cn(moves[0])}")
                                    self._last_move_spoken = 0
                            else:
                                display = put_text_cn(display, "状态不可解，请重新摆放", (20, 40), color=(0,0,255))
            else:
                display = put_text_cn(display, "未检测到魔方，请将魔方放在摄像头前", (20, 40), color=(0,0,255))

            # HUD render
            r_for_hud = self.vision.rvec_smooth if self.vision.rvec_smooth is not None else self.current_rvec
            display = self.hud.render(display, r_for_hud, self.cube.sm)

            # HUD info
            comp = self.cube.completeness()
            info = f"状态完整度: {comp*100:.1f}%"
            rem = self.solver.remaining()
            if rem:
                info += f" | 剩余步骤: {len(rem)}"
            display = put_text_cn(display, info, (20, 80), font_size=20, color=(255,255,255))

            # FPS
            self._fps_counter += 1
            if time.time() - self._fps_time > 1.0:
                self.current_fps = self._fps_counter
                self._fps_counter = 0
                self._fps_time = time.time()
            cv2.putText(display, f"FPS: {self.current_fps}", (display.shape[1]-120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("AI Rubik Auto Guide", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord('r'), ord('R')):
                self.cube.reset(); self.solver.reset(); self._last_completeness = 0.0
                speak("已重置，请重新扫描")

        self.camera.release()
        cv2.destroyAllWindows()
        print("\n程序结束，感谢使用！")


if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
