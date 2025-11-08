#!/usr/bin/env python3
"""
完整流程测试：YOLO11分割模型 - 精确轮廓检测
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detectors.roboflow_detector import LocalYOLODetector
from src.core.wb_lab import AdvancedColorRecognizer

# 固定模型路径（YOLO11分割模型）
MODEL_PATH = "../models/rubik_cube_yolo11_seg.pt"

def test_image(image_path):
    """测试图像"""
    print(f"\n{'='*70}")
    print("YOLO11分割模型测试")
    print('='*70)
    print(f"模型: {MODEL_PATH}")
    print(f"图像: {image_path}")
    print('='*70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法加载图像: {image_path}")
        return
    
    print(f"✓ 图像尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 初始化
    detector = LocalYOLODetector(model_path=MODEL_PATH, device='cpu', half=False)
    
    print("\n正在检测...")
    
    # 检测
    detections = detector.detect(img, conf_threshold=0.5)
    
    if not detections:
        print("❌ 未检测到色块")
        return
    
    print(f"✓ 检测成功\n")
    print(f"检测到 {len(detections)} 个对象:\n")
    
    display = img.copy()
    
    # 颜色名称映射
    color_map = {
        'Blue': '蓝色',
        'Center': '中心',
        'Face': '面',
        'Green': '绿色',
        'Orange': '橙色',
        'Red': '红色',
        'White': '白色',
        'Yellow': '黄色'
    }
    
    # 处理每个检测
    for i, det in enumerate(detections):
        # 获取检测信息
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        conf = det['conf']
        cls_id = det['cls']
        points = det.get('points')
        class_name = detector.model.names[cls_id]
        
        color_name = color_map.get(class_name, class_name)
        print(f"[{i}] {color_name} ({class_name}) - 置信度: {conf:.2f} - 位置: ({x1}, {y1}, {x2}, {y2})")
        
        # 如果有轮廓点，绘制轮廓；否则绘制矩形框
        if points and len(points) > 2:
            # 绘制轮廓（绿色）
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            # 填充半透明颜色
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
        else:
            # 绘制边界框（绿色）
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签（左上角上方）
        label = f"{class_name} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # 背景框（绿色背景）
        cv2.rectangle(display, (x1, y1-text_h-8), (x1+text_w+4, y1), (0, 255, 0), -1)
        cv2.putText(display, label, (x1+2, y1-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    print("\n显示结果（按任意键退出）...")
    cv2.imshow('YOLO Detection Result', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_camera():
    """测试摄像头"""
    print(f"\n{'='*70}")
    print("YOLO11分割模型 - 实时摄像头测试")
    print('='*70)
    print("按空格键检测，ESC退出")
    print('='*70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    # 初始化
    detector = LocalYOLODetector(model_path=MODEL_PATH, device='cpu', half=False)
    print("✓ 模型加载完成\n")
    
    # 颜色名称映射
    color_map = {
        'Blue': '蓝色',
        'Center': '中心',
        'Face': '面',
        'Green': '绿色',
        'Orange': '橙色',
        'Red': '红色',
        'White': '白色',
        'Yellow': '黄色'
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        cv2.putText(display, "Press SPACE to detect", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Camera', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            print("\n正在检测...")
            
            # 检测
            detections = detector.detect(frame, conf_threshold=0.5)
            
            if not detections:
                print("未检测到色块")
                continue
            
            print(f"检测到 {len(detections)} 个对象:")
            
            result_display = frame.copy()
            
            for i, det in enumerate(detections):
                # 获取检测信息
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                conf = det['conf']
                cls_id = det['cls']
                points = det.get('points')
                class_name = detector.model.names[cls_id]
                
                color_name = color_map.get(class_name, class_name)
                print(f"  [{i}] {color_name} - {conf:.2f}")
                
                # 如果有轮廓点，绘制轮廓；否则绘制矩形框
                if points and len(points) > 2:
                    print(f"  [{i}] {color_name} - {conf:.2f}")
                    print(f"  points: {points}")
                    # 绘制轮廓（绿色）
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(result_display, [pts], True, (0, 255, 0), 2)
                    # 填充半透明颜色
                    overlay = result_display.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, result_display, 0.85, 0, result_display)
                else:
                    # 绘制边界框（绿色）
                    cv2.rectangle(result_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签（左上角上方）
                label = f"{class_name} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result_display, (x1, y1-text_h-8), (x1+text_w+4, y1), (0, 255, 0), -1)
                cv2.putText(result_display, label, (x1+2, y1-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            cv2.imshow('Result', result_display)
            print("按任意键继续...")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            cv2.destroyWindow('Result')
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("="*70)
    print("颜色识别测试工具")
    print("="*70)
    print("1. 测试图像")
    print("2. 测试摄像头")
    print("="*70)
    
    choice = input("选择 (1/2): ").strip()
    
    if choice == '1':
        path = input("图像路径: ").strip()
        if os.path.exists(path):
            test_image(path)
        else:
            print(f"❌ 文件不存在: {path}")
    elif choice == '2':
        test_camera()
    else:
        print("❌ 无效选择")
