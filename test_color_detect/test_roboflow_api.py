#!/usr/bin/env python3
"""
测试 Roboflow API
"""
import cv2
import numpy as np
import sys
import os

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    print("❌ 未安装 inference_sdk")
    print("安装命令: pip install inference-sdk")
    sys.exit(1)

# API配置
API_URL = "https://serverless.roboflow.com"
API_KEY = "mR4E9SSfj6upoRTIzEmP"
MODEL_ID = "rubik-cube-last/1"

def test_api(image_path):
    """测试API"""
    print(f"\n{'='*70}")
    print("Roboflow API 测试")
    print('='*70)
    print(f"API URL: {API_URL}")
    print(f"Model ID: {MODEL_ID}")
    print(f"图像: {image_path}")
    print('='*70)
    
    # 检查图像
    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法加载图像")
        return
    
    print(f"✓ 图像尺寸: {img.shape[1]}x{img.shape[0]}")
    
    # 初始化客户端
    print("\n正在调用 API...")
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )
    
    try:
        # 调用API
        result = client.infer(image_path, model_id=MODEL_ID)
        
        print("✓ API 调用成功\n")
        
        # 显示结果
        if 'predictions' in result:
            predictions = result['predictions']
            print(f"检测到 {len(predictions)} 个对象:\n")
            
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
            
            display = img.copy()
            
            for i, pred in enumerate(predictions):
                cls = pred.get('class', 'Unknown')
                conf = pred.get('confidence', 0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', 0)
                h = pred.get('height', 0)
                points = pred.get('points', [])
                
                # 计算边界框
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                
                color_name = color_map.get(cls, cls)
                print(f"[{i}] {color_name} ({cls}) - 置信度: {conf:.2f} - 位置: ({x1}, {y1}, {x2}, {y2})")
                
                # 如果有轮廓点，绘制轮廓；否则绘制矩形框
                if points and len(points) > 2:
                    # 绘制轮廓（绿色）
                    pts = np.array([[int(p['x']), int(p['y'])] for p in points], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                    # 填充半透明颜色
                    overlay = display.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
                else:
                    # 绘制边界框
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签
                label = f"{cls} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 背景框
                cv2.rectangle(display, (x1, y1-text_h-8), (x1+text_w+4, y1), (0, 255, 0), -1)
                cv2.putText(display, label, (x1+2, y1-4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 显示结果
            print("\n显示结果（按任意键退出）...")
            cv2.imshow('Roboflow API Result', display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        else:
            print("API 返回结果:")
            print(result)
    
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        import traceback
        traceback.print_exc()

def test_camera():
    """测试摄像头"""
    print(f"\n{'='*70}")
    print("实时摄像头 API 测试")
    print('='*70)
    print("按空格键检测，ESC退出")
    print('='*70)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    # 初始化客户端
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )
    print("✓ API 客户端初始化完成\n")
    
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
    
    temp_file = "/tmp/temp_frame.jpg"
    
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
            # 保存临时图像
            cv2.imwrite(temp_file, frame)
            
            print("\n正在调用 API...")
            
            try:
                # 调用API
                result = client.infer(temp_file, model_id=MODEL_ID)
                
                if 'predictions' in result:
                    predictions = result['predictions']
                    print(f"检测到 {len(predictions)} 个对象:")
                    
                    result_display = frame.copy()
                    
                    for i, pred in enumerate(predictions):
                        cls = pred.get('class', 'Unknown')
                        conf = pred.get('confidence', 0)
                        x = pred.get('x', 0)
                        y = pred.get('y', 0)
                        w = pred.get('width', 0)
                        h = pred.get('height', 0)
                        points = pred.get('points', [])
                        
                        x1 = int(x - w/2)
                        y1 = int(y - h/2)
                        x2 = int(x + w/2)
                        y2 = int(y + h/2)
                        
                        color_name = color_map.get(cls, cls)
                        print(f"  [{i}] {color_name} - {conf:.2f}")
                        
                        # 如果有轮廓点，绘制轮廓；否则绘制矩形框
                        if points and len(points) > 2:
                            # 绘制轮廓（绿色）
                            pts = np.array([[int(p['x']), int(p['y'])] for p in points], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(result_display, [pts], True, (0, 255, 0), 2)
                            # 填充半透明颜色
                            overlay = result_display.copy()
                            cv2.fillPoly(overlay, [pts], (0, 255, 0))
                            cv2.addWeighted(overlay, 0.15, result_display, 0.85, 0, result_display)
                        else:
                            # 绘制边界框
                            cv2.rectangle(result_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{cls} {conf:.2f}"
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
                
            except Exception as e:
                print(f"❌ API 调用失败: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 清理临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    print("="*70)
    print("Roboflow API 测试工具")
    print("="*70)
    print("1. 测试图像")
    print("2. 测试摄像头")
    print("="*70)
    
    choice = input("选择 (1/2): ").strip()
    
    if choice == '1':
        path = input("图像路径: ").strip()
        test_api(path)
    elif choice == '2':
        test_camera()
    else:
        print("❌ 无效选择")


