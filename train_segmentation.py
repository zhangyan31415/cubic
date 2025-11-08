#!/usr/bin/env python3
"""
训练 YOLOv8 分割模型（支持多GPU）
与检测模型的区别：返回轮廓点而不只是边界框
"""
from ultralytics import YOLO
import torch

def train_segmentation():
    print("="*70)
    print("🎲 魔方分割模型训练（YOLO11-seg - 2024最新版本）")
    print("="*70)
    
    # 数据集配置
    data_yaml = "./rubik-cube-last-1/data.yaml"
    
    print(f"\n✅ 数据集: {data_yaml}")
    
    # 检测硬件资源
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if gpu_count == 0:
        print("⚠ 未检测到 GPU，使用 CPU")
        config = {
            'model': 'yolo11m-seg.pt',  # YOLO11 nano（最新版本）
            'epochs': 50,
            'imgsz': 640,
            'batch': 8,
            'name': 'rubik_cube_yolo11_seg',
            'device': 'cpu',
            'workers': 4,
        }
    else:
        print(f"✓ 检测到 {gpu_count} 个 GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # 让用户选择使用几个GPU
        print(f"\n💡 请选择使用多少个GPU (1-{gpu_count}):")
        print("  1: 单卡训练（稳定，2-3小时）")
        if gpu_count >= 2:
            print("  2: 双卡训练（快一倍，1-1.5小时）")
        if gpu_count >= 4:
            print("  4: 四卡训练（快4倍，30-45分钟）")
        if gpu_count >= 8:
            print("  8: 八卡训练（最快，20-30分钟）")
        
        while True:
            try:
                num_gpus = int(input(f"\n请输入GPU数量 (1-{gpu_count}，回车默认1): ").strip() or "1")
                if 1 <= num_gpus <= gpu_count:
                    break
                else:
                    print(f"❌ 请输入1到{gpu_count}之间的数字")
            except ValueError:
                print("❌ 请输入有效的数字")
        
        # 根据GPU数量自动调整参数（避免OOM）
        gpu_configs = {
            1: {'imgsz': 640, 'batch': 16, 'workers': 8},
            2: {'imgsz': 640, 'batch': 24, 'workers': 12},
            4: {'imgsz': 800, 'batch': 48, 'workers': 20},
            8: {'imgsz': 1024, 'batch': 64, 'workers': 32},
        }
        
        # 找到最接近的配置
        best_cfg = gpu_configs.get(num_gpus)
        if not best_cfg:
            # 插值计算
            if num_gpus < 4:
                best_cfg = gpu_configs[2]
            else:
                best_cfg = gpu_configs[4]
        
        print(f"\n⚙️ 使用 {num_gpus} 个GPU训练")
        
        if num_gpus == 1:
            device = 0
        else:
            device = list(range(num_gpus))
        
        config = {
            'model': 'yolo11m-seg.pt',
            'epochs': 100,
            'imgsz': best_cfg['imgsz'],
            'batch': best_cfg['batch'],
            'name': 'rubik_cube_yolo11_seg',
            'device': device,
            'workers': best_cfg['workers'],
            'cache': 'ram',
            'amp': True,
            'close_mosaic': 20,
        }
    
    print(f"\n⚙️ 训练参数:")
    print(f"  模型: {config['model']} (YOLO11分割模型)")
    print(f"  Epochs: {config['epochs']}")
    print(f"  图像大小: {config['imgsz']}")
    
    if isinstance(config['device'], int):
        num_gpu_used = 1
        print(f"  Batch: {config['batch']} (单GPU)")
        print(f"  GPU: 单卡 (GPU {config['device']})")
    elif isinstance(config['device'], list):
        num_gpu_used = len(config['device'])
        print(f"  Batch: {config['batch']} (每GPU约 {config['batch']//num_gpu_used})")
        print(f"  GPU: {num_gpu_used} 卡并行 (GPU {', '.join(map(str, config['device']))})")
    else:
        num_gpu_used = 0
        print(f"  Batch: {config['batch']}")
        print(f"  Device: {config['device']}")
    
    print(f"  Workers: {config['workers']}")
    print(f"  数据缓存: {config.get('cache', 'False')}")
    print(f"  混合精度: {config.get('amp', False)}")
    
    choice = input("\n开始训练？(y/n): ").strip().lower()
    if choice != 'y':
        print("已取消")
        return
    
    print("\n" + "="*70)
    print("🚀 开始训练分割模型...")
    print("="*70)
    
    # 加载模型
    print(f"\n📦 加载 {config['model']} 预训练模型...")
    model = YOLO(config['model'])
    
    # 训练
    print("🔥 开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        name=config['name'],
        device=config['device'],
        workers=config['workers'],
        patience=10,
        save=True,
        plots=True,
        val=True,
    )
    
    print("\n" + "="*70)
    print("✅ YOLO11分割模型训练完成！")
    print("="*70)
    print(f"\n📦 模型保存位置: runs/segment/{config['name']}/weights/best.pt")
    
    print("\n📊 训练统计:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        if 'metrics/mAP50(M)' in metrics:
            print(f"  mAP50: {metrics['metrics/mAP50(M)']:.4f}")
        if 'metrics/mAP50-95(M)' in metrics:
            print(f"  mAP50-95: {metrics['metrics/mAP50-95(M)']:.4f}")
    
    print("\n🔍 验证命令:")
    print(f"  yolo segment val model=runs/segment/{config['name']}/weights/best.pt data={data_yaml}")
    
    print("\n🎯 测试命令:")
    print(f"  yolo segment predict model=runs/segment/{config['name']}/weights/best.pt source=path/to/image.jpg")
    
    print("\n💾 复制模型到本地:")
    print(f"  scp runs/segment/{config['name']}/weights/best.pt ~/code/cubic/models/rubik_cube_yolo11_seg.pt")
    
    print("\n⚡ 性能统计:")
    if isinstance(config['device'], int):
        print(f"  训练用时: 预计 2-3 小时（单GPU H200）")
    else:
        print(f"  训练用时: 预计 30-60 分钟（{len(config['device'])} GPU并行）")
    print(f"  最终模型大小: 约 50-100 MB")
    print(f"  推理速度: 预计 3-8ms/张（YOLO11比v8快20-30%）")

if __name__ == "__main__":
    train_segmentation()

