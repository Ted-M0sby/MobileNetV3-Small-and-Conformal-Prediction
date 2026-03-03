# MobileNetV3-Small + 共形预测

一个结合MobileNetV3-Small预训练模型和共形预测技术的图像分类器，为预测结果提供统计保证的不确定性量化。




### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from mobilenet_conformal import MobileNetConformal

# 创建分类器
classifier = MobileNetConformal(confidence_level=0.95)

# 使用校准集进行校准（需要提供图像文件和真实标签）
calib_images = ["calib_img1.jpg", "calib_img2.jpg", ...]
calib_labels = [0, 1, ...]  # ImageNet类别索引
classifier.calibrate(calib_images, calib_labels)

# 预测单张图像
result = classifier.predict("test_image.jpg")
print(result)
```

### 预测结果示例

```python
{
    'base_prediction': {
        'class_index': 284,
        'class_name': 'class_284', 
        'probability': 0.85
    },
    'prediction_set': [
        {'class_index': 284, 'class_name': 'class_284', 'probability': 0.85},
        {'class_index': 283, 'class_name': 'class_283', 'probability': 0.12}
    ],
    'set_size': 2,
    'confidence_level': 0.95,
    'uncertainty': True
}
```

## 核心功能

### 1. 模型信息查询

```python
info = classifier.get_model_info()
print(info)
# 输出: {'model_name': 'MobileNetV3-Small', 'parameters': '2,400,000', ...}
```

### 2. 批量预测

```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = classifier.batch_predict(image_paths)
```

### 3. 不确定性解读

- **set_size = 1**: 模型很确定，只有一个可能的类别
- **set_size > 1**: 模型不确定，有多个可能的类别
- **uncertainty = True**: 建议人工复核

## 文件结构

```
.
├── mobilenet_conformal.py  # 主程序文件
├── conformal_prediction.py # 共形预测基础类
├── requirements.txt        # 依赖列表
└── README.md              # 说明文档
```

## 技术原理

### 共形预测

共形预测是一种为机器学习模型提供统计保证的框架：

1. **校准阶段**: 使用校准集计算非符合度分数
2. **预测阶段**: 基于校准阈值生成预测集合
3. **统计保证**: 真实标签有95%概率在预测集合中

### MobileNetV3-Small

- 参数量: 240万
- 输入尺寸: 224×224
- 输出类别: ImageNet 1000类
- 内存占用: ~100MB

1. **校准数据**: 需要提供代表性的校准图像集
2. **图像格式**: 支持常见图像格式（JPG、PNG等）
3. **内存需求**: 建议8GB以上内存
4. **首次运行**: 会自动下载预训练模型（约15MB）
