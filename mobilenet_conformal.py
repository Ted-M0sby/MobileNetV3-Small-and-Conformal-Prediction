import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from typing import List, Dict, Optional

class MobileNetConformal:
    """
    结合MobileNetV3-Small和共形预测的图像分类器
    为预测结果提供不确定性量化
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化分类器
        
        Args:
            confidence_level: 置信水平，0-1之间
        """
        self.confidence_level = confidence_level
        self.model = None
        self.calibration_scores = None
        self.threshold = None
        self.class_names = self._get_imagenet_classes()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self._load_model()
    
    def _load_model(self):
        """加载预训练的MobileNetV3-Small模型"""
        try:
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.model.eval()  # 设置为评估模式
            print("✓ MobileNetV3-Small模型加载成功")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def _get_imagenet_classes(self) -> List[str]:
        """获取ImageNet的1000个类别名称"""
        # 简化的类别名称（实际使用时可以加载完整列表）
        return [f"class_{i}" for i in range(1000)]
    
    def calibrate(self, image_paths: List[str], true_labels: List[int]):
        """
        使用校准集进行共形预测校准
        
        Args:
            image_paths: 校准图像路径列表
            true_labels: 对应的真实标签
        """
        if len(image_paths) != len(true_labels):
            raise ValueError("图像路径和标签数量不匹配")
        
        print(f"开始校准，使用 {len(image_paths)} 张图像...")
        
        calibration_scores = []
        
        for img_path, true_label in zip(image_paths, true_labels):
            try:
                # 加载并预处理图像
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0)
                
                # 获取模型预测
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.softmax(output[0], dim=0)
                
                # 计算非符合度分数（1 - 真实类别的概率）
                true_prob = probabilities[true_label].item()
                score = 1.0 - true_prob
                calibration_scores.append(score)
                
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                continue
        
        self.calibration_scores = np.array(calibration_scores)
        
        # 计算校准阈值
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * self.confidence_level) / n
        self.threshold = np.quantile(self.calibration_scores, q_level, method='higher')
        
        print(f"✓ 校准完成，阈值: {self.threshold:.4f}")
    
    def predict(self, image_path: str, top_k: int = 5) -> Dict:
        """
        对单张图像进行预测，返回预测集合
        
        Args:
            image_path: 图像文件路径
            top_k: 考虑的前k个最可能类别
            
        Returns:
            包含预测结果的字典
        """
        if self.threshold is None:
            raise ValueError("请先调用calibrate方法进行校准")
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            
            # 模型预测
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output[0], dim=0)
            
            # 获取top-k预测
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # 应用共形预测
            prediction_set = []
            confidence_scores = []
            
            for prob, idx in zip(top_probs, top_indices):
                score = 1.0 - prob.item()
                if score <= self.threshold:
                    prediction_set.append({
                        'class_index': idx.item(),
                        'class_name': self.class_names[idx.item()],
                        'probability': prob.item(),
                        'conformity_score': score
                    })
                confidence_scores.append(score)
            
            # 基础预测（概率最高的类别）
            best_idx = top_indices[0].item()
            best_prob = top_probs[0].item()
            
            return {
                'base_prediction': {
                    'class_index': best_idx,
                    'class_name': self.class_names[best_idx],
                    'probability': best_prob
                },
                'prediction_set': prediction_set,
                'set_size': len(prediction_set),
                'confidence_level': self.confidence_level,
                'uncertainty': len(prediction_set) > 1  # 不确定性标志
            }
            
        except Exception as e:
            print(f"预测图像 {image_path} 时出错: {e}")
            return {'error': str(e)}
    
    def batch_predict(self, image_paths: List[str]) -> List[Dict]:
        """批量预测多张图像"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {'status': '模型未加载'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'model_name': 'MobileNetV3-Small',
            'parameters': f"{total_params:,}",
            'input_size': '224x224',
            'num_classes': 1000,
            'calibrated': self.threshold is not None,
            'confidence_level': self.confidence_level
        }


# 使用示例
# classifier = MobileNetConformal(confidence_level=0.95)
# classifier.calibrate(calib_images, calib_labels)
# result = classifier.predict("your_image.jpg")