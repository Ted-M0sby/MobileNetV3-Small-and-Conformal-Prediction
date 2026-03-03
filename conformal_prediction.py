import numpy as np
from typing import List, Tuple, Union, Optional
from sklearn.base import BaseEstimator

class ConformalPredictor:
    """
    基础共形预测类
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.nonconformity_scores = None
        self.calibration_threshold = None
    
    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray, 
            model: BaseEstimator, nonconformity_func: callable):
        """
        拟合共形预测模型
        
        Args:
            X_calib: 校准集特征
            y_calib: 校准集标签
            model: 基础预测模型
            nonconformity_func: 非符合度函数
        """
        self.model = model
        self.nonconformity_func = nonconformity_func
        
        # 计算校准集的非符合度分数
        self.nonconformity_scores = self._calculate_nonconformity_scores(X_calib, y_calib)
        
        # 计算校准阈值
        self.calibration_threshold = self._calculate_calibration_threshold()
        
        return self
    
    def predict(self, X_test: np.ndarray, candidate_labels: Optional[np.ndarray] = None) -> List[List]:
        """
        为测试样本生成预测集合
        
        Args:
            X_test: 测试样本
            candidate_labels: 候选标签（分类任务需要）
            
        Returns:
            每个测试样本的预测集合
        """
        if self.calibration_threshold is None:
            raise ValueError("模型未拟合，请先调用fit方法")
        
        predictions = []
        
        for i in range(len(X_test)):
            if candidate_labels is not None:
                # 分类任务：检查每个候选标签
                prediction_set = []
                for label in candidate_labels:
                    nonconformity_score = self.nonconformity_func(
                        self.model, X_test[i:i+1], label
                    )
                    if nonconformity_score <= self.calibration_threshold:
                        prediction_set.append(label)
                predictions.append(prediction_set)
            else:
                # 回归任务：计算预测区间
                base_prediction = self.model.predict(X_test[i:i+1])[0]
                lower_bound = base_prediction - self.calibration_threshold
                upper_bound = base_prediction + self.calibration_threshold
                predictions.append([lower_bound, upper_bound])
        
        return predictions
    
    def _calculate_nonconformity_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """计算非符合度分数"""
        scores = []
        for i in range(len(X)):
            score = self.nonconformity_func(self.model, X[i:i+1], y[i])
            scores.append(score)
        return np.array(scores)
    
    def _calculate_calibration_threshold(self) -> float:
        """计算校准阈值"""
        n = len(self.nonconformity_scores)
        q_level = np.ceil((n + 1) * self.confidence_level) / n
        threshold = np.quantile(self.nonconformity_scores, q_level, method='higher')
        return threshold


class ConformalClassifier(ConformalPredictor):
    """分类任务的共形预测"""
    
    def __init__(self, confidence_level: float = 0.95):
        super().__init__(confidence_level)
    
    @staticmethod
    def margin_nonconformity(model: BaseEstimator, X: np.ndarray, y: Union[int, float]) -> float:
        """基于间隔的非符合度函数（用于分类）"""
        probas = model.predict_proba(X)[0]
        true_class_prob = probas[int(y)]
        max_other_prob = np.max([p for i, p in enumerate(probas) if i != y])
        return max_other_prob - true_class_prob
    
    @staticmethod
    def hinge_nonconformity(model: BaseEstimator, X: np.ndarray, y: Union[int, float]) -> float:
        """基于hinge loss的非符合度函数"""
        probas = model.predict_proba(X)[0]
        true_class_prob = probas[int(y)]
        return 1 - true_class_prob


class ConformalRegressor(ConformalPredictor):
    """回归任务的共形预测"""
    
    def __init__(self, confidence_level: float = 0.95):
        super().__init__(confidence_level)
    
    @staticmethod
    def absolute_error_nonconformity(model: BaseEstimator, X: np.ndarray, y: Union[int, float]) -> float:
        """基于绝对误差的非符合度函数（用于回归）"""
        prediction = model.predict(X)[0]
        return abs(prediction - y)
    
    @staticmethod
    def squared_error_nonconformity(model: BaseEstimator, X: np.ndarray, y: Union[int, float]) -> float:
        """基于平方误差的非符合度函数"""
        prediction = model.predict(X)[0]
        return (prediction - y) ** 2


class InductiveConformalPredictor(ConformalPredictor):
    """归纳共形预测（更高效版本）"""
    
    def __init__(self, confidence_level: float = 0.95):
        super().__init__(confidence_level)
        self.proper_training_scores = None
    
    def fit(self, X_proper: np.ndarray, y_proper: np.ndarray, 
            X_calib: np.ndarray, y_calib: np.ndarray,
            model: BaseEstimator, nonconformity_func: callable):
        """
        拟合归纳共形预测模型
        
        Args:
            X_proper: 适当训练集特征
            y_proper: 适当训练集标签
            X_calib: 校准集特征
            y_calib: 校准集标签
            model: 基础预测模型
            nonconformity_func: 非符合度函数
        """
        # 在适当训练集上训练模型
        self.model = model
        self.model.fit(X_proper, y_proper)
        self.nonconformity_func = nonconformity_func
        
        # 计算校准集的非符合度分数
        self.nonconformity_scores = self._calculate_nonconformity_scores(X_calib, y_calib)
        
        # 计算校准阈值
        self.calibration_threshold = self._calculate_calibration_threshold()
        
        return self


def create_prediction_intervals_regression(
    model: BaseEstimator, 
    X_calib: np.ndarray, 
    y_calib: np.ndarray, 
    X_test: np.ndarray, 
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为回归任务创建预测区间的便捷函数
    
    Returns:
        lower_bounds: 预测区间下界
        upper_bounds: 预测区间上界
    """
    cp = ConformalRegressor(confidence_level)
    cp.fit(X_calib, y_calib, model, ConformalRegressor.absolute_error_nonconformity)
    
    predictions = cp.predict(X_test)
    lower_bounds = np.array([p[0] for p in predictions])
    upper_bounds = np.array([p[1] for p in predictions])
    
    return lower_bounds, upper_bounds


def create_prediction_sets_classification(
    model: BaseEstimator,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    confidence_level: float = 0.95
) -> List[List]:
    """
    为分类任务创建预测集合的便捷函数
    
    Returns:
        每个测试样本的预测标签集合
    """
    cp = ConformalClassifier(confidence_level)
    cp.fit(X_calib, y_calib, model, ConformalClassifier.margin_nonconformity)
    
    return cp.predict(X_test, classes)