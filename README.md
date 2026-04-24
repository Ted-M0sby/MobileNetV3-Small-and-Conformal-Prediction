# MobileNetV3-Small + 共形预测（C++ 重写版）

本项目已从 Python 重写为 C++17 版本，保留了“校准 + 预测集合 + 不确定性量化”的核心流程。

## 项目结构

- `include/conformal_prediction.hpp`：通用共形预测组件（分类/回归）
- `include/mobilenet_conformal.hpp`：MobileNet 共形预测封装（可插拔后端）
- `src/main.cpp`：示例入口
- `CMakeLists.txt`：构建配置

## 构建

```bash
cmake -S . -B build
cmake --build build
```

## 运行

```bash
./build/gongxingyuce
```

> Windows + MSVC 下可执行文件位于 `build\\Debug\\gongxingyuce.exe` 或 `build\\Release\\gongxingyuce.exe`。

## 说明

当前默认使用 `MockMobileNetBackend` 作为演示后端（随机但可复现的概率分布），接口已抽象为：

- `IImageClassifierBackend::predictProbabilities(imagePath)`
- `IImageClassifierBackend::numClasses()`

后续接入 ONNX Runtime / LibTorch 时，仅需实现该接口。
