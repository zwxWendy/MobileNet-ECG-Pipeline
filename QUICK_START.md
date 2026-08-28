# MobileNet ECG Classification Pipeline - Quick Start Guide

## 快速开始（5分钟）

### 方式1：一键运行（推荐）

```bash
# Linux/Mac
bash setup.sh
python main.py --download --train --test

# Windows
setup.bat
python main.py --download --train --test
```

### 方式2：分步运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 快速检查
python quickstart.py

# 3. 下载数据
python data/download_dataset.py

# 4. 训练模型
python train.py

# 5. 测试模型  
python test.py
```

### 方式3：交互式教程

```bash
python TUTORIAL.py
```

---

## 完整流程详解

### 1️⃣ 环境检查

```bash
python quickstart.py
```

**输出包括：**
- ✓ PyTorch 版本和 CUDA 可用性
- ✓ 目录结构检查
- ✓ 当前配置总结
- ✓ 快速推理测试

---

### 2️⃣ 数据下载

```bash
python data/download_dataset.py
```

**功能：**
- 从 PhysioNet 自动下载 2017 ECG Challenge 数据集
- 验证数据完整性
- 输出日志到 `logs/download.log`

**数据统计：**
- 训练集：~8,500 条记录
- 4 个类别：N(正常), A(房颤), O(其他), ~(噪声)
- 采样率：300 Hz，30秒/条

---

### 3️⃣ 模型训练

```bash
python train.py
```

**功能：**
- 自动划分训练/验证/测试集（70%/15%/15%）
- 数据预处理（滤波、归一化、截断/填充）
- MobileNet 训练（深度可分离卷积）
- 实时指标计算
- 早停与自动检查点保存

**输出：**
- `checkpoints/best_model.pt` - 最佳模型
- `results/training_history.json` - 训练历史
- `logs/train.log` - 训练日志

**训练时间估算：**
| 硬件 | 时间 |
|------|------|
| CPU | 1-2 小时 |
| GPU (RTX 2080) | 10-15 分钟 |
| GPU (V100) | 3-5 分钟 |

---

### 4️⃣ 模型测试

```bash
python test.py
```

**功能：**
- 加载最佳模型
- 在测试集上评估
- 计算完整指标
- 支持两种阈值策略

**输出指标：**

#### 总体指标
```
Accuracy (准确率)      : 0.XXXX
Macro F1 (F1平均值)    : 0.XXXX
Weighted F1 (加权F1)  : 0.XXXX
Weighted AUC (加权AUC): 0.XXXX
```

#### 单类指标（对每个类别N, A, O, ~ 单独计算）
```
AUC        : ROC 曲线下面积 [0-1]
F1 Score   : 精确度与召回率的调和平均 [0-1]
Precision  : 预测正确率 = TP/(TP+FP)
Recall     : 召回率/敏感性 = TP/(TP+FN)
Specificity: 特异性 = TN/(TN+FP)
```

**输出文件：**
- `results/test_results_*.json` - 详细指标
- `results/predictions_*.npz` - 预测值和概率
- `logs/test.log` - 测试日志

---

## 配置调整指南

编辑 `config.py` 调整参数：

### 数据相关
```python
TRAIN_SPLIT = 0.7      # 训练集比例
VAL_SPLIT = 0.15       # 验证集比例
TEST_SPLIT = 0.15      # 测试集比例

USE_DATA_AUG = True    # 是否数据增强
AUG_NOISE_STD = 0.01   # 噪声标准差
AUG_SCALE_RANGE = (0.95, 1.05)  # 缩放范围
```

### 模型相关
```python
# 模型大小（0.5=小, 1.0=标准, 2.0=大）
MOBILENET_WIDTH_MULTIPLIER = 1.0
DROPOUT_RATE = 0.3     # Dropout 比率
BATCH_SIZE = 32        # 批大小
```

### 训练相关
```python
EPOCHS = 100           # 训练轮数
LEARNING_RATE = 1e-3   # 学习率
OPTIMIZER = "adam"     # 优化器 (adam/sgd)
WEIGHT_DECAY = 1e-4    # L2 正则化
EARLY_STOPPING_PATIENCE = 15  # 早停耐心值
```

### 测试相关
```python
THRESHOLD_STRATEGY = "max_prob"  # 阈值策略
# "max_prob": 选择最高概率的类
# "fixed": 只有概率 >= THRESHOLD_FIXED 时才分类

THRESHOLD_FIXED = 0.5  # 固定阈值
```

---

## 常见问题排查

### ❌ CUDA Out of Memory

**解决方案：**
```python
# config.py
BATCH_SIZE = 16  # 减小批大小
MOBILENET_WIDTH_MULTIPLIER = 0.5  # 减小模型
```

### ❌ 数据下载失败

**检查：**
```bash
# 验证网络连接
python -c "import wfdb; print(wfdb.get_record_list('challenge-2017')[:5])"
```

**手动下载：**
访问 https://physionet.org/content/challenge-2017/1.0.0/ 手动下载后放入 `data/challenge2017/`

### ❌ 导入错误

**重新安装依赖：**
```bash
pip install --upgrade -r requirements.txt
```

### ❌ 模型很差

**调整策略：**
1. 增加 `EPOCHS`（从 100 → 200）
2. 调整 `LEARNING_RATE`（尝试 1e-4 或 1e-2）
3. 启用数据增强：`USE_DATA_AUG = True`
4. 增大模型：`MOBILENET_WIDTH_MULTIPLIER = 1.5`

---

## 高级用法

### 自定义模型大小

```python
# config.py - 轻量化版本（移动设备）
MOBILENET_WIDTH_MULTIPLIER = 0.25  # 只有 12.5% 参数
BATCH_SIZE = 64  # 更大批量以补偿

# 或大型版本（更高精度）
MOBILENET_WIDTH_MULTIPLIER = 2.0   # 4 倍参数
BATCH_SIZE = 16  # 更小批量
```

### 导出为 ONNX（跨平台部署）

```python
import torch
from models import create_mobilenet_ecg

model = create_mobilenet_ecg()
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])

dummy_input = torch.randn(1, 1, 9000)
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['ecg_signal'],
    output_names=['classification'],
    dynamic_axes={'ecg_signal': {0: 'batch_size'}}
)
```

### 批量预测

```python
from models import create_mobilenet_ecg
from test import Tester
import numpy as np
import torch

model = create_mobilenet_ecg()
tester = Tester(model, device='cuda')
tester.load_checkpoint('checkpoints/best_model.pt')

# 自定义数据
my_signals = np.load('my_signals.npy')  # (N, 1, 9000)
labels = np.load('my_labels.npy')  # (N,)

# 预测
with torch.no_grad():
    signals_tensor = torch.FloatTensor(my_signals).cuda()
    output = model(signals_tensor)
    proba = torch.softmax(output, dim=1).cpu().numpy()
    predictions = np.argmax(proba, axis=1)
```

---

## 性能优化建议

| 目标 | 配置 |
|------|------|
| **快速原型** | `width=0.5, epochs=10, batch=64` |
| **最高精度** | `width=2.0, epochs=200, batch=16, lr=1e-4` |
| **移动部署** | `width=0.25, epochs=100, batch=32` |
| **边缘设备** | `width=0.5, epochs=50, batch=128` |

---

## 引用与参考

- Howard et al. 2017: MobileNets - Efficient CNNs for Mobile Vision
- Chollet 2017: Xception - Depthwise Separable Convolutions  
- PhysioNet Challenge 2017: AF Detection from Short ECG

---

## 获取帮助

1. 查看日志：`logs/` 目录
2. 检查配置：运行 `python quickstart.py`
3. 查看教程：`TUTORIAL.py`
4. 参考代码：各模块注释详细

