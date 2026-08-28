# MobileNet ECG Classification Pipeline

完整的 **MobileNet 深度可分离卷积** ECG 分类管道，包括数据下载、预处理、训练和测试。

## 功能特性

- ✅ **自动下载** PhysioNet 2017 ECG Challenge 数据集
- ✅ **完整预处理** 管道（滤波、归一化、重采样）
- ✅ **轻量化 MobileNet** 架构（深度可分离卷积）
- ✅ **训练/验证/测试** 集自动划分
- ✅ **混合精度训练** 支持
- ✅ **完整评估指标**：AUC、F1、Sensitivity、Specificity、Precision、Recall
- ✅ **固定阈值策略** 支持
- ✅ **模块化设计**，易于扩展

## 项目结构

```
.
├── config.py                    # 所有配置参数
├── main.py                      # 管道入口
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
├── data/
│   ├── __init__.py
│   ├── download_dataset.py      # 数据下载
│   ├── preprocess.py            # 信号预处理
│   └── data_loader.py           # 数据加载
├── models/
│   ├── __init__.py
│   └── mobilenet_ecg.py         # MobileNet 架构
├── utils/
│   ├── __init__.py
│   ├── logger.py                # 日志记录
│   └── metrics.py               # 指标计算
├── checkpoints/                 # 模型检查点
├── results/                     # 结果输出
└── logs/                        # 日志文件
```

## 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn
pip install wfdb
pip install tensorboard
```

## 快速开始

### 方式1：运行完整管道

```bash
# 下载数据 + 训练 + 测试
python main.py --download --train --test
```

### 方式2：分步运行

```bash
# 步骤1: 下载数据
python data/download_dataset.py

# 步骤2: 训练模型
python train.py

# 步骤3: 测试模型
python test.py
```

### 方式3：只运行特定步骤

```bash
# 仅训练（跳过下载）
python main.py --no-download --train

# 仅测试（使用已训练的模型）
python main.py --no-download --no-train --test
```

## 配置说明

编辑 `config.py` 调整所有参数：

### 数据集配置
```python
TRAIN_SPLIT = 0.7      # 训练集比例
VAL_SPLIT = 0.15       # 验证集比例
TEST_SPLIT = 0.15      # 测试集比例
RANDOM_SEED = 42       # 随机种子
```

### 模型配置
```python
MOBILENET_WIDTH_MULTIPLIER = 1.0  # 模型大小倍数 (0.5, 1.0, 1.5, 2.0)
DROPOUT_RATE = 0.3                # Dropout 率
BATCH_SIZE = 32                   # 批次大小
```

### 训练配置
```python
EPOCHS = 100                      # 训练轮数
LEARNING_RATE = 1e-3              # 学习率
OPTIMIZER = "adam"                # 优化器 (adam, sgd)
WEIGHT_DECAY = 1e-4               # 权重衰减
EARLY_STOPPING_PATIENCE = 15      # 早停耐心
```

### 测试配置
```python
THRESHOLD_STRATEGY = "max_prob"   # 阈值策略 (max_prob, fixed)
THRESHOLD_FIXED = 0.5             # 固定阈值
```

## 输出结果

### 模型检查点
- `checkpoints/best_model.pt` - 最佳模型
- `checkpoints/checkpoint_epoch_*.pt` - 定期保存的检查点

### 训练历史
- `results/training_history.json` - 每个 epoch 的指标

### 测试结果
- `results/test_results_*.json` - 完整评估指标
- `results/predictions_*.npz` - 预测结果和概率

### 日志
- `logs/pipeline.log` - 管道执行日志
- `logs/download.log` - 下载日志
- `logs/train.log` - 训练日志
- `logs/test.log` - 测试日志

## 评估指标详解

对于 PhysioNet 2017 数据集（4类：N, A, O, ~）：

### 总体指标
- **Accuracy**: (TP+TN)/(Total) - 总体准确率
- **Macro F1**: 各类 F1 的平均值（无加权）
- **Weighted F1**: 各类 F1 的加权平均值
- **Weighted AUC**: 各类 AUC 的加权平均值

### 单类指标（对每个类都计算）
- **AUC**: Area Under ROC Curve - 受试者工作特征曲线下面积
- **F1 Score**: 2×(Precision×Recall)/(Precision+Recall)
- **Precision**: TP/(TP+FP) - 精准度
- **Recall (Sensitivity)**: TP/(TP+FN) - 召回率/敏感性
- **Specificity**: TN/(TN+FP) - 特异性

### 阈值策略

1. **max_prob** (默认)
   - 选择概率最高的类
   - 适合明确的分类决策

2. **fixed**
   - 只有当最大概率 ≥ 固定阈值时才进行分类
   - 其他情况标记为"不确定"
   - 适合高置信度场景

## 示例：自定义模型大小

```python
# config.py

# 轻量级模型（~100KB）
MOBILENET_WIDTH_MULTIPLIER = 0.5  # 参数减少75%

# 标准模型（~1MB）
MOBILENET_WIDTH_MULTIPLIER = 1.0

# 较大模型（~4MB）
MOBILENET_WIDTH_MULTIPLIER = 2.0  # 参数增加4倍
```

## 性能参考

| 配置 | 参数数 | 内存 | 推理时间* |
|------|--------|------|----------|
| 0.5x (轻量) | ~50K | ~200KB | <50ms |
| 1.0x (标准) | ~200K | ~1MB | ~80ms |
| 2.0x (大) | ~800K | ~4MB | ~200ms |

*推理时间基于单条 9000 样本 ECG 信号，GPU 环境

## 已知问题和注意事项

1. **首次运行很慢**：数据下载和预处理需要时间（取决于网络和存储）
2. **GPU 内存**：若 CUDA OOM，减少 `BATCH_SIZE`
3. **类别不均衡**：数据集中各类别样本数不均匀，考虑使用加权损失
4. **数据路径**：确保数据下载到正确的位置 (`data/challenge2017/`)

## 扩展功能

### 1. 添加自定义数据增强
编辑 `data/data_loader.py` 中的 `apply_augmentation_to_batch()` 方法

### 2. 修改模型架构
编辑 `models/mobilenet_ecg.py` 中的 `mobilenet_config` 列表

### 3. 使用自定义损失函数
编辑 `config.py` 中的 `LOSS_FUNCTION` 并在 `train.py` 中实现

## 参考文献

- Howard et al. 2017: MobileNets - Efficient Convolutional Neural Networks for Mobile Vision Applications
- Chollet 2017: Xception - Deep Learning with Depthwise Separable Convolutions
- PhysioNet Challenge 2017: Atrial Fibrillation Detection from Short Single Lead ECG Recordings

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
