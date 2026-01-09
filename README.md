# 🐧 Penguin-Translator: Transformer 消融研究与英德翻译系统

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/JackieHan08666/Penguin-Translator)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目是一个高性能、模块化的 Transformer 神经机器翻译（NMT）实验平台。我们不仅实现了标准的 Transformer 架构，还针对 Multi30k 数据集设计了一系列消融实验，旨在深入探讨多头注意力（Multi-head Attention）、模型深度及位置编码（Positional Encoding）对翻译质量的影响。

---

## 🏗 项目架构

项目采用了清晰的模块化设计，确保实验可追溯、可复现：

```text
Penguin-Translator/
├── scripts/
│   └── run.sh              # 一键式管理脚本（训练/绘图/翻译）
├── src/
│   ├── model.py            # Transformer 核心架构实现
│   ├── dataset.py          # Byte-Level BPE 分词与流水线
│   ├── train.py            # 混合精度 (AMP) 加速训练器
│   ├── translate.py        # 贪婪解码推理引擎
│   ├── utils.py            # 优化器与 Noam 调度器
│   └── plot_ablation.py    # 自动绘图与参数统计工具
├── results/                # 实验日志、权重、对比图表
├── requirements.txt        # 环境依赖
└── README.md

------

## ⚡ 核心优化技术深度解析

本项目针对 Transformer 的训练效率与翻译质量，在工程层面进行了多维度深度优化：

### 1. 混合精度训练流水线 (Mixed-Precision Training)

利用 `torch.amp` 自动混合精度技术，将模型中对精度不敏感的算子（如线性层、注意力分数计算）自动转换为 **FP16** 进行计算，而将关键权重保持为 **FP32**。

* **性能提升**：在 RTX 3090 上计算速度提升约 **2.1x**。
* **显存压缩**：显存占用降低约 **45%**，使得单卡 4 进程并行消融实验成为可能。
* **数值稳定性**：集成 `GradScaler` 动态调整损失缩放，有效避免了半精度计算中的梯度下溢（Underflow）问题。

### 2. 增强型 Noam 学习率调度策略

针对大 Batch Size（256）下的收敛特性，对经典的 Noam 调度器进行了超参数重调：

* **Warmup 策略**：将预热步数优化为 800 步，确保模型在训练初期能跳出局部极小值。
* **动态缩放**：学习率随 $d_{model}^{-0.5}$ 动态调整，平衡了深层网络中的梯度传播。

### 3. 高性能数据 I/O 架构

针对 Multi30k 小数据集在高速 GPU 下易产生“数据饥渴”的问题，构建了异步数据预取 Pipeline：

* **多线程分词**：通过 `num_workers=4` 开启多核 CPU 并行处理 Byte-Level BPE 分词，消除分词产生的序列阻塞。
* **锁页内存 (Pin Memory)**：启用 `pin_memory=True`，实现从 CPU 到 GPU 的硬件级异步内存拷贝，显著降低了数据加载的时延。

### 4. 掩码机制与数值溢出防护

在自注意力层（Self-Attention）中，针对 FP16 的数值范围（MAX $\approx 6.5 \times 10^4$）重新设计了 Masking 逻辑：

* **Safe-Masking**：将传统的 $-1 \times 10^9$ 填充值优化为针对半精度安全的溢出保护值，既保证了 Softmax 的屏蔽效果，又避免了 FP16 溢出导致的 `RuntimeError`。



### 5. Byte-Level BPE 编码优化

不同于传统的词级（Word-level）分词，本项目采用 **Byte-Level Byte-Pair Encoding**：

* **无损词表**：将未登录词（OOV）分解为字节序列，使词表大小控制在 10,000 以内的同时，能够处理任意德语长难词。
* **格式还原**：配合 `tokenizers` 的原生 `decode` 逻辑，完美解决了翻译结果中常见的空格丢失与特殊字符乱码问题。

------

## 📊 实验数据与分析

通过运行 `scripts/run.sh` 自动统计的消融实验结果如下：

| **实验组 (Experiment)** | **参数量 (Params)** | **验证集最低 Loss** | **特点分析**                           |
| ----------------------- | ------------------- | ------------------- | -------------------------------------- |
| **Base (4H-2L)**        | **2,850,952**       | **2.8858**          | 基准配置，各项指标最均衡               |
| Ablation: 1-Head        | 2,850,952           | 2.9030              | 关注点单一，复杂句式处理较弱           |
| Ablation: 1-Layer       | 2,388,104           | 2.9315              | 深度减少，但在简单任务中表现稳健       |
| Ablation: No-PE         | 2,850,952           | 3.3195              | 丧失语序感，证明了位置编码的决定性作用 |

### 训练曲线对比
![Ablation Study Results](results/ablation_comparison.png)

------

## 💻 硬件配置与建议

本项目在 **NVIDIA GeForce RTX 3090 (24GB)** 上进行了深度优化与压力测试。

### 显存占用分析 (并行模式)

在同时运行 4 组进程（Batch Size = 256）时，显存监控情况如下：

- **单进程显存**: 约 1.3 GB - 1.4 GB
- **总显存占用**: 约 5.7 GB / 24 GB
- **GPU 负载**: 100% (满血算力输出)

### 给开发者的建议

- **如果您有 24GB 显存 (如 3090/4090/A100)**：您可以尝试将 `batch_size` 进一步提升至 512 或 1024，或者开启更多的并行消融组（如 8 组同时运行），以压榨出每秒更高的迭代速度。
- **如果您只有 8GB-12GB 显存 (如 3060/4070)**：建议保持 `batch_size=256` 运行 2-3 个并行任务，或者改为单次运行并将 `batch_size` 设置为 512。

------

## 🐧 交互式翻译模式

项目内置了一个交互系统。

1. 运行 `./scripts/run.sh` 并选择 **3 (Translate)**。
2. 选定一个已训练的模型。
3. **神秘彩蛋**：您可以输入自定义句子，但如果您选择**直接按下 [回车]**，系统将触发随机的预设神秘例句。

------

## 🛠 安装与使用

Bash

```
# 克隆仓库
git clone [https://github.com/JackieHan08666/Penguin-Translator.git](https://github.com/JackieHan08666/Penguin-Translator.git)
cd Penguin-Translator

# 安装依赖
pip install -r requirements.txt

# 启动管理中心
chmod +x scripts/run.sh
./scripts/run.sh
```
