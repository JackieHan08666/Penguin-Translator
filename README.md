# 🐧 Penguin-Translator: NMT Ablation Study

![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **"Ein kleiner Pinguin sucht nach seiner Liebe."** (A small penguin is looking for its love.)
> 
> 本项目是 Transformer 模型在英德翻译任务上的深度消融实验研究。通过对位置编码、多头注意力和模型深度的剥离测试，定量分析了 Transformer 架构在小规模数据集（Multi30k）上的稳健性。

---

## 📖 目录
- [项目背景](#-项目背景)
- [消融实验设计](#-消融实验设计)
- [快速开始](#-快速开始)
- [实验数据分析](#-实验数据分析)
- [人机对话示例](#-人机对话示例)
- [致谢与引用](#-致谢与引用)

---

## 🧩 项目背景
在 Transformer 架构中，各个组件（Multi-head Attention, PE, Stacked Layers）的贡献度在不同规模的任务中表现不一。本项目通过 **Penguin-Translator** 系统，不仅实现了一个完整的英德翻译器，更通过工业级的自动化脚本探索了模型超参数的边界。

### 技术栈
- **Tokenizer**: Byte-Level BPE (Byte-Pair Encoding)，有效解决德语复合词 OOV 问题。
- **Optimization**: Noam Scheduler + Label Smoothing + Adam。
- **Hardware Acceleration**: Mixed Precision (FP16) 训练，支持 AMAX 高性能计算集群。

---

## 🔬 消融实验设计 (Ablation Study)

我们设计了 4 组对比实验，变量控制如下：

| 实验组 | $N$ (Layers) | $H$ (Heads) | $PE$ (Positional Encoding) | 目的 |
| :--- | :---: | :---: | :---: | :--- |
| **Base** | 2 | 4 | Yes | 建立性能基准 |
| **H-Ablation** | 2 | **1** | Yes | 验证多头注意力的语义提取能力 |
| **L-Ablation** | **1** | 4 | Yes | 测试模型深度对翻译准确度的影响 |
| **PE-Ablation**| 2 | 4 | **No** | 证明位置信息在 Non-RNN 架构中的核心地位 |

---

## 🚀 快速开始

### 🛠 环境配置
```bash
# 克隆仓库并安装依赖
git clone [https://github.com/your-username/Penguin-Translator.git](https://github.com/your-username/Penguin-Translator.git)
cd Penguin-Translator
pip install -r requirements.txt