<div align="center">

# 🌙 DeepSleep

**睡眠健康领域轻量级大语言模型**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**从零开始训练的 ~201.6M 参数 MoE 大模型** | 中英双语 | 睡眠健康领域

[快速开始](#-快速开始) · [模型架构](#-模型架构) · [训练流程](#-训练流程) · [Web演示](#-web演示) · [项目结构](#-项目结构)

</div>

---

## 📖 项目简介

DeepSleep 是一个从零开始构建的睡眠健康领域大语言模型，灵感来自 [MiniMind](https://github.com/jingyaogong/minimind)。项目完整实现了大模型的全部流程：**Tokenizer 训练 → 预训练 → SFT → DPO → LoRA → Web 部署**，所有代码开源可复现。

**核心特点：**
- 🔬 **全流程从零构建** — Tokenizer、模型架构、训练循环全部手写
- 🧠 **MoE 架构** — DeepSeek 风格 softmax 路由，201.6M 总参数 / 64M 活跃参数
- 🇨🇳 **中英双语** — 32K BPE 词表，支持中文和英文输入
- 💻 **单卡可训练** — RTX 3090 (24GB) 即可完成全流程训练
- 🌐 **一键部署** — Gradio Web 界面，开箱即用

---

## 🏗 模型架构

```
DeepSleepForCausalLM
├── Embedding (vocab=32K, dim=768, tied with lm_head)
├── 8 × Decoder Layer (all MoE)
│   ├── DeepSleepAttention (GQA: 8Q / 4KV heads, head_dim=96, RoPE)
│   ├── RMSNorm (pre-norm)
│   └── DeepSleepMoE
│       ├── 6 routed experts + 0 shared experts, top_k=2
│       ├── DeepSeek-style softmax routing
│       ├── aux_loss = 0.1, z_loss = 0.01
│       └── SwiGLU expert FFN (moe_intermediate=1472)
├── Final RMSNorm
└── LM Head (tied, no bias)

总参数: ~201.6M | 每token激活: ~64M | 层模式: all_moe
```

| 超参数 | 值 |
|--------|-----|
| d_model | 768 |
| n_layers | 8 |
| n_heads | 8 (GQA: n_kv_heads=4) |
| head_dim | 96 |
| num_experts | 6 (routed) |
| top_k | 2 |
| moe_intermediate_size | 1472 |
| vocab_size | 32,000 |
| max_position_embeddings | 2048 |
| tie_word_embeddings | True |

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/L-0915/deepsleep.git
cd deepsleep

# 创建虚拟环境
conda create -n deepsleep python=3.10 -y
conda activate deepsleep

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型权重

从 [HuggingFace](https://huggingface.co/L-0915/deepsleep) 或 [GitHub Release](https://github.com/L-0915/deepsleep/releases) 下载模型权重：

```bash
# 方式一: 从 HuggingFace 下载
pip install huggingface_hub
huggingface-cli download L-0915/deepsleep --local-dir checkpoints/deepsleep-final

# 方式二: 手动下载后放入该目录
mkdir -p checkpoints/deepsleep-final
# 将以下文件放入 checkpoints/deepsleep-final/ 目录:
#   config.json (820B)
#   pytorch_model.bin (770MB)
#   tokenizer.json (1.7MB)
#   tokenizer_config.json (501B)
```

### 3. 启动 Web 演示

```bash
# 指定模型路径并启动
python app.py
```

浏览器访问 `http://localhost:6006` 即可体验对话。

---

## 📊 训练流程

DeepSleep 的训练分为四个阶段，全部在单张 RTX 3090 上完成：

```
Stage 1: 预训练 (Pretrain)
├── 数据: IndustryCorpus 医学语料 (150万条, 中英双语)
├── 配置: batch=16, seq_len=512, lr=1e-3, 3 epochs
├── 耗时: ~1.5 小时
└── 结果: loss 8.4 → 3.0

Stage 2: 监督微调 (SFT)
├── 数据: 中文医疗问答指令数据
├── 配置: batch=16, seq_len=768, lr=2e-5, 3 epochs
├── 耗时: ~2 小时
└── 结果: train loss 2.60, eval loss 1.84

Stage 3: 偏好对齐 (DPO)
├── 数据: 5,405 条医学 DPO 对 (5,351 通用 + 54 睡眠领域)
├── 配置: batch=16, seq_len=768, lr=5e-7, beta=0.1, 1 epoch
├── 耗时: ~4 分钟
└── 结果: train loss 0.624, eval accuracy 63.6%

Stage 4: LoRA 微调
├── 数据: 32,333 条单轮对话
├── 配置: r=16, alpha=32, lr=5e-5, 3 epochs
├── 耗时: ~11 分钟
└── 结果: train loss 5.20, eval loss 4.44
```

### 训练命令

```bash
# Stage 1: 预训练
python src/training/pretrain.py --config configs/train/pretrain.yaml

# Stage 2: SFT
python src/training/sft.py --config configs/train/sft.yaml

# Stage 3: DPO
python src/training/dpo.py --config configs/train/dpo.yaml

# Stage 4: LoRA (使用独立脚本)
python /path/to/train_deepsleep_lora.py
```

---

## 🌐 Web 演示

基于 Gradio 构建的交互式对话界面：

```bash
python app.py
```

**功能：**
- 💬 实时对话，流式输出
- 🎛 可调节 Temperature、Top-p、生成长度
- 📋 内置快捷提问按钮（睡眠健康相关）
- 🌙 清爽的睡眠主题 UI

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `checkpoints/deepsleep-final` | 模型路径 |
| `--port` | 6006 | 服务端口 |
| `--share` | False | 是否生成公网链接 |

也可通过环境变量配置：

```bash
DEEPSLEEP_MODEL=/path/to/model python app.py
```

---

## 📁 项目结构

```
deepsleep/
├── app.py                          # Gradio Web 演示 (一键启动)
├── requirements.txt                # Python 依赖
├── pyproject.toml                  # 项目配置
├── Makefile                        # 常用命令
│
├── src/
│   ├── model/                      # 模型架构
│   │   ├── config.py               # DeepSleepConfig 配置类
│   │   ├── modeling_deepsleep.py   # DeepSleepForCausalLM 主模型
│   │   ├── attention.py            # GQA + RoPE 注意力
│   │   ├── moe.py                  # DeepSeek-style MoE 路由
│   │   ├── layers.py               # Decoder Layer + SwiGLU MLP
│   │   ├── embedding.py            # 嵌入层
│   │   └── tokenization_deepsleep.py # 自定义 Tokenizer
│   │
│   ├── data/                       # 数据处理
│   │   ├── dataset/                # 数据集类 (Pretrain, SFT, DPO)
│   │   ├── crawling/               # 爬虫 (PubMed, arXiv, Wikipedia 等)
│   │   ├── processing/             # 清洗、去重、质量过滤
│   │   ├── synthetic/              # 合成数据生成
│   │   └── tokenizer/              # Tokenizer 训练
│   │
│   ├── training/                   # 训练循环
│   │   ├── pretrain.py             # 预训练 (FSDP, BF16)
│   │   ├── sft.py                  # SFT (全量 + LoRA)
│   │   ├── dpo.py                  # DPO 偏好对齐
│   │   ├── callbacks.py            # 训练回调
│   │   ├── loss.py                 # 损失函数
│   │   └── schedulers.py           # 学习率调度
│   │
│   ├── evaluation/                 # 评估框架
│   │   ├── benchmarks.py           # 基准测试
│   │   ├── judge.py                # LLM 评委
│   │   └── safety.py               # 安全评估
│   │
│   ├── inference/                  # 推理部署
│   │   ├── server.py               # FastAPI 服务 (OpenAI 兼容)
│   │   ├── chat.py                 # 命令行对话
│   │   └── quantize.py             # GGUF 量化导出
│   │
│   └── utils/                      # 工具函数
│       ├── distributed.py          # 分布式训练工具
│       ├── fsdp_config.py          # FSDP 配置
│       ├── checkpoint.py           # Checkpoint 管理
│       └── logging.py              # 日志
│
├── configs/                        # YAML 配置
│   ├── model/                      # 模型配置
│   ├── data/                       # 数据配置
│   ├── train/                      # 训练配置 (pretrain, sft, dpo)
│   ├── eval/                       # 评估配置
│   └── deploy/                     # 部署配置
│
├── tests/                          # 测试
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_data_pipeline.py
│   └── test_training.py
│
└── data_cleanse_pipeline.py        # 独立数据清洗管道
```

---

## 🔧 核心模块说明

### 模型架构 (`src/model/`)

| 文件 | 说明 |
|------|------|
| `config.py` | DeepSleepConfig — 所有模型超参数的 dataclass |
| `modeling_deepsleep.py` | DeepSleepForCausalLM — 完整的因果语言模型，兼容 HuggingFace |
| `attention.py` | GQA (Grouped Query Attention) + RoPE 旋转位置编码 |
| `moe.py` | DeepSeek-style softmax routing MoE，支持 aux_loss 和 z_loss |
| `layers.py` | Decoder 层 + SwiGLU MLP + RMSNorm |

### 训练 (`src/training/`)

| 文件 | 说明 |
|------|------|
| `pretrain.py` | 预训练入口，支持 FSDP + BF16 + 梯度累积 |
| `sft.py` | 监督微调，支持全量微调和 LoRA |
| `dpo.py` | DPO 偏好对齐，内置 reference model 冻结 |

---

## 📈 训练结果

### 训练曲线

| 阶段 | Train Loss | Eval Loss | 最佳指标 |
|------|-----------|-----------|---------|
| Pretrain | 3.00 | 2.73 | 150万条数据, 3 epochs |
| SFT | 2.60 | 1.84 | 中文医疗问答, 3 epochs |
| DPO | 0.624 | 0.628 | Accuracy 63.6%, 1 epoch |
| LoRA | 5.20 | 4.44 | 32K 单轮对话, 3 epochs |

### 硬件需求

| 阶段 | 显存需求 | 训练时间 |
|------|---------|---------|
| Pretrain | ~8 GB (BF16) | ~1.5 小时 |
| SFT | ~10 GB (BF16) | ~2 小时 |
| DPO | ~4 GB (BF16, 需同时加载 policy + ref) | ~4 分钟 |
| LoRA | ~4 GB (BF16) | ~11 分钟 |
| **合计** | **RTX 3090 (24GB) 足够** | **~4 小时** |

---

## 🗂 数据集

### 预训练数据

| 来源 | 数量 | 语言 |
|------|------|------|
| IndustryCorpus EN | 2,000,000 | 英文 |
| IndustryCorpus ZH | 1,214,293 | 中文 |
| 爬虫数据 (PubMed 等) | 23,504 | 中英混合 |
| **合计** | **3,237,797 docs** | **62% EN, 38% ZH** |

### SFT 数据
- 中文医疗问答指令数据 (`industry_instruction_fixed_医疗`)

### DPO 数据
- 5,351 条通用医学偏好对 + 54 条高质量睡眠领域偏好对

### LoRA 数据
- 32,333 条单轮对话数据 (心理咨询类)

---

## 💡 参考 & 致谢

- [MiniMind](https://github.com/jingyaogong/minimind) — 轻量级 LLM 训练框架，本项目的重要参考
- [Qwen2.5-MoE](https://qwenlm.github.io/blog/qwen2.5-moe/) — MoE 架构设计灵感
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — softmax routing + aux/z-loss 机制
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 模型框架
- [PEFT](https://github.com/huggingface/peft) — LoRA 实现

---

## 📄 License

MIT License

---

<div align="center">

**DeepSleep** © 2026 by L-0915

</div>
