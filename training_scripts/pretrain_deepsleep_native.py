#!/usr/bin/env python3
"""
DeepSleep 原生模型预训练脚本
使用 DeepSleep 自研 MoE 架构（非 GPT-2）进行预训练
工业级训练：BF16、梯度累积、余弦退火、检查点恢复、损失曲线
"""
import os
import sys
import json
import gzip
import math
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 导入 DeepSleep 模型 ----
sys.path.insert(0, "/root/deepsleep/src")
from model.config import DeepSleepConfig
from model.modeling_deepsleep import DeepSleepForCausalLM


# ==================== 训练配置 ====================
class TrainConfig:
    # 模型架构 (~200M total, MoE alternating)
    vocab_size = 32000          # 匹配已有 tokenizer
    d_model = 1024
    n_layers = 10               # 5 dense + 5 MoE (alternating)
    n_heads = 8
    n_kv_heads = 4              # GQA: 8Q / 4KV
    max_position_embeddings = 1024
    num_experts = 6             # 5 routed + 1 shared
    num_routed_experts = 5
    num_shared_experts = 1
    top_k = 2
    aux_loss_coeff = 0.01
    z_loss_coeff = 0.001
    use_flash_attention = False # flash-attn 未安装，使用 SDPA
    tie_word_embeddings = True

    # 训练超参
    batch_size = 16
    gradient_accumulation = 4   # 有效 batch = 64
    max_seq_length = 512
    num_epochs = 1
    learning_rate = 6e-4
    min_lr = 6e-5
    weight_decay = 0.1
    warmup_steps = 500
    max_grad_norm = 1.0
    max_samples = 800_000       # 80 万条

    # 数据 & 分词器
    data_dir = "/root/autodl-tmp/data/IndustryCorpus_medicine"
    tokenizer_path = "/root/autodl-tmp/data/sleep_med_tokenizer_hf"

    # 输出
    output_dir = "/root/autodl-tmp/data/deepsleep_model_native"
    save_steps = 1000
    logging_steps = 50

    # BF16 (RTX 5090)
    use_bf16 = True
    seed = 42


# ==================== 数据集 ====================
class MedicalTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # padding 位置用 -100 屏蔽 loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_data(data_dir, max_samples=None):
    """从 IndustryCorpus 加载训练数据"""
    data_dir = Path(data_dir)
    texts = []

    # 英文
    en_files = sorted(data_dir.glob("en/*.jsonl.gz"))
    print(f"Loading EN data: {len(en_files)} files")
    for f in tqdm(en_files, desc="EN"):
        try:
            with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if max_samples and len(texts) >= max_samples // 2:
                        break
                    try:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if text and len(text) > 50:
                            texts.append(text)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except Exception:
            pass
        if max_samples and len(texts) >= max_samples // 2:
            break

    # 中文
    zh_files = sorted(data_dir.glob("zh/*.jsonl.gz"))
    print(f"Loading ZH data: {len(zh_files)} files")
    for f in tqdm(zh_files, desc="ZH"):
        try:
            with gzip.open(f, "rt", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if max_samples and len(texts) >= max_samples:
                        break
                    try:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if text and len(text) > 50:
                            texts.append(text)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except Exception:
            pass
        if max_samples and len(texts) >= max_samples:
            break

    print(f"Loaded {len(texts):,} texts total")
    return texts


# ==================== 学习率调度 ====================
def get_cosine_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ==================== 检查点 ====================
def save_model(model, tokenizer, save_dir):
    """保存模型和分词器，绕过 HuggingFace save_pretrained 的兼容性问题"""
    os.makedirs(save_dir, exist_ok=True)
    # 直接保存 state dict
    state_dict = model.state_dict()
    # 移除 tied weights 中重复的 key (lm_head.weight 与 embed_tokens.weight 共享)
    tied_key = "lm_head.weight"
    embed_key = "model.embed_tokens.embed_tokens.weight"
    if tied_key in state_dict and embed_key in state_dict:
        if torch.equal(state_dict[tied_key], state_dict[embed_key]):
            del state_dict[tied_key]
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    # 保存 config
    model.config.save_pretrained(save_dir)
    # 保存 tokenizer
    tokenizer.save_pretrained(save_dir)


def save_checkpoint(model, optimizer, global_step, best_loss, recent_losses,
                    output_dir, tokenizer, rng_states):
    """保存完整训练状态，支持恢复训练"""
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"step-{global_step}")
    save_model(model, tokenizer, ckpt_dir)

    # 优化器 & 调度器状态
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "best_loss": best_loss,
        "recent_losses": recent_losses,
        "rng_states": rng_states,
    }, os.path.join(ckpt_dir, "trainer_state.pt"))

    print(f"  -> Checkpoint saved: {ckpt_dir}")


def load_checkpoint(ckpt_path, model, optimizer):
    """从检查点恢复训练状态"""
    state = torch.load(os.path.join(ckpt_path, "trainer_state.pt"), map_location="cpu")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return (
        state["global_step"],
        state["best_loss"],
        state.get("recent_losses", []),
        state.get("rng_states", None),
    )


def get_rng_states():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_states(states):
    if states is None:
        return
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch"])
    if states["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


# ==================== 训练 ====================
def train(resume_from=None):
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

    # 固定随机种子
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载 tokenizer ----
    print("=" * 60)
    print("DeepSleep Native MoE Pretraining")
    print("=" * 60)
    print(f"\nLoading tokenizer from {cfg.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab_size={tokenizer.vocab_size}, pad={tokenizer.pad_token_id}, "
          f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")

    # ---- 创建 DeepSleep 模型 ----
    model_config = DeepSleepConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        num_experts=cfg.num_experts,
        num_routed_experts=cfg.num_routed_experts,
        num_shared_experts=cfg.num_shared_experts,
        top_k=cfg.top_k,
        aux_loss_coeff=cfg.aux_loss_coeff,
        z_loss_coeff=cfg.z_loss_coeff,
        use_flash_attention=cfg.use_flash_attention,
        tie_word_embeddings=cfg.tie_word_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        layer_pattern="alternating",
    )

    model = DeepSleepForCausalLM(model_config)
    model.to(device)

    # 启用 gradient checkpointing 节省显存
    model.model.gradient_checkpointing = True
    # 确保非连续的forward输出不会出问题
    model.model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 估算 active params (每个 token 实际计算的参数量)
    # Dense layers: 全部参数都活跃
    dense_params = 0
    moe_total_params = 0
    moe_active_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'is_moe'):
            if module.is_moe:
                layer_attn = sum(p.numel() for p in module.self_attn.parameters())
                layer_norm = sum(p.numel() for p in module.input_layernorm.parameters()) + \
                             sum(p.numel() for p in module.post_attention_layernorm.parameters())
                moe_total = sum(p.numel() for p in module.mlp.parameters())
                # Active: shared expert + top_k routed experts
                shared_params = sum(p.numel() for p in module.mlp.shared_expert.parameters())
                router_params = sum(p.numel() for p in module.mlp.gate.parameters())
                routed_expert_params = sum(p.numel() for p in module.mlp.experts[0].parameters())
                active_moe = shared_params + router_params + cfg.top_k * routed_expert_params

                moe_total_params += layer_attn + layer_norm + moe_total
                moe_active_params += layer_attn + layer_norm + active_moe
            else:
                dense_params += sum(p.numel() for p in module.parameters())

    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    final_norm_params = sum(p.numel() for p in model.model.norm.parameters())
    lm_head_params = 0 if cfg.tie_word_embeddings else sum(p.numel() for p in model.lm_head.parameters())

    active_per_token = embed_params + dense_params + moe_active_params + final_norm_params + lm_head_params

    print(f"\nModel: DeepSleep MoE (native architecture)")
    print(f"  Total params:    {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable:       {trainable_params:,}")
    print(f"  Active per token: ~{active_per_token:,} ({active_per_token/1e6:.1f}M)")
    print(f"  Dense layers:    {dense_params:,} ({dense_params/1e6:.1f}M)")
    print(f"  MoE layers:      {moe_total_params:,} total, {moe_active_params:,} active")
    print(f"  Embedding:       {embed_params:,} ({embed_params/1e6:.1f}M, tied={cfg.tie_word_embeddings})")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, "
          f"n_kv_heads={cfg.n_kv_heads}")
    print(f"  MoE: {cfg.num_routed_experts} routed + {cfg.num_shared_experts} shared, top_k={cfg.top_k}")

    # ---- 加载数据 ----
    print(f"\nLoading data (max {cfg.max_samples:,} samples)...")
    texts = load_data(cfg.data_dir, max_samples=cfg.max_samples)

    dataset = MedicalTextDataset(texts, tokenizer, max_length=cfg.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ---- 优化器 ----
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        fused=True,
    )

    # ---- 总步数 ----
    total_steps = len(dataloader) // cfg.gradient_accumulation * cfg.num_epochs

    # ---- BF16 ----
    use_bf16 = cfg.use_bf16 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"\nMixed precision: {'BF16' if use_bf16 else 'FP16'}")

    # ---- 恢复训练 ----
    global_step = 0
    best_loss = float("inf")
    recent_losses = []

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        global_step, best_loss, recent_losses, rng_states = load_checkpoint(
            resume_from, model, optimizer
        )
        set_rng_states(rng_states)
        model.to(device)
        print(f"  Resumed at step {global_step}, best_loss={best_loss:.4f}")

    # ---- 打印训练配置 ----
    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Samples:          {len(dataset):,}")
    print(f"  Batch size:       {cfg.batch_size}")
    print(f"  Grad accumulation:{cfg.gradient_accumulation}")
    print(f"  Effective batch:  {cfg.batch_size * cfg.gradient_accumulation}")
    print(f"  Seq length:       {cfg.max_seq_length}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  LR:               {cfg.learning_rate} -> {cfg.min_lr}")
    print(f"  Warmup:           {cfg.warmup_steps} steps")
    print(f"  Save every:       {cfg.save_steps} steps")
    print(f"{'='*60}")

    # ---- 训练循环 ----
    model.train()
    tokens_seen = 0
    start_time = time.time()
    log_entries = []

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # 前向 (BF16 autocast)
            # 不传 attention_mask: 模型默认使用 causal attention (is_causal=True)
            # padding 的 loss 通过 labels=-100 屏蔽
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                )
                loss = outputs.loss / cfg.gradient_accumulation

            # 反向
            loss.backward()

            # 梯度累积
            if (step + 1) % cfg.gradient_accumulation == 0:
                global_step += 1

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                # 手动余弦退火
                lr = get_cosine_lr(
                    global_step, cfg.warmup_steps, total_steps,
                    cfg.learning_rate, cfg.min_lr,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # 追踪 loss
                current_loss = loss.item() * cfg.gradient_accumulation
                recent_losses.append(current_loss)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)
                avg_loss = sum(recent_losses) / len(recent_losses)

                tokens_seen += input_ids.numel()
                elapsed = time.time() - start_time
                speed = tokens_seen / elapsed
                remaining = (total_steps - global_step) * (elapsed / max(global_step, 1))

                progress_bar.set_postfix({
                    "loss": f"{current_loss:.3f}",
                    "avg": f"{avg_loss:.3f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{speed:.0f}",
                    "eta": f"{remaining/60:.0f}m",
                })

                if avg_loss < best_loss and global_step >= cfg.warmup_steps:
                    best_loss = avg_loss

                # 保存 checkpoint
                if global_step % cfg.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, global_step, best_loss,
                        recent_losses, cfg.output_dir, tokenizer,
                        get_rng_states(),
                    )

                # 日志
                if global_step % cfg.logging_steps == 0:
                    log_entry = {
                        "step": global_step,
                        "loss": current_loss,
                        "avg_loss": avg_loss,
                        "best_loss": best_loss,
                        "lr": lr,
                        "tokens": tokens_seen,
                        "elapsed_sec": round(elapsed),
                        "timestamp": datetime.now().isoformat(),
                    }
                    log_entries.append(log_entry)
                    with open(os.path.join(cfg.output_dir, "train_log.jsonl"), "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                # 跳过已完成步数（resume 时）
                if global_step >= total_steps:
                    break

        print(f"\nEpoch {epoch+1} done - avg_loss(100): {avg_loss:.4f}")

    # ==================== 保存最终模型 ====================
    print("\nSaving final model...")
    final_dir = os.path.join(cfg.output_dir, "final_model")
    save_model(model, tokenizer, final_dir)

    # 训练报告
    total_time = time.time() - start_time
    report = {
        "total_steps": global_step,
        "total_tokens": tokens_seen,
        "total_time_hours": round(total_time / 3600, 2),
        "best_loss": round(best_loss, 4),
        "final_loss": round(avg_loss, 4),
        "tokens_per_second": round(tokens_seen / total_time),
        "total_params_M": round(total_params / 1e6, 1),
        "active_params_M": round(active_per_token / 1e6, 1),
        "model_arch": "DeepSleep MoE (native)",
        "config": {
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "num_experts": cfg.num_experts,
            "top_k": cfg.top_k,
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.max_seq_length,
        },
    }
    with open(os.path.join(cfg.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Steps:      {global_step:,}")
    print(f"  Time:       {total_time/3600:.2f}h ({total_time/60:.0f}m)")
    print(f"  Final loss: {avg_loss:.4f}")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Speed:      {tokens_seen/total_time:.0f} tok/s")
    print(f"  Model:      {cfg.output_dir}")
    print(f"{'='*60}")

    # ---- 损失曲线 ----
    plot_training_curves(log_entries, cfg.output_dir)

    # ---- 快速生成测试 ----
    run_generation_test(model, tokenizer, device)

    return model, tokenizer


def plot_training_curves(log_entries, output_dir):
    """绘制训练损失和学习率曲线"""
    if not log_entries:
        return

    steps = [e["step"] for e in log_entries]
    losses = [e["loss"] for e in log_entries]
    avg_losses = [e["avg_loss"] for e in log_entries]
    lrs = [e["lr"] for e in log_entries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 损失 (raw + 移动平均)
    axes[0].plot(steps, losses, alpha=0.3, label="raw loss", linewidth=1)
    axes[0].plot(steps, avg_losses, linewidth=2, label="avg(100)", color="blue")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 学习率
    axes[1].plot(steps, lrs, linewidth=2, color="orange")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("LR Schedule (Cosine)")
    axes[1].grid(True, alpha=0.3)

    # log scale loss
    axes[2].plot(steps, avg_losses, linewidth=2, color="green")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Loss (log)")
    axes[2].set_title("Training Loss (log scale)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")


def run_generation_test(model, tokenizer, device):
    """训练后快速生成测试"""
    print("\n" + "=" * 60)
    print("Generation Test")
    print("=" * 60)
    model.eval()

    prompts = [
        "Sleep apnea is",
        "失眠的主要症状包括",
        "The treatment for insomnia includes",
        "睡眠呼吸暂停综合征的诊断标准是",
        "Melatonin is a hormone that",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Q: {prompt}")
        print(f"  A: {generated}")

    model.train()


# ==================== 入口 ====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    train(resume_from=args.resume)
