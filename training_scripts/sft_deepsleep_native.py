#!/usr/bin/env python3
"""
DeepSleep Native SFT - 医疗指令微调
基于 DeepSleep MoE 原生架构的监督微调
工业级标准：eval loss、best model 追踪、生成采样、完整指标
"""
import os
import sys
import json
import math
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

import torch
sys.path.insert(0, "/root/deepsleep/src")
from model.config import DeepSleepConfig
from model.modeling_deepsleep import DeepSleepForCausalLM
from transformers import AutoTokenizer
import re

# ---- 中文 decode 修复 ----
_CJK = r'一-鿿㐀-䶿豈-﫿'
_CJK_RE = re.compile(f'(?<=[{_CJK}])\\s+(?=[{_CJK}])')

def decode_text(tokenizer, ids, skip_special=True):
    text = tokenizer.decode(ids, skip_special_tokens=skip_special)
    return _CJK_RE.sub('', text)


# ==================== 配置 ====================
class Config:
    # 预训练模型（v4 fixed MoE with DeepSeek-style routing）
    pretrain_dir = "/root/autodl-tmp/data/deepsleep_model_v4/final_model"

    # SFT 数据
    train_file = "/root/autodl-tmp/data/IndustryInstruction_SFT-Medicine/industry_instruction_fixed_医疗_train.jsonl"
    eval_file = "/root/autodl-tmp/data/IndustryInstruction_SFT-Medicine/industry_instruction_fixed_医疗_eval.jsonl"

    # 训练超参
    batch_size = 16
    gradient_accumulation = 2        # 有效 batch = 32
    max_seq_length = 768
    num_epochs = 3
    learning_rate = 2e-5             # SFT 用小学习率
    min_lr_ratio = 0.1
    warmup_ratio = 0.03
    weight_decay = 0.01
    max_grad_norm = 1.0

    # 输出
    output_dir = "/root/autodl-tmp/data/deepsleep_model_sft_v4"
    save_steps = 2000
    eval_steps = 1000
    logging_steps = 50
    generate_steps = 2000

    # 生成测试 prompts
    test_prompts = [
        "问：失眠应该怎么办？\n答：",
        "问：What are the symptoms of sleep apnea?\n答：",
        "问：睡眠呼吸暂停怎么治疗？\n答：",
        "问：宝宝晚上睡觉总是出汗是什么原因？\n答：",
        "问：How does melatonin help with sleep?\n答：",
        "问：经常做噩梦是什么原因？\n答：",
        "问：褪黑素可以长期服用吗？\n答：",
    ]

    use_bf16 = True
    max_train_samples = None         # None = 全部数据
    seed = 42


# ==================== 模型保存（绕过 HF save_pretrained 兼容性问题）====================
def save_model(model, tokenizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()
    # 移除 tied weights 中重复的 key
    tied_key = "lm_head.weight"
    embed_key = "model.embed_tokens.embed_tokens.weight"
    if tied_key in state_dict and embed_key in state_dict:
        if torch.equal(state_dict[tied_key], state_dict[embed_key]):
            del state_dict[tied_key]
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def load_pretrained_model(pretrain_dir, device):
    """从预训练 checkpoint 加载 DeepSleep 模型"""
    tokenizer = AutoTokenizer.from_pretrained(pretrain_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_path = os.path.join(pretrain_dir, "config.json")
    model_path = os.path.join(pretrain_dir, "pytorch_model.bin")

    with open(config_path) as f:
        config_json = json.load(f)

    # 从保存的 config 中读取所有参数（适配不同版本的模型配置）
    def get(key, default=None):
        return config_json.get(key, default)

    config = DeepSleepConfig(
        vocab_size=get("vocab_size", 32000),
        d_model=get("d_model", 768),
        n_layers=get("n_layers", 8),
        n_heads=get("n_heads", 8),
        n_kv_heads=get("n_kv_heads", 4),
        max_position_embeddings=get("max_position_embeddings", 2048),
        num_experts=get("num_experts", 6),
        num_routed_experts=get("num_routed_experts", 6),
        num_shared_experts=get("num_shared_experts", 0),
        top_k=get("top_k", 2),
        moe_intermediate_size=get("moe_intermediate_size", 1472),
        aux_loss_coeff=get("aux_loss_coeff", 0.1),
        z_loss_coeff=get("z_loss_coeff", 0.01),
        use_flash_attention=get("use_flash_attention", False),
        tie_word_embeddings=get("tie_word_embeddings", True),
        layer_pattern=get("layer_pattern", "all_moe"),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = DeepSleepForCausalLM(config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    # 处理 tied weights（保存时可能删除了 lm_head.weight）
    if "lm_head.weight" not in state_dict and "model.embed_tokens.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    return model, tokenizer, config


# ==================== 数据集 ====================
class SFTDataset(Dataset):
    """SFT 数据集：多轮对话，仅对 assistant (gpt) 回复计算 loss"""

    def __init__(self, filepath, tokenizer, max_length=768, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print(f"Loading {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    convs = data.get("conversations", [])
                    if len(convs) < 2:
                        continue
                    self.samples.append(convs)
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Loaded {len(self.samples):,} conversations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        convs = self.samples[idx]
        segments = []
        for msg in convs:
            role = msg.get("from", "")
            value = msg.get("value", "").strip()
            if not value:
                continue
            if role == "human":
                segments.append((f"问：{value}\n", False))
            elif role == "gpt":
                segments.append((f"答：{value}\n", True))

        if not segments:
            segments = [("问：你好\n", False), ("答：你好！有什么可以帮你的吗？\n", True)]

        # 拼接全文
        full_text = "".join(s[0] for s in segments)

        enc = self.tokenizer(
            full_text, max_length=self.max_length - 1,
            truncation=True, add_special_tokens=False,
        )
        input_ids = [self.tokenizer.bos_token_id] + enc["input_ids"]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        seq_len = len(input_ids)

        # 构建 assistant mask：标注哪些 token 属于 gpt 回复
        assistant_mask = [0] * seq_len
        pos = 1  # 跳过 bos
        for text, is_assistant in segments:
            seg_enc = self.tokenizer(text, add_special_tokens=False)
            seg_len = len(seg_enc["input_ids"])
            if pos + seg_len > seq_len:
                seg_len = seq_len - pos
                if seg_len > 0 and is_assistant:
                    for j in range(pos, pos + seg_len):
                        assistant_mask[j] = 1
                break
            if is_assistant:
                for j in range(pos, pos + seg_len):
                    assistant_mask[j] = 1
            pos += seg_len

        # Padding
        pad_len = self.max_length - seq_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        # Labels: 仅 assistant token 计算 loss，其余 -100
        labels = []
        for i in range(seq_len):
            labels.append(input_ids[i] if assistant_mask[i] == 1 else -100)
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ==================== 评估 ====================
def evaluate(model, eval_dataloader, device, dtype):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
            n_valid = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_valid
            total_tokens += n_valid
    model.train()
    return total_loss / max(total_tokens, 1)


# ==================== 生成 ====================
def generate_samples(model, tokenizer, prompts, device, max_new_tokens=150):
    model.eval()
    results = []
    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
        input_ids = torch.cat([bos, inp["input_ids"].to(device)], dim=1)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = decode_text(tokenizer, out[0].tolist())
        results.append({"prompt": prompt, "response": text})
    model.train()
    return results


# ==================== 学习率 ====================
def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ==================== 绘图 ====================
def plot_curves(train_steps, train_losses, eval_steps_list, eval_losses, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Train loss
    axes[0].plot(train_steps, train_losses, alpha=0.15, linewidth=0.5, color="blue")
    window = min(200, len(train_losses))
    if window > 1:
        ma = np.convolve(train_losses, np.ones(window) / window, mode="valid")
        axes[0].plot(train_steps[window - 1 :], ma, linewidth=2, color="blue", label=f"Train (ma={window})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("SFT Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Eval loss
    if eval_losses:
        axes[1].plot(eval_steps_list, eval_losses, linewidth=2, color="red", marker="o", markersize=4)
        axes[1].axhline(y=min(eval_losses), color="green", linestyle="--", alpha=0.5, label=f"Best: {min(eval_losses):.4f}")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Eval Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Combined
    if eval_losses:
        axes[2].plot(train_steps, train_losses, alpha=0.1, linewidth=0.5, color="blue")
        if window > 1:
            axes[2].plot(train_steps[window - 1 :], ma, linewidth=1.5, color="blue", label="Train")
        axes[2].plot(eval_steps_list, eval_losses, linewidth=2, color="red", marker="o", markersize=4, label="Eval")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Train vs Eval Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sft_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ==================== 主训练 ====================
def train():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 60)
    print("DeepSleep Native SFT - 医疗指令微调")
    print("=" * 60)

    # ---- 加载预训练模型 ----
    print(f"\nLoading pretrained model from {cfg.pretrain_dir}")
    model, tokenizer, model_config = load_pretrained_model(cfg.pretrain_dir, device)

    # 关闭 gradient checkpointing（SFT 不需要，显存够）
    model.model.gradient_checkpointing = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params/1e6:.1f}M")

    dtype = torch.bfloat16 if (cfg.use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"  Precision: {'BF16' if dtype == torch.bfloat16 else 'FP16'}")

    # ---- 加载 SFT 数据 ----
    print("\nLoading SFT datasets...")
    train_dataset = SFTDataset(cfg.train_file, tokenizer, max_length=cfg.max_seq_length, max_samples=cfg.max_train_samples)
    eval_dataset = SFTDataset(cfg.eval_file, tokenizer, max_length=cfg.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- 优化器 ----
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = AdamW(
        [{"params": decay_params, "weight_decay": cfg.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(0.9, 0.95), fused=True,
    )

    total_steps = len(train_loader) // cfg.gradient_accumulation * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    min_lr = cfg.learning_rate * cfg.min_lr_ratio

    print(f"\n{'='*60}")
    print(f"SFT Training Config:")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Eval samples:  {len(eval_dataset):,}")
    print(f"  Batch:         {cfg.batch_size} x {cfg.gradient_accumulation} = {cfg.batch_size * cfg.gradient_accumulation}")
    print(f"  Seq length:    {cfg.max_seq_length}")
    print(f"  Epochs:        {cfg.num_epochs}")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  LR:            {cfg.learning_rate} -> {min_lr}")
    print(f"  Warmup:        {warmup_steps} steps ({cfg.warmup_ratio*100:.0f}%)")
    print(f"  Eval every:    {cfg.eval_steps} steps")
    print(f"  Save every:    {cfg.save_steps} steps")
    print(f"  Generate:      every {cfg.generate_steps} steps")
    print(f"{'='*60}")

    # ---- 训练循环 ----
    model.train()
    global_step = 0
    tokens_seen = 0
    start_time = time.time()
    recent_losses = []
    best_eval_loss = float("inf")
    all_train_losses = []
    all_train_steps = []
    all_eval_losses = []
    all_eval_steps = []

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # 不传 attention_mask，使用 causal attention
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / cfg.gradient_accumulation

            loss.backward()

            if (step + 1) % cfg.gradient_accumulation == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                lr = get_lr(global_step, warmup_steps, total_steps, cfg.learning_rate, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                current_loss = loss.item() * cfg.gradient_accumulation
                recent_losses.append(current_loss)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)
                avg_loss = sum(recent_losses) / len(recent_losses)
                tokens_seen += input_ids.numel()
                elapsed = time.time() - start_time
                tok_per_s = tokens_seen / elapsed
                remaining = (total_steps - global_step) * (elapsed / max(global_step, 1))

                all_train_losses.append(current_loss)
                all_train_steps.append(global_step)

                pbar.set_postfix({
                    "loss": f"{current_loss:.3f}",
                    "avg": f"{avg_loss:.3f}",
                    "lr": f"{lr:.2e}",
                    "tok/s": f"{tok_per_s:.0f}",
                    "eta": f"{remaining/60:.0f}m",
                })

                # Logging
                if global_step % cfg.logging_steps == 0:
                    log = {
                        "step": global_step, "epoch": epoch + 1,
                        "loss": current_loss, "avg_loss": avg_loss,
                        "lr": lr, "tokens": tokens_seen,
                        "tok_per_s": round(tok_per_s),
                        "elapsed_sec": round(elapsed),
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(os.path.join(cfg.output_dir, "train_log.jsonl"), "a") as f:
                        f.write(json.dumps(log) + "\n")

                # Eval
                if global_step % cfg.eval_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, dtype)
                    all_eval_losses.append(eval_loss)
                    all_eval_steps.append(global_step)
                    print(f"\n  [Eval] step={global_step}, eval_loss={eval_loss:.4f}, train_loss={avg_loss:.4f}")

                    eval_log = {"step": global_step, "eval_loss": eval_loss, "train_loss": avg_loss}
                    with open(os.path.join(cfg.output_dir, "eval_log.jsonl"), "a") as f:
                        f.write(json.dumps(eval_log) + "\n")

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_model(model, tokenizer, os.path.join(cfg.output_dir, "best_model"))
                        print(f"  [Best] New best! eval_loss={eval_loss:.4f}")

                # Generation samples
                if global_step % cfg.generate_steps == 0:
                    samples = generate_samples(model, tokenizer, cfg.test_prompts, device)
                    with open(os.path.join(cfg.output_dir, "generation_samples.jsonl"), "a") as f:
                        for s in samples:
                            s["step"] = global_step
                            f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    print(f"\n  [Sample] step={global_step}:")
                    for s in samples[:2]:
                        resp = s["response"]
                        prompt_text = s["prompt"].strip()
                        if resp.startswith(prompt_text[:20]):
                            resp = resp[len(prompt_text):]
                        print(f"    Q: {prompt_text[:50]}...")
                        print(f"    A: {resp.strip()[:100]}...")

                # Checkpoint
                if global_step % cfg.save_steps == 0:
                    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints", f"step-{global_step}")
                    save_model(model, tokenizer, ckpt_dir)
                    # 也保存优化器状态以支持恢复
                    torch.save({
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "best_eval_loss": best_eval_loss,
                        "recent_losses": recent_losses,
                    }, os.path.join(ckpt_dir, "trainer_state.pt"))
                    print(f"  [Save] Checkpoint step-{global_step}")

    # ==================== 训练结束 ====================
    print("\nSaving final model...")
    final_dir = os.path.join(cfg.output_dir, "final_model")
    save_model(model, tokenizer, final_dir)

    # 最终评估
    final_eval = evaluate(model, eval_loader, device, dtype)

    # 绘图
    plot_curves(all_train_steps, all_train_losses, all_eval_steps, all_eval_losses, cfg.output_dir)

    # 最终生成测试
    print("\nFinal generation test:")
    samples = generate_samples(model, tokenizer, cfg.test_prompts, device, max_new_tokens=200)
    for s in samples:
        prompt_text = s["prompt"].strip()
        resp = s["response"]
        if resp.startswith(prompt_text[:20]):
            resp = resp[len(prompt_text):]
        print(f"\n  Q: {prompt_text}")
        print(f"  A: {resp.strip()}")

    # 报告
    total_time = time.time() - start_time
    report = {
        "total_steps": global_step,
        "total_epochs": cfg.num_epochs,
        "total_train_samples": len(train_dataset),
        "total_eval_samples": len(eval_dataset),
        "total_tokens": tokens_seen,
        "total_time_hours": round(total_time / 3600, 2),
        "total_time_minutes": round(total_time / 60),
        "final_train_loss": round(avg_loss, 4),
        "final_eval_loss": round(final_eval, 4),
        "best_eval_loss": round(best_eval_loss, 4),
        "tok_per_second": round(tokens_seen / total_time),
        "model_params_M": round(total_params / 1e6, 1),
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.max_seq_length,
        "model_arch": "DeepSleep MoE (native)",
    }
    with open(os.path.join(cfg.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SFT Training Complete!")
    print(f"{'='*60}")
    print(f"  Steps:      {global_step:,}")
    print(f"  Time:       {total_time/60:.0f} min ({total_time/3600:.2f}h)")
    print(f"  Train loss: {avg_loss:.4f}")
    print(f"  Eval loss:  {final_eval:.4f}")
    print(f"  Best eval:  {best_eval_loss:.4f}")
    print(f"  Speed:      {tokens_seen/total_time:.0f} tok/s")
    print(f"  Final:      {cfg.output_dir}/final_model/")
    print(f"  Best:       {cfg.output_dir}/best_model/")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
