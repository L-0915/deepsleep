#!/usr/bin/env python3
"""
DeepSleep LoRA Fine-Tuning
基于 DPO v2 模型的 LoRA 微调，使用单轮对话数据集
"""
import os
import sys
import json
import math
import time
import re
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

sys.path.insert(0, "/root/deepsleep/src")
from model.config import DeepSleepConfig
from model.modeling_deepsleep import DeepSleepForCausalLM
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ---- 中文 decode 修复 ----
_CJK = r'一-鿿㐀-䶿豈-﫿'
_CJK_RE = re.compile(f'(?<=[{_CJK}])\\s+(?=[{_CJK}])')


def decode_text(tokenizer, ids, skip_special=True):
    text = tokenizer.decode(ids, skip_special_tokens=skip_special)
    return _CJK_RE.sub('', text)


# ==================== 配置 ====================
class Config:
    # 基座模型（DPO v2）
    base_model_dir = "/root/autodl-tmp/data/deepsleep_model_dpo_v4_r2/final_model"

    # LoRA 数据
    train_file = "/root/autodl-tmp/data/lora-single-turn-dataset/single_datas.jsonl"

    # LoRA 超参
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # 训练超参
    batch_size = 16
    gradient_accumulation = 2        # 有效 batch = 32
    max_seq_length = 768
    num_epochs = 3
    learning_rate = 5e-5             # LoRA 通常用更大的学习率
    min_lr_ratio = 0.1
    warmup_ratio = 0.03
    weight_decay = 0.01
    max_grad_norm = 1.0

    # 输出
    output_dir = "/root/autodl-tmp/data/deepsleep_model_lora_v1"
    save_steps = 2000
    eval_steps = 500
    logging_steps = 50
    generate_steps = 2000

    # 生成测试 prompts
    test_prompts = [
        "问：我最近总是失眠，该怎么办？\n答：",
        "问：睡眠呼吸暂停有哪些症状？\n答：",
        "问：宝宝晚上睡觉总是出汗是什么原因？\n答：",
        "问：经常做噩梦怎么办？\n答：",
        "问：褪黑素可以长期服用吗？\n答：",
        "问：晚上总是焦虑睡不着，怎么缓解？\n答：",
    ]

    use_bf16 = True
    eval_ratio = 0.05               # 5% 数据用于 eval
    seed = 42


# ==================== 模型加载 ====================
def load_base_model(model_dir, device):
    """从 checkpoint 加载 DeepSleep 基座模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "pytorch_model.bin")

    with open(config_path) as f:
        config_json = json.load(f)

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
    if "lm_head.weight" not in state_dict and "model.embed_tokens.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.embed_tokens.weight"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model, tokenizer, config


def apply_lora(model, cfg):
    """应用 LoRA adapters"""
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ==================== 数据集 ====================
class LoRADataset(Dataset):
    """LoRA 单轮对话数据集

    数据格式: {"conversation_id": int, "conversation": [{"human": "...", "assistant": "..."}]}
    转换为: 问：{human}\n答：{assistant}\n 格式，仅对 assistant 回复计算 loss
    """

    def __init__(self, samples, tokenizer, max_length=768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conv = self.samples[idx]
        human_text = conv["human"].strip()
        assistant_text = conv["assistant"].strip()

        # 构建问/答格式（与 SFT 训练一致）
        question_seg = f"问：{human_text}\n"
        answer_seg = f"答：{assistant_text}\n"
        full_text = question_seg + answer_seg

        enc = self.tokenizer(
            full_text, max_length=self.max_length - 1,
            truncation=True, add_special_tokens=False,
        )
        input_ids = [self.tokenizer.bos_token_id] + enc["input_ids"]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        seq_len = len(input_ids)

        # 构建 assistant mask：仅标注 "答：..." 部分的 token
        assistant_mask = [0] * seq_len
        pos = 1  # 跳过 bos

        # 跳过 question 部分
        q_enc = self.tokenizer(question_seg, add_special_tokens=False)
        pos += len(q_enc["input_ids"])

        # 标注 answer 部分
        a_enc = self.tokenizer(answer_seg, add_special_tokens=False)
        for j in range(pos, min(pos + len(a_enc["input_ids"]), seq_len)):
            assistant_mask[j] = 1

        # Padding
        pad_len = self.max_length - seq_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len

        # Labels: 仅 assistant token 计算 loss
        labels = []
        for i in range(seq_len):
            labels.append(input_ids[i] if assistant_mask[i] == 1 else -100)
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_data(filepath, eval_ratio=0.05, seed=42):
    """加载并划分 train/eval 数据"""
    all_samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                convs = data.get("conversation", [])
                if not convs:
                    continue
                # 取第一条对话（单轮）
                first = convs[0]
                human = first.get("human", "").strip()
                assistant = first.get("assistant", "").strip()
                if human and assistant:
                    all_samples.append({"human": human, "assistant": assistant})
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    # 随机划分
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_samples))
    n_eval = max(1, int(len(all_samples) * eval_ratio))
    eval_indices = set(indices[:n_eval].tolist())

    train_samples = [s for i, s in enumerate(all_samples) if i not in eval_indices]
    eval_samples = [s for i, s in enumerate(all_samples) if i in eval_indices]

    return train_samples, eval_samples


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

    axes[0].plot(train_steps, train_losses, alpha=0.15, linewidth=0.5, color="blue")
    window = min(200, len(train_losses))
    if window > 1:
        ma = np.convolve(train_losses, np.ones(window) / window, mode="valid")
        axes[0].plot(train_steps[window - 1:], ma, linewidth=2, color="blue", label=f"Train (ma={window})")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("LoRA Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if eval_losses:
        axes[1].plot(eval_steps_list, eval_losses, linewidth=2, color="red", marker="o", markersize=4)
        axes[1].axhline(y=min(eval_losses), color="green", linestyle="--", alpha=0.5, label=f"Best: {min(eval_losses):.4f}")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Eval Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    if eval_losses:
        axes[2].plot(train_steps, train_losses, alpha=0.1, linewidth=0.5, color="blue")
        if window > 1:
            axes[2].plot(train_steps[window - 1:], ma, linewidth=1.5, color="blue", label="Train")
        axes[2].plot(eval_steps_list, eval_losses, linewidth=2, color="red", marker="o", markersize=4, label="Eval")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Loss")
        axes[2].set_title("Train vs Eval Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lora_training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ==================== 保存 ====================
def save_lora_adapter(model, tokenizer, save_dir):
    """保存 LoRA adapter（仅保存增量部分）"""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  LoRA adapter saved to {save_dir}")


def save_merged_model(model, tokenizer, save_dir):
    """合并 LoRA weights 并保存完整模型"""
    os.makedirs(save_dir, exist_ok=True)
    merged = model.merge_and_unload()
    state_dict = merged.state_dict()
    tied_key = "lm_head.weight"
    embed_key = "model.embed_tokens.embed_tokens.weight"
    if tied_key in state_dict and embed_key in state_dict:
        if torch.equal(state_dict[tied_key], state_dict[embed_key]):
            del state_dict[tied_key]
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    merged.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Merged model saved to {save_dir}")


# ==================== 主训练 ====================
def train():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    print("=" * 60)
    print("DeepSleep LoRA Fine-Tuning")
    print("=" * 60)

    # ---- 加载基座模型 ----
    print(f"\nLoading base model from {cfg.base_model_dir}")
    model, tokenizer, model_config = load_base_model(cfg.base_model_dir, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Base model params: {total_params/1e6:.1f}M")

    # ---- 应用 LoRA ----
    print(f"\nApplying LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
    model = apply_lora(model, cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    dtype = torch.bfloat16 if (cfg.use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"  Precision: {'BF16' if dtype == torch.bfloat16 else 'FP16'}")

    # ---- 加载数据 ----
    print(f"\nLoading data from {cfg.train_file}")
    train_samples, eval_samples = load_data(cfg.train_file, cfg.eval_ratio, cfg.seed)
    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Eval samples:  {len(eval_samples):,}")

    train_dataset = LoRADataset(train_samples, tokenizer, max_length=cfg.max_seq_length)
    eval_dataset = LoRADataset(eval_samples, tokenizer, max_length=cfg.max_seq_length)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ---- 优化器（仅训练 LoRA 参数）----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.gradient_accumulation
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    min_lr = cfg.learning_rate * cfg.min_lr_ratio

    print(f"\n{'='*60}")
    print(f"LoRA Training Config:")
    print(f"  Train samples:    {len(train_dataset):,}")
    print(f"  Eval samples:     {len(eval_dataset):,}")
    print(f"  Batch:            {cfg.batch_size} x {cfg.gradient_accumulation} = {cfg.batch_size * cfg.gradient_accumulation}")
    print(f"  Seq length:       {cfg.max_seq_length}")
    print(f"  Epochs:           {cfg.num_epochs}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  LR:               {cfg.learning_rate} -> {min_lr}")
    print(f"  Warmup:           {warmup_steps} steps ({cfg.warmup_ratio*100:.0f}%)")
    print(f"  LoRA r/alpha:     {cfg.lora_r}/{cfg.lora_alpha}")
    print(f"  Target modules:   {cfg.target_modules}")
    print(f"  Eval every:       {cfg.eval_steps} steps")
    print(f"  Save every:       {cfg.save_steps} steps")
    print(f"  Generate:         every {cfg.generate_steps} steps")
    print(f"  Output:           {cfg.output_dir}")
    print(f"{'='*60}")

    # LoRA 权重文件名（每次覆盖）
    d_model = model_config.d_model
    lora_ckpt_name = f"lora_deepsleep_{d_model}.pth"

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

            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / cfg.gradient_accumulation

            loss.backward()

            if (step + 1) % cfg.gradient_accumulation == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)

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

                    # Save best adapter
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_lora_adapter(model, tokenizer, os.path.join(cfg.output_dir, "best_adapter"))
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

                # Checkpoint: 保存 lora 权重为单文件（每次覆盖）
                if global_step % cfg.save_steps == 0:
                    lora_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                    lora_state["__meta__"] = {
                        "global_step": global_step,
                        "best_eval_loss": best_eval_loss,
                        "train_loss": avg_loss,
                        "d_model": d_model,
                        "lora_r": cfg.lora_r,
                        "lora_alpha": cfg.lora_alpha,
                    }
                    ckpt_path = os.path.join(cfg.output_dir, lora_ckpt_name)
                    torch.save(lora_state, ckpt_path)
                    print(f"  [Save] {lora_ckpt_name} (step={global_step}, {len(lora_state)} keys)")

                    # 同时更新训练曲线
                    plot_curves(all_train_steps, all_train_losses, all_eval_steps, all_eval_losses, cfg.output_dir)

    # ==================== 训练结束 ====================
    # 保存最终 lora 权重文件（覆盖式）
    print("\nSaving final LoRA weights...")
    lora_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    lora_state["__meta__"] = {
        "global_step": global_step,
        "best_eval_loss": best_eval_loss,
        "train_loss": avg_loss,
        "d_model": d_model,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
    }
    ckpt_path = os.path.join(cfg.output_dir, lora_ckpt_name)
    torch.save(lora_state, ckpt_path)
    print(f"  Saved {lora_ckpt_name} ({len(lora_state)} keys)")

    # 同时保留 peft adapter 格式（兼容推理加载）
    save_lora_adapter(model, tokenizer, os.path.join(cfg.output_dir, "final_adapter"))

    print("\nMerging LoRA weights into base model...")
    save_merged_model(model, tokenizer, os.path.join(cfg.output_dir, "final_model"))

    # 最终评估
    final_eval = evaluate(model, eval_loader, device, dtype)

    # 最终绘图
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
        "base_model_params_M": round(total_params / 1e6, 1),
        "trainable_params_M": round(trainable_params / 1e6, 2),
        "trainable_ratio_pct": round(100 * trainable_params / total_params, 2),
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.max_seq_length,
        "model_arch": "DeepSleep MoE + LoRA",
    }
    with open(os.path.join(cfg.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"LoRA Training Complete!")
    print(f"{'='*60}")
    print(f"  Steps:           {global_step:,}")
    print(f"  Time:            {total_time/60:.0f} min ({total_time/3600:.2f}h)")
    print(f"  Train loss:      {avg_loss:.4f}")
    print(f"  Eval loss:       {final_eval:.4f}")
    print(f"  Best eval:       {best_eval_loss:.4f}")
    print(f"  Trainable:       {trainable_params/1e6:.2f}M / {total_params/1e6:.1f}M")
    print(f"  Speed:           {tokens_seen/total_time:.0f} tok/s")
    print(f"  LoRA weights:    {cfg.output_dir}/{lora_ckpt_name}")
    print(f"  Adapter:         {cfg.output_dir}/final_adapter/")
    print(f"  Merged model:    {cfg.output_dir}/final_model/")
    print(f"  Best adapter:    {cfg.output_dir}/best_adapter/")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
