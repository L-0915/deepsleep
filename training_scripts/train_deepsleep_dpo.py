#!/usr/bin/env python3
"""
DeepSleep DPO Training - 基于人类偏好对齐
工业级标准：reference model、DPO loss、eval、生成采样、完整指标
"""
import os, sys, json, math, time, copy, re, torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, "/root/deepsleep/src")
from model.config import DeepSleepConfig
from model.modeling_deepsleep import DeepSleepForCausalLM

# ---- 中文 decode 修复 ----
_CJK = r'一-鿿㐀-䶿豈-﫿'
_CJK_RE = re.compile(f'(?<=[{_CJK}])\\s+(?=[{_CJK}])')
_CJK_PUNC_RE = re.compile(f'(?<=[{_CJK}])\\s+(?=[，。！？、：；）】])')
_CJK_OPEN_RE = re.compile(f'(?<=[（【])\\s+(?=[{_CJK}])')

def clean_chinese(text):
    text = _CJK_RE.sub('', text)
    text = _CJK_PUNC_RE.sub('', text)
    text = _CJK_OPEN_RE.sub('', text)
    return text


class Config:
    base_model = "/root/autodl-tmp/data/deepsleep_model_sft_v4/final_model"
    data_file = "/root/autodl-tmp/data/medical_evidence_DPO/merged_dpo_v2.jsonl"

    batch_size = 4
    gradient_accumulation = 4
    max_seq_length = 768
    num_epochs = 1          # DPO 只用 1 epoch，避免 over-alignment
    learning_rate = 5e-7    # 更小的 LR，保持生成能力
    min_lr_ratio = 0.1
    warmup_ratio = 0.05
    weight_decay = 0.01
    max_grad_norm = 1.0
    beta = 0.1              # 更小的 beta，温和对齐

    output_dir = "/root/autodl-tmp/data/deepsleep_model_dpo_v4_r2"
    save_steps = 200
    eval_steps = 100
    logging_steps = 25
    generate_steps = 200

    eval_ratio = 0.05       # 5% 数据做 eval

    test_prompts = [
        "问：失眠应该怎么办？\n答：",
        "问：睡眠呼吸暂停有哪些危害？\n答：",
        "问：褪黑素可以长期服用吗？\n答：",
        "问：高血压患者如何改善睡眠质量？\n答：",
        "问：经常做噩梦是什么原因？\n答：",
        "问：宝宝晚上睡觉总是出汗是什么原因？\n答：",
        "问：What are the symptoms of sleep apnea?\n答：",
    ]

    use_bf16 = True


class DPODataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def _tokenize_pair(self, prompt, response):
        """Tokenize prompt+response, return input_ids and response_mask"""
        full_text = f"问：{prompt}\n答：{response}"
        enc = self.tokenizer(full_text, max_length=self.max_length, truncation=True, add_special_tokens=False)
        ids = [self.tokenizer.bos_token_id] + enc['input_ids']
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        seq_len = len(ids)

        # 找到 "答：" 的位置来分割 prompt/response
        prompt_text = f"问：{prompt}\n答："
        prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_enc['input_ids']) + 1  # +1 for bos

        # response mask: 1 for response tokens
        response_mask = [0] * seq_len
        for i in range(max(0, prompt_len), seq_len):
            response_mask[i] = 1

        # Pad
        pad_len = self.max_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        ids = ids + [self.tokenizer.pad_token_id] * pad_len
        response_mask = response_mask + [0] * pad_len

        return ids, attention_mask, response_mask

    def __getitem__(self, idx):
        s = self.samples[idx]
        c_ids, c_mask, c_resp = self._tokenize_pair(s['prompt'], s['chosen'])
        r_ids, r_mask, r_resp = self._tokenize_pair(s['prompt'], s['rejected'])

        return {
            'chosen_ids': torch.tensor(c_ids, dtype=torch.long),
            'chosen_mask': torch.tensor(c_mask, dtype=torch.long),
            'chosen_resp_mask': torch.tensor(c_resp, dtype=torch.long),
            'rejected_ids': torch.tensor(r_ids, dtype=torch.long),
            'rejected_mask': torch.tensor(r_mask, dtype=torch.long),
            'rejected_resp_mask': torch.tensor(r_resp, dtype=torch.long),
        }


def get_log_probs(model, input_ids, attention_mask, response_mask):
    """计算 response 部分的 log probabilities"""
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    # Log softmax
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    # Gather log prob of actual token
    per_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    # Mask to response tokens only
    per_token_log_probs = per_token_log_probs * shift_mask
    # ★ Mean over response tokens (NOT sum) to avoid length bias
    n_tokens = shift_mask.sum(dim=-1).clamp(min=1)
    mean_log_prob = per_token_log_probs.sum(dim=-1) / n_tokens
    return mean_log_prob, n_tokens


def dpo_loss(policy_chosen_logps, ref_chosen_logps, policy_rejected_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss"""
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        margin = (chosen_rewards - rejected_rewards).mean()

    return loss, accuracy, margin, chosen_rewards.mean(), rejected_rewards.mean()


def evaluate_dpo(policy_model, ref_model, eval_loader, device, beta):
    policy_model.eval()
    total_loss = 0
    total_acc = 0
    total_margin = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            c_ids = batch['chosen_ids'].to(device)
            c_mask = batch['chosen_mask'].to(device)
            c_resp = batch['chosen_resp_mask'].to(device)
            r_ids = batch['rejected_ids'].to(device)
            r_mask = batch['rejected_mask'].to(device)
            r_resp = batch['rejected_resp_mask'].to(device)

            pol_c_logps, _ = get_log_probs(policy_model, c_ids, c_mask, c_resp)
            ref_c_logps, _ = get_log_probs(ref_model, c_ids, c_mask, c_resp)
            pol_r_logps, _ = get_log_probs(policy_model, r_ids, r_mask, r_resp)
            ref_r_logps, _ = get_log_probs(ref_model, r_ids, r_mask, r_resp)

            loss, acc, margin, _, _ = dpo_loss(pol_c_logps, ref_c_logps, pol_r_logps, ref_r_logps, beta)
            total_loss += loss.item()
            total_acc += acc.item()
            total_margin += margin.item()
            n_batches += 1

    policy_model.train()
    return {
        'loss': total_loss / n_batches,
        'accuracy': total_acc / n_batches,
        'margin': total_margin / n_batches,
    }


def generate_samples(model, tokenizer, prompts, device, max_new_tokens=200):
    model.eval()
    results = []
    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
        input_ids = torch.cat([bos, inp['input_ids']], dim=1)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens,
                temperature=0.7, top_k=40, top_p=0.9, do_sample=True,
                repetition_penalty=1.3, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        answer = text.split('答：', 1)[-1].strip() if '答：' in text else text
        # 去除 BPE 分词导致的中文多余空格
        import re
        answer = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', answer)
        answer = re.sub(r'(?<=[一-鿿])\s+(?=[，。！？、：；）】])', '', answer)
        answer = re.sub(r'(?<=[（【])\s+(?=[一-鿿])', '', answer)
        results.append({"prompt": prompt, "response": answer})
    model.train()
    return results


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _save_model(model, tokenizer, save_dir):
    """Save DeepSleep model in pytorch_model.bin format (handles tied weights)"""
    os.makedirs(save_dir, exist_ok=True)
    state_dict = model.state_dict()
    # Remove tied lm_head weight if identical to embedding
    if "lm_head.weight" in state_dict and "model.embed_tokens.embed_tokens.weight" in state_dict:
        if torch.equal(state_dict["lm_head.weight"], state_dict["model.embed_tokens.embed_tokens.weight"]):
            del state_dict["lm_head.weight"]
    torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
    model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def train():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/checkpoints", exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("DeepSleep DPO Training - 人类偏好对齐")
    print("=" * 60)

    # Load tokenizer and policy model
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model_config = DeepSleepConfig.from_pretrained(cfg.base_model)
    policy_model = DeepSleepForCausalLM(model_config)
    state_dict = torch.load(os.path.join(cfg.base_model, "pytorch_model.bin"), map_location="cpu", weights_only=False)
    if "lm_head.weight" not in state_dict and "model.embed_tokens.embed_tokens.weight" in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.embed_tokens.weight"]
    policy_model.load_state_dict(state_dict, strict=False)
    del state_dict
    policy_model.to(device)

    # Reference model (frozen copy)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"  Policy model: {sum(p.numel() for p in policy_model.parameters())/1e6:.1f}M params")
    print(f"  Reference model: frozen copy")

    # Load data
    print("\nLoading DPO data...")
    all_samples = []
    with open(cfg.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d.get('prompt') and d.get('chosen') and d.get('rejected'):
                all_samples.append(d)

    # Train/eval split
    import random
    random.seed(42)
    random.shuffle(all_samples)
    n_eval = max(int(len(all_samples) * cfg.eval_ratio), 1)
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]
    print(f"  Train: {len(train_samples):,}, Eval: {len(eval_samples):,}")

    train_dataset = DPODataset(train_samples, tokenizer, cfg.max_seq_length)
    eval_dataset = DPODataset(eval_samples, tokenizer, cfg.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer (only policy model params)
    decay_params = [p for n, p in policy_model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in policy_model.named_parameters() if p.dim() < 2]
    optimizer = AdamW(
        [{"params": decay_params, "weight_decay": cfg.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(0.9, 0.95), fused=True,
    )

    total_steps = len(train_loader) // cfg.gradient_accumulation * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    min_lr = cfg.learning_rate * cfg.min_lr_ratio

    print(f"\n{'='*60}")
    print(f"DPO Training config:")
    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Eval samples: {len(eval_samples):,}")
    print(f"  Batch size: {cfg.batch_size} × {cfg.gradient_accumulation} = {cfg.batch_size * cfg.gradient_accumulation}")
    print(f"  Seq length: {cfg.max_seq_length}")
    print(f"  Epochs: {cfg.num_epochs}")
    print(f"  Total gradient steps: {total_steps:,}")
    print(f"  Learning rate: {cfg.learning_rate} → {min_lr}")
    print(f"  Beta (DPO): {cfg.beta}")
    print(f"  Warmup: {warmup_steps} steps")
    print(f"{'='*60}")

    # ==================== Training Loop ====================
    policy_model.train()
    global_step = 0
    start_time = time.time()
    recent_losses = []
    best_eval_loss = float('inf')
    all_metrics = []

    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            c_ids = batch['chosen_ids'].to(device)
            c_mask = batch['chosen_mask'].to(device)
            c_resp = batch['chosen_resp_mask'].to(device)
            r_ids = batch['rejected_ids'].to(device)
            r_mask = batch['rejected_mask'].to(device)
            r_resp = batch['rejected_resp_mask'].to(device)

            # Policy model log probs
            pol_c_logps, pol_c_ntok = get_log_probs(policy_model, c_ids, c_mask, c_resp)
            pol_r_logps, pol_r_ntok = get_log_probs(policy_model, r_ids, r_mask, r_resp)

            # Reference model log probs (no grad)
            with torch.no_grad():
                ref_c_logps, ref_c_ntok = get_log_probs(ref_model, c_ids, c_mask, c_resp)
                ref_r_logps, ref_r_ntok = get_log_probs(ref_model, r_ids, r_mask, r_resp)

            # DPO loss
            loss, accuracy, margin, chosen_reward, rejected_reward = dpo_loss(
                pol_c_logps, ref_c_logps, pol_r_logps, ref_r_logps, cfg.beta
            )
            loss = loss / cfg.gradient_accumulation
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm)

                lr = get_lr(global_step, warmup_steps, total_steps, cfg.learning_rate, min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()

                cur_loss = loss.item() * cfg.gradient_accumulation
                recent_losses.append(cur_loss)
                if len(recent_losses) > 100:
                    recent_losses.pop(0)
                avg_loss = sum(recent_losses) / len(recent_losses)
                elapsed = time.time() - start_time
                remaining = (total_steps - global_step) * (elapsed / global_step)

                pbar.set_postfix({
                    'loss': f'{cur_loss:.3f}',
                    'avg': f'{avg_loss:.3f}',
                    'acc': f'{accuracy:.2f}',
                    'margin': f'{margin:.3f}',
                    'lr': f'{lr:.2e}',
                    'eta': f'{remaining/60:.0f}m',
                })

                # Logging
                if global_step % cfg.logging_steps == 0:
                    metrics = {
                        'step': global_step, 'epoch': epoch + 1,
                        'loss': cur_loss, 'avg_loss': avg_loss,
                        'accuracy': accuracy.item(),
                        'margin': margin.item(),
                        'chosen_reward': chosen_reward.item(),
                        'rejected_reward': rejected_reward.item(),
                        'lr': lr, 'elapsed': round(elapsed),
                    }
                    all_metrics.append(metrics)
                    with open(f"{cfg.output_dir}/train_log.jsonl", 'a') as f:
                        f.write(json.dumps(metrics) + '\n')

                # Eval
                if global_step % cfg.eval_steps == 0:
                    eval_result = evaluate_dpo(policy_model, ref_model, eval_loader, device, cfg.beta)
                    print(f"\n  [Eval] step={global_step}, loss={eval_result['loss']:.4f}, "
                          f"acc={eval_result['accuracy']:.3f}, margin={eval_result['margin']:.4f}")

                    eval_log = {'step': global_step, **eval_result, 'elapsed': round(time.time() - start_time)}
                    with open(f"{cfg.output_dir}/eval_log.jsonl", 'a') as f:
                        f.write(json.dumps(eval_log) + '\n')

                    if eval_result['loss'] < best_eval_loss:
                        best_eval_loss = eval_result['loss']
                        best_dir = f"{cfg.output_dir}/best_model"
                        os.makedirs(best_dir, exist_ok=True)
                        _save_model(policy_model, tokenizer, best_dir)
                        print(f"  [Best] New best! eval_loss={eval_result['loss']:.4f}")

                # Generation
                if global_step % cfg.generate_steps == 0:
                    samples = generate_samples(policy_model, tokenizer, cfg.test_prompts, device)
                    with open(f"{cfg.output_dir}/generation_samples.jsonl", 'a') as f:
                        for s in samples:
                            s['step'] = global_step
                            f.write(json.dumps(s, ensure_ascii=False) + '\n')
                    print(f"\n  [Sample] step={global_step}:")
                    for s in samples[:2]:
                        print(f"    Q: {s['prompt'][:30]}...")
                        print(f"    A: {s['response'][:80]}...")

                # Checkpoint
                if global_step % cfg.save_steps == 0:
                    ckpt_dir = f"{cfg.output_dir}/checkpoints/step-{global_step}"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    _save_model(policy_model, tokenizer, ckpt_dir)
                    print(f"  [Save] Checkpoint step-{global_step}")

    # ==================== Save Final ====================
    print(f"\nSaving final model...")
    final_dir = f"{cfg.output_dir}/final_model"
    _save_model(policy_model, tokenizer, final_dir)

    # Plot curves
    print(f"Plotting training curves...")
    if all_metrics:
        steps = [m['step'] for m in all_metrics]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # DPO Loss
        losses = [m['loss'] for m in all_metrics]
        axes[0,0].plot(steps, losses, alpha=0.3, linewidth=0.5)
        w = min(50, len(losses))
        if w > 1:
            ma = np.convolve(losses, np.ones(w)/w, mode='valid')
            axes[0,0].plot(steps[w-1:], ma, linewidth=2, color='blue', label=f'MA({w})')
        axes[0,0].set_title('DPO Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

        # Accuracy
        accs = [m['accuracy'] for m in all_metrics]
        axes[0,1].plot(steps, accs, linewidth=1.5, color='green')
        axes[0,1].set_title('Preference Accuracy'); axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Rewards
        cr = [m['chosen_reward'] for m in all_metrics]
        rr = [m['rejected_reward'] for m in all_metrics]
        axes[1,0].plot(steps, cr, linewidth=1.5, color='blue', label='Chosen reward')
        axes[1,0].plot(steps, rr, linewidth=1.5, color='red', label='Rejected reward')
        axes[1,0].set_title('Rewards'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

        # Margin
        margins = [m['margin'] for m in all_metrics]
        axes[1,1].plot(steps, margins, linewidth=1.5, color='purple')
        axes[1,1].set_title('Reward Margin (Chosen - Rejected)'); axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{cfg.output_dir}/training_curves.png", dpi=150, bbox_inches='tight')

    # Final eval
    final_eval = evaluate_dpo(policy_model, ref_model, eval_loader, device, cfg.beta)
    total_time = time.time() - start_time

    report = {
        'total_steps': global_step,
        'total_epochs': cfg.num_epochs,
        'total_samples': len(train_samples) * cfg.num_epochs,
        'total_time_minutes': round(total_time / 60),
        'final_train_loss': round(avg_loss, 4),
        'final_eval_loss': round(final_eval['loss'], 4),
        'final_eval_accuracy': round(final_eval['accuracy'], 4),
        'final_eval_margin': round(final_eval['margin'], 4),
        'best_eval_loss': round(best_eval_loss, 4),
        'beta': cfg.beta,
        'learning_rate': cfg.learning_rate,
    }
    with open(f"{cfg.output_dir}/report.json", 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"DPO Training Complete!")
    print(f"{'='*60}")
    print(f"  Steps: {global_step}")
    print(f"  Time: {total_time/60:.0f} min")
    print(f"  Final train loss: {avg_loss:.4f}")
    print(f"  Final eval loss: {final_eval['loss']:.4f}")
    print(f"  Final eval accuracy: {final_eval['accuracy']:.3f}")
    print(f"  Final eval margin: {final_eval['margin']:.4f}")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Model: {cfg.output_dir}/final_model/")
    print(f"{'='*60}")

    # Final generation
    print(f"\nFinal generation test:")
    samples = generate_samples(policy_model, tokenizer, cfg.test_prompts, device, max_new_tokens=200)
    for s in samples:
        print(f"\n  Q: {s['prompt'].strip()}")
        print(f"  A: {s['response'][:150]}...")


if __name__ == "__main__":
    train()
