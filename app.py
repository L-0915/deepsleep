#!/usr/bin/env python3
"""
DeepSleep Chat - 睡眠健康领域大模型对话界面
Gradio web UI with streaming inference
"""
import os
import sys
import re
import time
import torch
import json
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from model.config import DeepSleepConfig
from model.modeling_deepsleep import DeepSleepForCausalLM
from transformers import AutoTokenizer

# ---- 中文 decode 修复 ----
_CJK = r'一-鿿㐀-䶿豈-﫿'
_CJK_RE = re.compile(f'(?<=[{_CJK}])\\s+(?=[{_CJK}])')
_PUNCT_SPACE_RE = re.compile(r'\s+([，。！？；：、）】」』])')
_PRE_PUNCT_SPACE_RE = re.compile(r'([（【「『])\s+')


def clean_decode(text):
    """清理 decode 后的多余空格"""
    text = _CJK_RE.sub('', text)
    text = _PUNCT_SPACE_RE.sub(r'\1', text)
    text = _PRE_PUNCT_SPACE_RE.sub(r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ==================== 模型加载 ====================
class DeepSleepEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            cj = json.load(f)

        config = DeepSleepConfig(
            vocab_size=cj.get("vocab_size", 32000),
            d_model=cj.get("d_model", 768),
            n_layers=cj.get("n_layers", 8),
            n_heads=cj.get("n_heads", 8),
            n_kv_heads=cj.get("n_kv_heads", 4),
            max_position_embeddings=cj.get("max_position_embeddings", 2048),
            num_experts=cj.get("num_experts", 6),
            num_routed_experts=cj.get("num_routed_experts", 6),
            num_shared_experts=cj.get("num_shared_experts", 0),
            top_k=cj.get("top_k", 2),
            moe_intermediate_size=cj.get("moe_intermediate_size", 1472),
            aux_loss_coeff=cj.get("aux_loss_coeff", 0.1),
            z_loss_coeff=cj.get("z_loss_coeff", 0.01),
            use_flash_attention=cj.get("use_flash_attention", False),
            tie_word_embeddings=cj.get("tie_word_embeddings", True),
            layer_pattern=cj.get("layer_pattern", "all_moe"),
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        model = DeepSleepForCausalLM(config)
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu", weights_only=False,
        )
        if "lm_head.weight" not in state_dict and "model.embed_tokens.embed_tokens.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.embed_tokens.weight"]
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self.model = model
        total = sum(p.numel() for p in model.parameters())
        print(f"DeepSleep loaded: {total/1e6:.1f}M params, device={self.device}")

    def generate(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """生成完整回复，返回清理后的文本"""
        # few-shot 上下文：通过示例让模型自然学会身份和回答风格
        context = (
            "问：你是谁？\n"
            "答：我是DeepSleep（深睡），由L-0915个人开发的睡眠健康AI助手。\n"
            "问：你好\n"
            "答：你好！我是DeepSleep，由L-0915开发的睡眠健康AI助手。有什么睡眠相关的问题可以问我！\n"
            "问：失眠怎么办？\n"
            "答：失眠可以从以下几个方面改善：1.保持规律作息，每天固定时间上床和起床；2.睡前避免使用手机等电子设备；3.营造安静、黑暗、凉爽的睡眠环境；4.适当运动，但避免睡前剧烈运动；5.如果持续失眠，建议就医咨询。\n"
        )
        input_text = context + f"问：{prompt}\n答："
        enc = self.tokenizer(input_text, add_special_tokens=False)
        bos = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        input_ids = torch.cat([bos, inp], dim=1)

        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        full_text = clean_decode(full_text)
        # 取最后一个 "答：" 之后的内容
        answer = re.sub(r'.*答\s*[：:]\s*', '', full_text, flags=re.DOTALL)
        answer = re.sub(r'^问[：:].*?(?=\n|$)', '', answer).strip()
        return answer.strip()

    def generate_stream(self, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """生成完整回复后，模拟流式输出"""
        answer = self.generate(prompt, max_new_tokens, temperature, top_p)
        if not answer:
            yield answer
            return

        # 模拟逐字输出
        chunk_size = max(1, len(answer) // 40)
        for i in range(0, len(answer), chunk_size):
            yield answer[:i + chunk_size]
            time.sleep(0.02)


# ==================== Gradio App ====================

EXAMPLE_QUESTIONS = [
    "失眠了怎么办？有哪些有效的助眠方法？",
    "睡眠呼吸暂停综合征有哪些症状？",
    "褪黑素可以长期服用吗？有什么副作用？",
    "每天睡多久才算健康？",
    "宝宝晚上睡觉总是出汗正常吗？",
    "如何改善睡眠质量？",
]


def create_app():
    model_path = os.environ.get(
        "DEEPSLEEP_MODEL",
        os.path.join(os.path.dirname(__file__), "checkpoints", "deepsleep-final")
    )
    print(f"Loading model from {model_path}...")
    engine = DeepSleepEngine(model_path)

    def chat_respond(message, history, max_tokens, temperature, top_p):
        """处理用户消息，流式返回（Gradio 6.x messages 格式）"""
        if not message.strip():
            return

        # 添加用户消息
        history = history + [{"role": "user", "content": message}]

        # 流式生成 assistant 回复
        buffer = ""
        for partial in engine.generate_stream(
            prompt=message,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
        ):
            buffer = partial
            yield history + [{"role": "assistant", "content": buffer}]

    # ---- UI ----
    with gr.Blocks(title="DeepSleep - 睡眠健康AI助手") as app:
        gr.Markdown(
            """
            # 🌙 DeepSleep - 睡眠健康 AI 助手
            > 基于 DeepSleep MoE 大模型 (~201.6M 参数) 的睡眠健康领域智能问答系统
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    layout="bubble",
                    placeholder="你好！我是 DeepSleep 睡眠健康助手\n有任何睡眠相关的问题都可以问我",
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的睡眠健康问题...",
                        show_label=False,
                        scale=4,
                        autofocus=True,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                gr.Markdown("**快捷提问：**")
                with gr.Row():
                    for i in range(0, len(EXAMPLE_QUESTIONS), 3):
                        with gr.Column():
                            for j in range(i, min(i + 3, len(EXAMPLE_QUESTIONS))):
                                q = EXAMPLE_QUESTIONS[j]
                                gr.Button(q, size="sm").click(
                                    lambda x=q: x, outputs=msg_input
                                )

            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                max_tokens = gr.Slider(
                    64, 512, value=256, step=32,
                    label="最大生成长度",
                )
                temperature = gr.Slider(
                    0.1, 1.5, value=0.7, step=0.1,
                    label="Temperature (创造性)",
                )
                top_p = gr.Slider(
                    0.5, 1.0, value=0.9, step=0.05,
                    label="Top-p (多样性)",
                )
                gr.Markdown(
                    """
                    ---
                    ### 关于 DeepSleep
                    - **架构**: MoE (Mixture of Experts)
                    - **参数**: ~201.6M (64M active/token)
                    - **词表**: 32K BPE
                    - **训练**: Pretrain → SFT → DPO → LoRA
                    - **领域**: 睡眠健康 & 医学

                    ---
                    *仅供研究使用，不构成医疗建议*
                    """
                )

        # 事件绑定
        msg_input.submit(
            chat_respond,
            inputs=[msg_input, chatbot, max_tokens, temperature, top_p],
            outputs=[chatbot],
        )
        send_btn.click(
            chat_respond,
            inputs=[msg_input, chatbot, max_tokens, temperature, top_p],
            outputs=[chatbot],
        )

        gr.Markdown(
            """
            ---
            <center>DeepSleep v1.0 | Powered by MoE Architecture | © 2026 DeepSleep Team</center>
            """
        )

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSleep Chat Web UI")
    parser.add_argument("--model", type=str, default=None, help="模型路径 (默认: checkpoints/deepsleep-final 或 DEEPSLEEP_MODEL 环境变量)")
    parser.add_argument("--port", type=int, default=6006, help="服务端口 (默认: 6006)")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网链接")
    args = parser.parse_args()

    if args.model:
        os.environ["DEEPSLEEP_MODEL"] = args.model

    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
