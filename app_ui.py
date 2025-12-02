"""
Gradio UI for the LLM QA Pipeline
Tabs:
  1) Dataset Creator
  2) Data Cleaner
  3) Fine-Tuning (LoRA)
  4) Inference Chat
"""

import os
import io
import json
import time
import traceback
import gradio as gr

# --- Import your project modules ---
# These imports assume app_ui.py sits next to the other .py files you uploaded.
# If your project is in a package folder, adjust imports accordingly.
from dataset_creator import QADatasetCreator
from data_cleaner import load_jsonl, clean_qa_dataset, save_jsonl, print_stats
from fine_tuner import load_model_and_tokenizer, load_dataset, train
from inference import QAInference


# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    if path:
        parent = os.path.dirname(path) if os.path.splitext(path)[1] else path
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

def prettify_stats(stats: dict) -> str:
    # Pretty text for cleaner stats (same numbers that print_stats shows)
    total = stats.get("total", 0)
    valid = stats.get("valid", 0)
    removed = total - valid if total else 0
    lines = [
        "==========================================",
        "DATA CLEANING STATISTICS",
        "==========================================",
        f"Total entries processed: {total}",
        f"Valid entries kept:     {valid}",
        f"Entries removed:        {removed}",
        f"Retention rate:         {((valid/total)*100 if total else 0):.1f}%",
        "",
        "Removal breakdown:",
        f"  Missing fields:  {stats.get('missing_fields', 0)}",
        f"  Invalid question:{stats.get('invalid_question', 0)}",
        f"  Invalid answer:  {stats.get('invalid_answer', 0)}",
        f"  Duplicates:      {stats.get('duplicate', 0)}",
        f"  Incomplete pairs:{stats.get('incomplete', 0)}",
    ]
    return "\n".join(lines)


# =========================
# 1) Dataset Creator
# =========================
def run_dataset_creator(
    input_dir: str,
    output_file: str,
    model_name: str,
    device: str,
    use_pipeline: bool,
    chunk_size: int,
    overlap: int,
    num_questions: int
):
    logs = io.StringIO()
    try:
        if not os.path.isdir(input_dir):
            return "", f"❌ Input directory not found: {input_dir}"

        ensure_dir(output_file)
        print(f"[Creator] input_dir={input_dir}", file=logs)
        print(f"[Creator] output_file={output_file}", file=logs)
        print(f"[Creator] model={model_name} device={device} pipeline={use_pipeline}", file=logs)
        print(f"[Creator] chunk_size={chunk_size} overlap={overlap} num_questions={num_questions}", file=logs)

        creator = QADatasetCreator(model_name=model_name, device=device, use_pipeline=use_pipeline)
        creator.create_dataset(
            input_dir=input_dir,
            output_file=output_file,
            chunk_size=int(chunk_size),
            overlap=int(overlap),
            num_questions_per_chunk=int(num_questions)
        )

        return f"✅ Dataset saved to: {output_file}", logs.getvalue()
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error: {e}", logs.getvalue() + "\n" + traceback.format_exc()


# =========================
# 2) Data Cleaner
# =========================
def run_data_cleaner(input_jsonl: str, output_jsonl: str, show_samples: bool):
    logs = io.StringIO()
    try:
        if not os.path.isfile(input_jsonl):
            return "", "", f"❌ Input file not found: {input_jsonl}"

        ensure_dir(output_jsonl)
        print(f"[Cleaner] input={input_jsonl}", file=logs)
        print(f"[Cleaner] output={output_jsonl}", file=logs)

        data = load_jsonl(input_jsonl)
        cleaned, stats = clean_qa_dataset(data)
        save_jsonl(cleaned, output_jsonl)

        # Build human summary
        stats_text = prettify_stats(stats)
        sample_block = ""
        if show_samples and cleaned:
            sample_block = "=== Sample of cleaned entries ===\n"
            for i, item in enumerate(cleaned[:3]):
                q = item.get('question', '')
                a = item.get('answer', '')
                sample_block += f"\nEntry {i+1}:\nQ: {q}\nA: {a}\n"

        return f"✅ Cleaned file saved to: {output_jsonl}", stats_text, logs.getvalue() + "\n" + sample_block
    except Exception as e:
        traceback.print_exc()
        return "", "", f"❌ Error: {e}\n" + traceback.format_exc()


# =========================
# 3) Fine-tuning
# =========================
def run_fine_tune(
    base_model_name: str,
    dataset_path: str,
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    logging_steps: int,
    save_steps: int
):
    logs = io.StringIO()
    try:
        if not os.path.isfile(dataset_path):
            return f"❌ Dataset file not found: {dataset_path}", ""

        ensure_dir(output_dir)
        print(f"[Train] base_model={base_model_name}", file=logs)
        print(f"[Train] dataset={dataset_path}", file=logs)
        print(f"[Train] output_dir={output_dir}", file=logs)
        print(f"[Train] epochs={num_train_epochs}, bs={per_device_train_batch_size}, lr={learning_rate}", file=logs)

        model, tokenizer = load_model_and_tokenizer(base_model_name)
        dset = load_dataset(dataset_path, tokenizer)

        train(
            model=model,
            tokenizer=tokenizer,
            dataset=dset,
            output_dir=output_dir,
            num_train_epochs=int(num_train_epochs),
            per_device_train_batch_size=int(per_device_train_batch_size),
            learning_rate=float(learning_rate),
            warmup_steps=int(warmup_steps),
            logging_steps=int(logging_steps),
            save_steps=int(save_steps)
        )

        return f"✅ Fine-tuned weights saved to: {output_dir}", logs.getvalue()
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error: {e}", logs.getvalue() + "\n" + traceback.format_exc()


# =========================
# 4) Inference / Chat
# =========================
# We keep a small cache so the model isn't reloaded every user message.
_infer_obj_cache = {}

def load_infer_obj(base_model: str, peft_path: str, device: str):
    key = (base_model, peft_path, device)
    if key not in _infer_obj_cache:
        _infer_obj_cache[key] = QAInference(base_model_name=base_model, peft_model_path=peft_path, device=device)
    return _infer_obj_cache[key]

def chat_predict(message, history, base_model, peft_dir, device, max_new_tokens, temperature, top_p):
    try:
        infer = load_infer_obj(base_model, peft_dir, device)
        answer = infer.generate_answer(
            question=message,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p)
        )
        return answer
    except Exception as e:
        return f"❌ Error during inference: {e}"


# =========================
# Build Gradio App
# =========================
with gr.Blocks(title="LLM QA Pipeline UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐝 LLM QA Pipeline — Visual UI\nFour steps: Generate → Clean → Fine-Tune → Chat")

    with gr.Tab("1) Dataset Creator"):
        with gr.Row():
            input_dir = gr.Textbox(label="Input directory (docs)", value="data_input")
            output_file = gr.Textbox(label="Output JSONL", value="data_output/qa_dataset.jsonl")
        with gr.Row():
            model_name = gr.Textbox(label="HuggingFace model", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="cuda", label="Device")
            use_pipeline = gr.Checkbox(value=True, label="Use transformers pipeline (faster to get started)")
        with gr.Row():
            chunk_size = gr.Number(label="Chunk size", value=1000, precision=0)
            overlap = gr.Number(label="Chunk overlap", value=200, precision=0)
            num_questions = gr.Number(label="Questions per chunk", value=3, precision=0)

        run_btn_1 = gr.Button("Generate Dataset", variant="primary")
        result_creator = gr.Textbox(label="Result", interactive=False)
        logs_creator = gr.Code(label="Logs")

        run_btn_1.click(
            fn=run_dataset_creator,
            inputs=[input_dir, output_file, model_name, device, use_pipeline, chunk_size, overlap, num_questions],
            outputs=[result_creator, logs_creator]
        )

    with gr.Tab("2) Data Cleaner"):
        with gr.Row():
            raw_jsonl = gr.Textbox(label="Raw dataset (JSONL)", value="data_output/qa_dataset.jsonl")
            cleaned_jsonl = gr.Textbox(label="Cleaned dataset (JSONL)", value="data_output/qa_dataset_cleaned.jsonl")
        show_samples = gr.Checkbox(value=True, label="Show a few cleaned samples")
        run_btn_2 = gr.Button("Clean Dataset", variant="primary")
        cleaner_result = gr.Textbox(label="Result", interactive=False)
        cleaner_stats = gr.Textbox(label="Stats", lines=14)
        cleaner_logs = gr.Code(label="Details / Samples")

        run_btn_2.click(
            fn=run_data_cleaner,
            inputs=[raw_jsonl, cleaned_jsonl, show_samples],
            outputs=[cleaner_result, cleaner_stats, cleaner_logs]
        )

    with gr.Tab("3) Fine-Tuning (LoRA)"):
        with gr.Row():
            ft_base_model = gr.Textbox(label="Base model", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            ft_dataset = gr.Textbox(label="Cleaned dataset (JSONL)", value="data_output/qa_dataset_cleaned.jsonl")
        with gr.Row():
            ft_output_dir = gr.Textbox(label="Output dir for LoRA weights", value="fine_tuned_weights")
        with gr.Row():
            ft_epochs = gr.Number(label="Epochs", value=10, precision=0)
            ft_batch = gr.Number(label="Batch size / device", value=2, precision=0)
            ft_lr = gr.Number(label="Learning rate", value=3e-5)
        with gr.Row():
            ft_warmup = gr.Number(label="Warmup steps", value=100, precision=0)
            ft_logging = gr.Number(label="Logging steps", value=20, precision=0)
            ft_savesteps = gr.Number(label="Save every N steps", value=200, precision=0)

        run_btn_3 = gr.Button("Start Fine-Tuning", variant="primary")
        ft_result = gr.Textbox(label="Result", interactive=False)
        ft_logs = gr.Code(label="Training Log")

        run_btn_3.click(
            fn=run_fine_tune,
            inputs=[ft_base_model, ft_dataset, ft_output_dir, ft_epochs, ft_batch, ft_lr, ft_warmup, ft_logging, ft_savesteps],
            outputs=[ft_result, ft_logs]
        )

    with gr.Tab("4) Inference Chat"):
        with gr.Row():
            inf_base = gr.Textbox(label="Base model", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            inf_peft = gr.Textbox(label="LoRA weights directory", value="fine_tuned_weights")
            inf_device = gr.Dropdown(choices=["auto", "cpu", "cuda"], value="auto", label="Device")
        with gr.Row():
            inf_max_new = gr.Slider(64, 1024, value=256, step=16, label="Max new tokens")
            inf_temp = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Temperature")
            inf_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

        # ChatInterface binds at build-time; wrap the fn to read current widgets
        def _chat_fn(msg, hist):
            return chat_predict(
                msg, hist,
                inf_base.value, inf_peft.value, inf_device.value,
                int(inf_max_new.value), float(inf_temp.value), float(inf_top_p.value)
            )

        chat = gr.ChatInterface(
            fn=_chat_fn,
            title="Fine-Tuned QA Chatbot",
            multimodal=False,
            examples=["What is the Asian citrus psyllid?", "How does LoRA fine-tuning work?", "Summarize my domain in two sentences."],
            textbox=gr.Textbox(placeholder="Ask a question from your domain…")
        )

    gr.Markdown("— Built for your pipeline. Happy fine-tuning!")
    
if __name__ == "__main__":
    # You can change server_name=\"0.0.0.0\" to expose on LAN if needed.
    demo.launch(server_name="127.0.0.1", server_port=7860)
