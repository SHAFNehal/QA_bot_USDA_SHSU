# QA Chat GUI

Simple chat interface to get answers from (1) the fine-tuned model, (2) RAG only, or (3) Hybrid RAG. Each mode has its own conversation. Plain, muted design.

## How to run

1. Open a terminal in the **project root** (the folder that contains `GUI`, `src`, and `config.py`).
2. Install dependencies if needed: `pip install -r requirements.txt`
3. Run:
   ```bash
   streamlit run GUI/app.py
   ```
4. Your browser will open the chat. Use the sidebar to choose **Answer from**: Fine-tuned model, RAG only, or Hybrid RAG.

## Config file (for non-technical users)

Edit **`GUI/gui_config.env`** in any text editor. Each line is either a comment (starts with `#`) or a setting in the form `NAME=value`. Paths are relative to the project root.

- **Fine-tuned model**
  - `BASE_MODEL` — Name of the base model (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
  - `PEFT_MODEL_PATH` — Folder where your trained (fine-tuned) weights are saved (e.g. `fine_tuned_weights`).

- **RAG only**
  - `RAG_DB_PATH` — Folder where the RAG database was built (e.g. `rag_db`). Build it first with the RAG ingest script if needed.
  - `RAG_MODEL_NAME` — Leave blank to use the default model, or set a model name.

- **Hybrid RAG**
  - `HYBRID_RAG_DB_PATH` — Same as the RAG database folder (e.g. `rag_db`).
  - Uses the same `BASE_MODEL` and `PEFT_MODEL_PATH` as the fine-tuned mode.
  - `HYBRID_SMALL_LLM` — Leave blank for default, or set a model name for restating/synthesizing.

- **Optional (all modes)**
  - `TEMPERATURE` — Number like `0.7` (higher = more varied answers).
  - `MAX_NEW_TOKENS` — Max length of each reply (e.g. `256`).
  - `RAG_TOP_K` — How many chunks RAG uses per question (e.g. `5`).

Save the file and restart the chat app if it is already running. If `gui_config.env` is missing, the app uses built-in defaults and shows a note in the sidebar.

## Muted theme

A muted color theme is set in `.streamlit/config.toml` at the project root. It is used automatically when you run `streamlit run GUI/app.py` from the project root.
