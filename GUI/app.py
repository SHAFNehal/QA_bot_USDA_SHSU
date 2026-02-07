"""
Streamlit chat GUI: Fine-tuned, RAG only, or Hybrid RAG.
Plain design, muted colors; separate conversation per mode.
Config: GUI/gui_config.env (key=value with comments).
Run from project root: streamlit run GUI/app.py
"""

import os
import sys
from pathlib import Path

# Project root = parent of GUI folder (so paths in config are relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import streamlit as st

# -----------------------------------------------------------------------------
# Config parser (key=value, skip # and blank lines)
# -----------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "gui_config.env"
PATH_KEYS = ("PEFT_MODEL_PATH", "RAG_DB_PATH", "HYBRID_RAG_DB_PATH")


def load_config():
    """Load gui_config.env; return dict. Paths resolved relative to PROJECT_ROOT."""
    out = {
        "BASE_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "PEFT_MODEL_PATH": "",
        "RAG_DB_PATH": "rag_db",
        "RAG_MODEL_NAME": "",
        "HYBRID_RAG_DB_PATH": "rag_db",
        "HYBRID_SMALL_LLM": "",
        "TEMPERATURE": "0.7",
        "MAX_NEW_TOKENS": "256",
        "RAG_TOP_K": "5",
    }
    if not CONFIG_PATH.is_file():
        return out
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                out[k] = v
    for key in PATH_KEYS:
        val = out.get(key, "")
        if val and not Path(val).is_absolute():
            out[key] = str(PROJECT_ROOT / val)
    return out


# -----------------------------------------------------------------------------
# Page config and muted theme
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="QA Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Muted colors: light gray backgrounds for chat messages
st.markdown("""
<style>
  .stChatMessage { border-radius: 0.5rem; }
  [data-testid="stAppViewContainer"] { background-color: #f5f5f5; }
  .block-container { padding-top: 1rem; max-width: 720px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

if "history_finetuned" not in st.session_state:
    st.session_state.history_finetuned = []
if "history_rag" not in st.session_state:
    st.session_state.history_rag = []
if "history_hybrid" not in st.session_state:
    st.session_state.history_hybrid = []
if "backend_finetuned" not in st.session_state:
    st.session_state.backend_finetuned = None
if "backend_rag" not in st.session_state:
    st.session_state.backend_rag = None
if "backend_hybrid" not in st.session_state:
    st.session_state.backend_hybrid = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "finetuned"

cfg = load_config()
config_exists = CONFIG_PATH.is_file()

# -----------------------------------------------------------------------------
# Sidebar: mode selector and config note
# -----------------------------------------------------------------------------

with st.sidebar:
    st.title("QA Chat")
    mode_options = ["Fine-tuned model", "RAG only", "Hybrid RAG"]
    mode_map = {"Fine-tuned model": "finetuned", "RAG only": "rag", "Hybrid RAG": "hybrid"}
    mode_index = {"finetuned": 0, "rag": 1, "hybrid": 2}.get(st.session_state.current_mode, 0)
    mode_label = st.radio("Answer from", mode_options, index=mode_index, key="mode_radio")
    st.session_state.current_mode = mode_map[mode_label]

    if not config_exists:
        st.caption("Using defaults. Add GUI/gui_config.env to set model paths.")
    else:
        st.caption(f"Config: {CONFIG_PATH.name}")

    st.divider()
    st.caption("Each mode has its own conversation. Switch mode to see another thread.")

# -----------------------------------------------------------------------------
# Current history and display
# -----------------------------------------------------------------------------

history_key = f"history_{st.session_state.current_mode}"
messages = getattr(st.session_state, history_key)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------------------
# Chat input and send
# -----------------------------------------------------------------------------

def get_backend_finetuned():
    if st.session_state.backend_finetuned is not None:
        return st.session_state.backend_finetuned, None
    base = cfg.get("BASE_MODEL", "").strip()
    peft = cfg.get("PEFT_MODEL_PATH", "").strip()
    if not peft or not base:
        return None, "Set BASE_MODEL and PEFT_MODEL_PATH in gui_config.env."
    try:
        from src.inference.inference import QAInference
        backend = QAInference(base_model_name=base, peft_model_path=peft)
        st.session_state.backend_finetuned = backend
        return backend, None
    except Exception as e:
        return None, f"Could not load fine-tuned model: {e}"


def get_backend_rag():
    if st.session_state.backend_rag is not None:
        return st.session_state.backend_rag, None
    db_path = cfg.get("RAG_DB_PATH", "").strip()
    if not db_path:
        return None, "Set RAG_DB_PATH in gui_config.env."
    try:
        from src.rag.pipeline import RAGPipeline
        model_name = cfg.get("RAG_MODEL_NAME", "").strip() or None
        top_k = cfg.get("RAG_TOP_K", "5")
        try:
            top_k = int(top_k)
        except ValueError:
            top_k = 5
        backend = RAGPipeline(db_path=db_path, model_name=model_name, top_k=top_k)
        st.session_state.backend_rag = backend
        return backend, None
    except Exception as e:
        return None, f"Could not load RAG: {e}"


def get_backend_hybrid():
    if st.session_state.backend_hybrid is not None:
        return st.session_state.backend_hybrid, None
    db_path = cfg.get("HYBRID_RAG_DB_PATH", "").strip()
    peft = cfg.get("PEFT_MODEL_PATH", "").strip()
    base = cfg.get("BASE_MODEL", "").strip()
    if not db_path or not peft or not base:
        return None, "Set HYBRID_RAG_DB_PATH, PEFT_MODEL_PATH, and BASE_MODEL in gui_config.env."
    try:
        from src.Hybrid_RAG.pipeline import HybridRAGPipeline
        small_llm = cfg.get("HYBRID_SMALL_LLM", "").strip() or None
        backend = HybridRAGPipeline(
            db_path=db_path,
            peft_model_path=peft,
            base_model_name=base,
            small_llm_name=small_llm,
        )
        st.session_state.backend_hybrid = backend
        return backend, None
    except Exception as e:
        return None, f"Could not load Hybrid RAG: {e}"


if prompt := st.chat_input("Ask a question..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    mode = st.session_state.current_mode
    backend, err = None, None
    if mode == "finetuned":
        backend, err = get_backend_finetuned()
    elif mode == "rag":
        backend, err = get_backend_rag()
    else:
        backend, err = get_backend_hybrid()

    if err:
        reply = f"Configuration or load error: {err}"
    else:
        try:
            temp = float(cfg.get("TEMPERATURE", "0.7") or "0.7")
            max_tok = int(cfg.get("MAX_NEW_TOKENS", "256") or "256")
            if mode == "finetuned":
                reply = backend.generate_answer(
                    prompt, use_history=True, temperature=temp, max_new_tokens=max_tok
                )
            elif mode == "rag":
                reply = backend.answer(prompt)
            else:
                reply = backend.answer(prompt)
        except Exception as e:
            reply = f"Error generating reply: {e}"

    messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
