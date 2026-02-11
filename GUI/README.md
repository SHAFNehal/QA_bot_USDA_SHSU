# QA Chat GUI

Interactive **Streamlit** chat interface for the USDA Insect QA Bot with **three modes**:

1. **Fine-tuned Model** — Fast domain-specific answers (1-2s)
2. **RAG** — Document-grounded answers with sources (2-4s)
3. **Hybrid RAG** — Combined fine-tuned + RAG for best quality (5-8s)

Each mode maintains its own conversation history with a clean, muted design.

---

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have at least one system ready:
# Option A: Fine-tuned model
./scripts/run_pipeline.sh

# Option B: RAG database
python src/rag/ingest.py --input_dir data_input --db_path rag_db

# Option C: Run full pipeline (all three systems)
./scripts/run_full_pipeline_with_rag.sh
```

### 2. Launch GUI

```bash
# From project root
streamlit run GUI/app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Select Mode

Use the **sidebar** to choose:
- **Fine-tuned Model** (fast, domain-specific)
- **RAG** (document-based, updatable)
- **Hybrid RAG** (best quality, slower)

---

## Features

### Three Query Modes

| Mode | Speed | Accuracy | Updatable | Best For |
|------|-------|----------|-----------|----------|
| **Fine-tuned** | 1-2s | 70-80% | No | Quick domain questions |
| **RAG** | 2-4s | 70-85% | Yes | Document lookups |
| **Hybrid RAG** | 5-8s | Best (75-90%) | Yes | Critical/complex questions |

### Interactive Features

- **Separate Conversations**: Each mode has independent chat history
- **Source Citations**: RAG and Hybrid modes show document sources
- **Clear History**: Reset conversation with sidebar button
- **Session Persistence**: Chat history persists during browser session
- **Real-time Responses**: Streaming text generation
- **Muted Theme**: Professional, distraction-free design

### Sidebar Controls

- **Mode Selection**: Switch between three systems
- **Clear History**: Reset current mode's conversation
- **Configuration Info**: Shows active settings
- **Status Indicators**: Model/database loading status

---

## Configuration

### Option 1: Configuration File (Recommended)

Edit `GUI/gui_config.env` in project root:

```bash
# Fine-tuned Model Settings
BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
PEFT_MODEL_PATH=data_output/final_model

# RAG Settings
RAG_DB_PATH=rag_db
RAG_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Hybrid RAG Settings
HYBRID_RAG_DB_PATH=rag_db
HYBRID_SMALL_LLM=meta-llama/Llama-3.2-1B-Instruct

# Generation Parameters
TEMPERATURE=0.7
MAX_NEW_TOKENS=256
RAG_TOP_K=5
```

**Parameters Explained:**

- `BASE_MODEL`: HuggingFace model name (base model before fine-tuning)
- `PEFT_MODEL_PATH`: Path to fine-tuned LoRA weights
- `RAG_DB_PATH`: Path to Chroma vector database
- `RAG_MODEL_NAME`: Model for RAG generation
- `HYBRID_RAG_DB_PATH`: Same as RAG_DB_PATH (for hybrid mode)
- `HYBRID_SMALL_LLM`: Model for answer synthesis in hybrid mode
- `TEMPERATURE`: Sampling temperature (0.0-1.0, higher = more random)
- `MAX_NEW_TOKENS`: Maximum response length
- `RAG_TOP_K`: Number of documents to retrieve (3-10 recommended)

### Option 2: Default Configuration

If `gui_config.env` is missing, the app uses built-in defaults:

```python
{
    "BASE_MODEL": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "PEFT_MODEL_PATH": "data_output/final_model",
    "RAG_DB_PATH": "rag_db",
    "TEMPERATURE": 0.7,
    "MAX_NEW_TOKENS": 256,
    "RAG_TOP_K": 5
}
```

The sidebar will show a note if using defaults.

---

## File Structure

```
GUI/
├── app.py              # Main Streamlit application
├── gui_config.env      # Configuration file (create manually)
└── README.md           # This file

.streamlit/             # Auto-created on first run
└── config.toml         # Streamlit theme (muted colors)
```

---

## Usage Examples

### Example 1: Quick Question with Fine-tuned Model

1. Launch GUI: `streamlit run GUI/app.py`
2. Select **"Fine-tuned Model"** in sidebar
3. Ask: "What is integrated pest management?"
4. Get fast answer in 1-2 seconds

### Example 2: Document-Based Query with RAG

1. Select **"RAG"** in sidebar
2. Ask: "What does the IPM guide say about aphid control?"
3. Get answer with source citations:
   ```
   Answer: ...biological control with ladybugs...
   
   Sources:
   - ipm_guide.pdf (page 42)
   - pest_control.md
   ```

### Example 3: Complex Query with Hybrid RAG

1. Select **"Hybrid RAG"** in sidebar
2. Ask: "Compare chemical and biological control for corn pests"
3. Get comprehensive answer combining:
   - Fine-tuned model's domain knowledge
   - RAG's document references
   - Synthesized best answer
4. View both component answers in expandable sections

---

## Troubleshooting

### Common Issues

**1. "Model not found" error**

```bash
# Train model first
./scripts/run_pipeline.sh

# Or update PEFT_MODEL_PATH in gui_config.env
```

**2. "RAG database not found" error**

```bash
# Create RAG database
python src/rag/ingest.py --input_dir data_input --db_path rag_db

# Or update RAG_DB_PATH in gui_config.env
```

**3. "Address already in use" error**

```bash
# Stop existing Streamlit instance
streamlit stop

# Or use different port
streamlit run GUI/app.py --server.port 8502
```

**4. Slow responses**

```bash
# Check mode: Hybrid is slower (5-8s expected)
# Switch to Fine-tuned or RAG for faster responses

# Enable GPU if available
export CUDA_VISIBLE_DEVICES=0
streamlit run GUI/app.py
```

**5. Out of memory**

```bash
# Reduce MAX_NEW_TOKENS in gui_config.env
MAX_NEW_TOKENS=128

# Or use smaller model
BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**6. No responses in RAG mode**

```bash
# Check if documents exist in database
python -c "
from chromadb import PersistentClient
client = PersistentClient(path='rag_db')
print(client.list_collections())
"

# Re-ingest if needed
python src/rag/ingest.py --input_dir data_input --db_path rag_db
```

---

## Advanced Configuration

### Custom Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#6c757d"      # Muted gray
backgroundColor = "#f8f9fa"   # Light background
secondaryBackgroundColor = "#e9ecef"
textColor = "#212529"
font = "sans serif"
```

### Remote Deployment

```bash
# Install streamlit cloud CLI
pip install streamlit

# Deploy to Streamlit Cloud
streamlit deploy GUI/app.py
```

### Environment Variables

Override config without editing files:

```bash
export PEFT_MODEL_PATH="custom_model"
export RAG_DB_PATH="custom_rag_db"
streamlit run GUI/app.py
```

---

## API Integration

The GUI uses these backend modules:

### Fine-tuned Mode

```python
from src.inference.inference import FineTunedInference

inference = FineTunedInference(
    base_model=config.BASE_MODEL,
    peft_model=config.PEFT_MODEL_PATH
)
response = inference.generate(question)
```

### RAG Mode

```python
from src.rag.query import RAGQuery

rag = RAGQuery(
    db_path=config.RAG_DB_PATH,
    top_k=config.RAG_TOP_K
)
response, sources = rag.query(question)
```

### Hybrid RAG Mode

```python
from src.Hybrid_RAG.query import HybridRAGQuery

hybrid = HybridRAGQuery(
    model_dir=config.PEFT_MODEL_PATH,
    db_path=config.HYBRID_RAG_DB_PATH
)
result = hybrid.query(question)
# result contains: fine_tuned_answer, rag_answer, merged_answer, sources
```

---

## Development

### Running with Debug Mode

```bash
# Enable verbose logging
streamlit run GUI/app.py --logger.level=debug
```

### Hot Reload

Streamlit automatically reloads when you edit `app.py`. Just save and refresh browser.

### Adding New Features

1. Edit `GUI/app.py`
2. Add new configuration to `GUI/gui_config.env`
3. Update this README
4. Test all three modes

---

## Best Practices

1. **Mode Selection**: Choose based on needs
   - Time-sensitive? → Fine-tuned
   - Need sources? → RAG or Hybrid
   - Critical accuracy? → Hybrid

2. **Configuration**: Test different parameters
   - Start with defaults
   - Adjust TEMPERATURE for creativity
   - Tune RAG_TOP_K for relevance/speed

3. **Document Management**: 
   - Update documents? Re-ingest RAG database
   - Keep RAG_DB_PATH synced with actual data

4. **Performance**:
   - Use GPU for faster responses
   - Close other browser tabs
   - Monitor memory usage with Hybrid mode

5. **User Experience**:
   - Clear history for new topics
   - Compare modes for same question
   - Check sources in RAG/Hybrid

---

## Keyboard Shortcuts

- `Ctrl+K`: Focus on input box
- `Ctrl+Enter`: Send message
- `Ctrl+Shift+C`: Clear history (when button focused)
- `Ctrl+R`: Reload page
- `Ctrl+Shift+R`: Hard reload (clear cache)

---

## Architecture

```
User Input
    ↓
Streamlit Interface (app.py)
    ↓
Mode Router
    ├─→ Fine-tuned Model (inference.py)
    ├─→ RAG (query.py)
    └─→ Hybrid RAG (Hybrid_RAG/query.py)
    ↓
Response + Sources
    ↓
Display in Chat
```
