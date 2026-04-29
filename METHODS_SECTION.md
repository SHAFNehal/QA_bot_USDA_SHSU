# METHODS

## B. Text Processing and Segmentation

Source documents in `.docx`, `.txt`, and `.md` formats were processed using a multi-step pipeline to prepare training data. Document reading was implemented using the `python-docx` library with a fallback to `docx2txt` for compatibility. Text encoding was handled with UTF-8, with automatic fallback to Latin-1 for legacy documents.

Raw text underwent normalization through regular expression-based cleaning to standardize whitespace and remove non-textual artifacts while preserving punctuation essential for semantic comprehension. The cleaned text was then segmented into overlapping chunks using a sliding window approach with the following parameters:

- Chunk size: 1,000 characters
- Overlap: 200 characters
- Sentence boundary alignment: enabled

The segmentation algorithm prioritized sentence boundaries by scanning backwards up to 100 characters from the target endpoint to locate terminal punctuation marks (`.`, `!`, `?`), ensuring semantic coherence within chunks. This approach resulted in an average of 3-5 chunks per document, depending on document length.

## C. Question-Answer Pair Generation

Question-answer pairs were generated from text chunks using a Large Language Model (LLM) generation approach. The pipeline supported configurable model selection, with Mistral-7B-Instruct-v0.2 serving as the default for dataset generation due to its computational efficiency. However, the primary objective was comparative model evaluation rather than reliance on a single architecture.

### Generation Parameters

The LLM was configured with the following hyperparameters to balance diversity and quality:

- Temperature: 0.7
- Top-p (nucleus sampling): 0.9
- Repetition penalty: 1.1
- Max new tokens: 800
- Number of questions per chunk: 3-10 (configurable)

### Prompt Engineering

A structured prompt template was designed to elicit high-quality question-answer pairs:

```
You are an expert at creating high-quality question-answer pairs for training a helpful chatbot.

Based on this text, create exactly {N} question-answer pairs:
[TEXT CHUNK]

Rules:
1. Each question must be complete and end with a question mark
2. Each answer must be complete and informative
3. Questions should be natural and conversational
4. Answers should be helpful and accurate
5. Format as: Q: [complete question] A: [complete answer]
```

### Advanced Mode Enhancements

An advanced generation mode was implemented featuring:

1. **Diverse Prompt Templates**: Six specialized prompt types targeting factual, definitional, reasoning-based, comparison, list-based, and procedural questions to maximize coverage diversity.

2. **Semantic Filtering**: Generated question-answer pairs were evaluated for relevance using the `all-MiniLM-L6-v2` sentence transformer model. Cosine similarity between question embeddings and source chunk embeddings was computed, with pairs below a threshold of 0.5 filtered out.

3. **Deduplication**: A hybrid approach combining O(1) hash-based exact matching and Jaccard similarity-based fuzzy matching (threshold: 0.8) was applied. To maintain computational efficiency, similarity checks were bounded to the last 100 pairs.

### Parsing and Validation

Four parsing strategies with progressively relaxed patterns were employed to extract question-answer pairs from LLM outputs:
1. Pattern: `Q:\s*(.+?)\s*A:\s*(.+?)(?=\s*Q:|$)`
2. Pattern: `\d+\.\s*Q:\s*(.+?)\s*A:\s*(.+?)(?=\s*\d+\.|$)`
3. Pattern: `Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*Question:|$)`
4. Sequential line-based parsing with question mark detection

Validation criteria included:
- Minimum question length: 15 characters
- Minimum answer length: 20 characters
- Questions must terminate with `?`
- Answers must end with sentence-terminating punctuation
- Prohibition of prompt leakage phrases (e.g., "generate", "based on", "format as")
- Prohibition of incomplete sentence endings

### Paraphrase Augmentation

Two augmentation methods were implemented to increase question diversity:

**Rule-Based Augmentation**: Template-based question reformulation using 10 question type categories (what, how, why, when, where, who, which, can, is, are) with 4-10 prefix variations per category. Additionally, 12 imperative/command templates were applied (e.g., "Tell me about X", "Explain X"). This method generated 2-8 variations per question.

**LLM-Based Augmentation**: The generation model produced paraphrases using a structured prompt requesting variations with temperature 0.8 for increased diversity. Numbered list parsing extracted variations, with deduplication against the original question.

## D. Training Dataset Assembly

The training dataset was assembled through a multi-stage pipeline integrating single-turn QA pairs, multi-turn conversations, and conversational interactions.

### Data Cleaning and Validation

A two-mode cleaning system was implemented:

**Strict Mode**: Applied to domain-specific QA pairs with rigorous validation:
- Question length: 10-300 characters
- Answer length: 15-800 characters
- Word count: questions ≥2 words, answers ≥3 words
- Rejection of incomplete endings (e.g., " of", " and", " the")

**Relaxed Mode**: Applied to conversational data allowing:
- Question length: 2-500 characters
- Answer length: 5-1000 characters
- Short-form interactions (greetings, acknowledgments)

Duplicate detection employed the `DuplicateChecker` class with hash-based exact matching (O(1)) and bounded Jaccard similarity computation (last 100 pairs, threshold 0.8).

### Conversational Data Integration

A predefined conversational dataset comprising 99 hand-crafted examples was integrated across eight categories:
- Greetings: 12 examples
- Gratitude: 9 examples
- Farewells: 10 examples
- Acknowledgments: 10 examples
- Clarifications: 8 examples
- Meta-questions: 8 examples
- Affirmatives: 7 examples
- Negatives: 6 examples

To ensure adequate representation, conversational data was replicated 3× in the final training set (configurable multiplier).

### Multi-Turn Conversation Generation

Multi-turn conversational examples were synthesized to train coreference resolution and context maintenance. The system included:

**Predefined Coreference Examples**: Five hand-crafted multi-turn conversations (2-3 turns each) explicitly demonstrating pronoun resolution with topics including machine learning, the sun, photosynthesis, DNA, and Python programming. Each conversation formatted as sequential user-assistant exchanges with full history context.

**Synthetic Multi-Turn Chains**: A configurable fraction (default 20%) of single-turn QA pairs were converted to multi-turn conversations by appending 2 follow-up turns using template-based questions from five categories:
- Clarification: "Can you explain that more?"
- Detail: "What are the main steps?"
- Example: "Can you give me an example?"
- Pronoun-based: "How efficient is it?"
- Continuation: "What else should I know?"

Each multi-turn example was formatted with cumulative history using the template:
```
<|user|>\n{turn1_user}</s>\n<|assistant|>\n{turn1_assistant}</s>\n<|user|>\n{turn2_user}</s>\n<|assistant|>\n
```

### Final Dataset Composition

The complete training dataset was assembled by merging:
1. Cleaned single-turn QA pairs (question/answer format)
2. Multi-turn conversation examples (input/output format with history)
3. Conversational data (3× multiplier)

The dataset was shuffled with seed 42 for reproducibility and saved in JSONL format with the following structure:
```json
{"question": "...", "answer": "...", "type": "qa|conversational|multiturn"}
```

## E. Model Adaptation and Fine-Tuning

### Model Selection and Comparative Evaluation

To evaluate the effectiveness of domain-specific fine-tuning across different model architectures and parameter scales, eleven instruction-tuned language models were systematically trained and compared:

**Small-Scale Models (7-8B parameters)**:
- Mistral-7B-Instruct-v0.2
- Llama 3.1-8B-Instruct
- Qwen 2.5-7B-Instruct
- GLM-4-7B-Flash

**Medium-Scale Models (12-14B parameters)**:
- Mistral-12B-Instruct
- Llama 2-13B-Instruct
- Qwen 2.5-14B-Instruct
- Nemo-12B-Instruct

**Large-Scale Models (70B+ parameters)**:
- Llama 3.1-70B-Instruct
- Qwen 2.5-72B-Instruct
- Mixtral-8x22B-Instruct-v0.1

Each model was fine-tuned independently on the same training dataset using identical hyperparameters (where architecturally feasible) to enable direct performance comparison across parameter scales and architectural designs. Training was conducted via dedicated SLURM job scripts for each model variant, with separate output directories (`fine_tuned_weights_{model}_run1`) to facilitate systematic evaluation.

### Low-Rank Adaptation (LoRA) Configuration

Parameter-efficient fine-tuning was implemented using LoRA (Low-Rank Adaptation) with the following hyperparameters:

- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Bias: none
- Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

Target modules were automatically detected from the model architecture, with a fallback mechanism for compatibility across model families. LoRA adaptation resulted in approximately 1-2% trainable parameters relative to the full model.

### Training Configuration

The training procedure employed the following settings:

**Optimization Parameters**:
- Learning rate: 1×10⁻⁴ to 5×10⁻⁴
- Learning rate scheduler: Cosine annealing with warmup
- Warmup steps: 100
- Optimizer: AdamW (default from Transformers Trainer)
- Weight decay: default (0.0)

**Batch Configuration**:
- Per-device batch size: 1-2
- Gradient accumulation steps: 4-16 (effective batch size: 4-32)
- Maximum sequence length: 2048 tokens

**Training Duration**:
- Epochs: 3-10
- Evaluation strategy: every 200 steps
- Save strategy: every 200 steps (best model retention based on validation loss)
- Early stopping patience: 3 evaluations

**Memory Optimization**:
- Mixed precision: FP16 (CUDA) or BF16 (if supported)
- Gradient checkpointing: enabled
- Quantization support: 4-bit and 8-bit via BitsAndBytes

### Data Formatting and Loss Masking

Training examples were formatted using chat templates consistent with the model family:

**TinyLlama/Legacy Format**:
```
<|system|>\n{SYSTEM_PROMPT}</s>\n<|user|>\n{QUESTION}</s>\n<|assistant|>\n{ANSWER}</s>
```

**Llama 2 Format** (when `chat_format="llama2"`):
```
<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{USER_MESSAGE} [/INST] {ASSISTANT_RESPONSE} </s>
```

**Llama 3.1 Format** (when `chat_format="auto"` with chat template):
Applied `tokenizer.apply_chat_template()` with role-based message formatting.

**Response-Only Loss Masking**: To prevent the model from learning to predict user prompts, a custom `DataCollatorForCompletionOnly` was implemented. Loss computation was restricted to tokens following the assistant response marker (`<|assistant|>` or equivalent), with prompt tokens masked using the sentinel value -100. The collator employed robust template matching across multiple whitespace/newline variants to account for tokenizer sensitivity.

For models with native chat templates (e.g., Llama 3.1), a `DataCollatorForCausalLMWithLabels` was used with pre-computed label masks, where prompt tokens were masked during dataset preparation using:
```python
prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
full_ids = tokenizer.apply_chat_template(messages + [assistant_message])
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

### Training and Validation Split

The dataset was randomly partitioned into training (90%) and validation (10%) sets with seed 42. Validation loss was monitored at each evaluation step, with the best model checkpoint (lowest validation loss) saved for deployment. Training terminated early if validation loss failed to improve for 3 consecutive evaluations, preventing overfitting.

### System Prompt

A unified system prompt was employed across all training examples and inference:
```
You are a helpful, friendly assistant. You can answer questions, engage in conversation, and assist users with their needs.
```

This prompt was prepended to all interactions to establish consistent behavioral expectations.

### Hardware and Software Environment

Training was configured for SLURM-based HPC clusters with the following specifications:
- GPU: NVIDIA A100 (40GB/80GB) or equivalent
- Framework: PyTorch 2.x with Transformers 4.x
- PEFT library: 0.x for LoRA implementation
- Precision: Mixed precision (FP16/BF16) with automatic device detection

The training pipeline supported both single-GPU and distributed training configurations, with automatic device mapping for quantized models.
