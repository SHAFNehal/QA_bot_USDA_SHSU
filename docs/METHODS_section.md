# Methods Section (Draft)

**Paper:** A Structured Insect Knowledge Base and Lightweight LLM Framework for Agricultural Decision Support  
**For use in:** Materials and Methods (or Methods)

---

## 3. Materials and Methods

### 3.1. Knowledge Base and Data Sources

The knowledge base used in this study was built from expert-written reports prepared for agricultural pest species. We collected nine documents in total, each focusing on a single pest or related organism. The documents cover species such as the Asian citrus psyllid, citrus longhorned beetle, cotton cutworm, emerald ash borer, horse thistle, *Ips sexdentatus*, large pine weevil, red palm weevil, red ring nematode, and redheaded pine sawfly. The source material draws on expert-organized information from USDA and similar agricultural extension resources, and is intended to support farmer-oriented questions on identification, biology, damage, and management. The corpus is stored in portable document format (.docx), with support for plain text (.txt) and Markdown (.md) for future expansion.

Each document was read programmatically (UTF-8 encoding, with fallback to Latin-1 where needed) and normalized by removing extra whitespace and non-essential characters while preserving sentence boundaries. The text was then split into overlapping segments (chunks) of 1000 characters with 200 characters overlap so that continuity between adjacent segments was preserved. Chunk boundaries were aligned to sentence endings (., !, ?) where possible to avoid splitting mid-sentence. This yielded a sequence of text segments, each associated with source file and segment index, which served as the input to question–answer generation.

### 3.2. Question–Answer Pair Generation

Because manually creating a large question–answer (QA) set from the expert reports would be costly, we generated QA pairs automatically using a language model under controlled prompting. For each text chunk, a configurable number of question–answer pairs (e.g., up to 10 per chunk) were produced via a single prompt that asked the model to create natural, self-contained questions and informative answers from the given passage. Generation used sampling with a temperature of 0.7, top-p of 0.9, and repetition penalty of 1.1, with a cap on new tokens (e.g., 800) to keep answers concise. The prompt specified output format (e.g., “Q: … A: …”) and quality rules (complete sentences, question mark on questions, no meta-instructions in the answers). The same base model used later for fine-tuning (or a model of the same family) was used for generation to keep style and terminology consistent.

The raw model output was parsed with several strategies (e.g., “Q: … A: …”, numbered items, “Question: … Answer: …”) and validated. Pairs were discarded if the question or answer failed length bounds (e.g., minimum 15 characters for questions and 20 for answers in strict mode), lacked a closing question mark or sentence-ending punctuation, contained prompt-leakage phrases, or repeated the question in the answer. Optionally, the dataset was augmented with paraphrased questions: either rule-based variants (e.g., rephrasing “What is X?” as “Can you explain X?” or “Tell me about X”) or additional LLM-generated phrasings, so that the model would see multiple ways of asking the same factual content. Duplicate or near-duplicate pairs were removed using hash-based exact match and Jaccard similarity over word sets (threshold 0.8) to keep the training set diverse and non-redundant.

### 3.3. Training Dataset Assembly

The fine-tuning dataset combined three types of examples. First, the cleaned single-turn QA pairs from the pest documents formed the core domain content. Second, a fixed set of conversational pairs was added so that the system would respond appropriately to greetings (e.g., “Hi”, “Good morning”), thanks (“Thank you”, “Thanks”), farewells (“Bye”, “See you”), acknowledgments (“Okay”, “Got it”), and meta-questions (“Who are you?”, “What can you do?”). These were given relaxed validation (e.g., shorter answers allowed) and labeled as conversational so they could be weighted or repeated in the merge step; the conversational subset was repeated by a configurable multiplier (e.g., 3×) to improve coverage. Third, multi-turn examples were included to improve follow-up and pronoun resolution. A fraction of the single-turn QA pairs (e.g., 20%) was sampled and converted into short dialogue chains by appending a fixed number of follow-up turns (e.g., 2 per chain) with generic follow-up questions (e.g., “Can you explain that more?”, “Why is it important?”) and corresponding assistant replies. In addition, a small set of hand-crafted multi-turn conversations with explicit coreference (e.g., “What is X?” → “How does it work?”) was added to encourage the model to resolve “it” and “this” in context. The three components were merged with configurable repetition factors for conversational and multi-turn data, then shuffled with a fixed random seed (e.g., 42) to form the final training set used for supervised fine-tuning.

### 3.4. Model Adaptation

We adapted lightweight, open-source language models (e.g., LLaMA 3.2, TinyLlama, Mistral 7B Instruct) to the pest-management QA task using low-rank adaptation (LoRA). LoRA injects trainable low-rank matrices into selected linear layers of the frozen base model (e.g., query, value, key, and output projections and, where applicable, gate, up, and down projections in MLP blocks), with rank r = 16, LoRA alpha = 32, and dropout 0.05, so that only a small fraction of parameters is updated while preserving most of the pre-trained knowledge. This keeps training and storage costs low and allows the same base model to be reused for multiple specialized adapters.

Training was conducted in a chat format consistent with the base model (e.g., system prompt, user and assistant turns, special tokens such as <|user|>, <|assistant|>, and end-of-turn markers). Only the assistant response tokens were used for the loss; user and system tokens were masked (labels set to −100) so that the model learned to generate answers without being trained to predict the questions. The dataset was split into train and validation subsets (90% / 10%). We used a cosine learning-rate schedule with warmup (e.g., 100 steps), mixed-precision (FP16) where available, and gradient checkpointing to reduce memory use. Training used a per-device batch size of 1 with gradient accumulation (e.g., 16 steps) to obtain an effective batch size of 16, a learning rate on the order of 1×10⁻⁴ to 5×10⁻⁴, and a maximum sequence length of 1024–2048 tokens depending on the base model. Early stopping was applied when validation loss did not improve for three consecutive evaluation steps. The resulting LoRA weights can be loaded on top of the base model for inference. The system is designed to run locally (on a single machine or edge device), so that it can operate in low-connectivity or offline settings where farmers may not have reliable internet access.

### 3.5. Evaluation and Comparative Experiments

Comparative experiments against baseline and fine-tuned configurations, along with detailed evaluation metrics and results, are presented in Section 4 (Results).

---

*End of Methods section*
