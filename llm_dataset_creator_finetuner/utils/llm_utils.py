import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Any, Optional
import json
import re


def load_model_and_tokenizer(model_name: str, device: str = "auto") -> tuple:
    """Load model and tokenizer with appropriate settings."""
    print(f"Loading model: {model_name}")
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device set to use {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        model = model.to("cpu")
    
    return model, tokenizer


def create_qa_prompt(text_chunk: str, num_questions: int = 3) -> str:
    """Create a focused prompt for generating high-quality QA pairs for chatbot training."""
    
    # Clean and prepare the text chunk
    text_chunk = text_chunk.strip()
    if len(text_chunk) > 2000:
        text_chunk = text_chunk[:2000] + "..."
    
    prompt = f"""<|system|>
You are an expert at creating high-quality question-answer pairs for training a helpful chatbot. Your goal is to create natural, complete questions and comprehensive answers that a user might ask about this information.

<|user|>
Based on this text, create exactly {num_questions} question-answer pairs that would be useful for a chatbot to know:

{text_chunk}

Rules:
1. Each question must be complete and end with a question mark
2. Each answer must be complete and informative
3. Questions should be natural and conversational
4. Answers should be helpful and accurate
5. Format as: Q: [complete question] A: [complete answer]

<|assistant|>
"""
    return prompt


def generate_qa_pairs(model, tokenizer, text_chunk: str, num_questions: int = 3, 
                     max_length: int = 1024, temperature: float = 0.7) -> List[Dict[str, str]]:
    """Generate high-quality QA pairs from a text chunk using the LLM."""
    prompt = create_qa_prompt(text_chunk, num_questions)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Generate response with robust error handling
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,  # Increased for more complete answers
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1  # Prevent repetitive text
            )
    except Exception as e:
        print(f"Warning: Generation failed with error: {e}")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return []
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    generated_text = response[len(prompt):].strip()
    
    # Parse QA pairs from the generated text
    return parse_qa_pairs_improved(generated_text, num_questions)


def parse_qa_pairs_improved(text: str, expected_count: int = 3) -> List[Dict[str, str]]:
    """Parse QA pairs with improved validation for chatbot training."""
    qa_pairs = []
    
    # Clean the text
    text = text.strip()
    
    # Multiple parsing strategies with better patterns
    strategies = [
        parse_qa_format_improved_1,  # Q: ... A: ... (most common)
        parse_qa_format_improved_2,  # 1. Q: ... A: ...
        parse_qa_format_improved_3,  # Question: ... Answer: ...
        parse_qa_format_improved_4,  # Just questions and answers in sequence
    ]
    
    for strategy in strategies:
        pairs = strategy(text)
        if len(pairs) >= expected_count:
            qa_pairs = pairs[:expected_count]
            break
        elif len(pairs) > len(qa_pairs):
            qa_pairs = pairs
    
    # Enhanced validation and cleaning
    cleaned_pairs = []
    for pair in qa_pairs:
        if validate_qa_pair_improved(pair):
            cleaned_pair = {
                'question': clean_text_content_improved(pair['question']),
                'answer': clean_text_content_improved(pair['answer'])
            }
            # Additional check for completeness
            if is_complete_qa_pair(cleaned_pair):
                cleaned_pairs.append(cleaned_pair)
    
    return cleaned_pairs


def parse_qa_format_improved_1(text: str) -> List[Dict[str, str]]:
    """Parse Q: ... A: ... format with improved regex."""
    pairs = []
    # More robust pattern that handles various spacing and formatting
    pattern = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=\s*Q:|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        # Additional validation
        if question and answer and question.endswith('?'):
            pairs.append({
                'question': question,
                'answer': answer
            })
    
    return pairs


def parse_qa_format_improved_2(text: str) -> List[Dict[str, str]]:
    """Parse numbered format with improved pattern."""
    pairs = []
    pattern = r'\d+\.\s*Q:\s*(.+?)\s*A:\s*(.+?)(?=\s*\d+\.|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        if question and answer and question.endswith('?'):
            pairs.append({
                'question': question,
                'answer': answer
            })
    
    return pairs


def parse_qa_format_improved_3(text: str) -> List[Dict[str, str]]:
    """Parse Question: ... Answer: ... format."""
    pairs = []
    pattern = r'Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*Question:|$)'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        if question and answer and question.endswith('?'):
            pairs.append({
                'question': question,
                'answer': answer
            })
    
    return pairs


def parse_qa_format_improved_4(text: str) -> List[Dict[str, str]]:
    """Parse simple alternating questions and answers with better logic."""
    pairs = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    current_question = None
    current_answer = ""
    
    for line in lines:
        # Check if line looks like a question
        if line.endswith('?') and len(line) > 10:
            if current_question and current_answer:
                pairs.append({
                    'question': current_question,
                    'answer': current_answer.strip()
                })
            current_question = line
            current_answer = ""
        elif current_question:
            current_answer += " " + line
    
    # Add the last pair
    if current_question and current_answer:
        pairs.append({
            'question': current_question,
            'answer': current_answer.strip()
        })
    
    return pairs


def validate_qa_pair_improved(pair: Dict[str, str]) -> bool:
    """Enhanced validation for QA pairs suitable for chatbot training."""
    if not isinstance(pair, dict) or 'question' not in pair or 'answer' not in pair:
        return False
    
    question = pair['question'].strip()
    answer = pair['answer'].strip()
    
    # Basic validation
    if not question or not answer:
        return False
    
    # Length validation - ensure substantial content
    if len(question) < 15 or len(answer) < 20:
        return False
    
    # Question must end with question mark
    if not question.endswith('?'):
        return False
    
    # Check for incomplete sentences
    incomplete_endings = [
        ' of', ' and', ' the', ' a', ' an', ' in', ' on', ' at', ' to', ' for',
        ' with', ' by', ' from', ' that', ' this', ' these', ' those'
    ]
    
    for ending in incomplete_endings:
        if answer.endswith(ending):
            return False
    
    # Check for prompt leakage and formatting artifacts
    forbidden_phrases = [
        'question-answer', 'generate', 'create', 'based on', 'text:', 'rules:',
        'q:', 'a:', 'question:', 'answer:', 'format as', 'complete question',
        'complete answer', 'natural and conversational', 'helpful and accurate'
    ]
    
    text_to_check = (question + ' ' + answer).lower()
    for phrase in forbidden_phrases:
        if phrase in text_to_check:
            return False
    
    # Check for repetitive or nonsensical content
    if question.lower() == answer.lower() or question in answer:
        return False
    
    return True


def clean_text_content_improved(text: str) -> str:
    """Enhanced text cleaning for chatbot training."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove formatting artifacts more thoroughly
    text = re.sub(r'^(Q:|A:|Question:|Answer:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.\s*', '', text)
    
    # Remove common artifacts
    text = re.sub(r'^(Rules:|Format:|Create|Generate)\s*', '', text, flags=re.IGNORECASE)
    
    # Clean up punctuation
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    
    return text.strip()


def is_complete_qa_pair(pair: Dict[str, str]) -> bool:
    """Additional check for completeness of QA pairs."""
    question = pair['question']
    answer = pair['answer']
    
    # Check for balanced sentence structure
    if question.count('?') != 1:
        return False
    
    # Check for reasonable answer length relative to question
    if len(answer) < len(question) * 0.5:
        return False
    
    # Check for proper sentence endings in answer
    if not answer.endswith(('.', '!', '?')):
        return False
    
    # Check for reasonable word count
    question_words = len(question.split())
    answer_words = len(answer.split())
    
    if question_words < 3 or answer_words < 5:
        return False
    
    return True


def create_pipeline(model_name: str, device: str = "auto"):
    """Create a text generation pipeline."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True
    )


def generate_qa_with_pipeline(pipe, text_chunk: str, num_questions: int = 3, 
                            max_length: int = 1024, temperature: float = 0.7) -> List[Dict[str, str]]:
    """Generate QA pairs using a pipeline with improved quality."""
    prompt = create_qa_prompt(text_chunk, num_questions)
    
    try:
        response = pipe(
            prompt,
            max_new_tokens=800,  # Increased for more complete answers
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            repetition_penalty=1.1
        )[0]['generated_text']
    except Exception as e:
        print(f"Warning: Pipeline generation failed, trying alternative method: {e}")
        try:
            response = pipe(
                prompt,
                max_new_tokens=400,
                do_sample=False,
                num_return_sequences=1
            )[0]['generated_text']
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return []
    
    # Extract the generated part
    generated_text = response[len(prompt):].strip()
    
    # Parse QA pairs with improved parsing
    return parse_qa_pairs_improved(generated_text, num_questions)


def get_available_models() -> List[str]:
    """Return a list of recommended small models for QA generation."""
    return [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-3-mini-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]