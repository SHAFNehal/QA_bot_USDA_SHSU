"""
RAG generator: load LLM from config (model_utils), generate answer given context + question.
Uses same chat template as rest of project.
"""

from typing import Optional

import torch

from src.utils.model_utils import (
    load_model_and_tokenizer,
    format_chat_prompt,
    get_device,
)

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use only the context below to answer. If the context does not contain enough information, say so."
)


class RAGGenerator:
    """Generate answers using a pretrained LLM with retrieved context."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.model_name,
            device=self._device,
            for_training=False,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(self, context: str, question: str) -> str:
        """
        Generate answer given context (concatenated retrieved chunks) and question.
        Returns only the assistant reply text.
        """
        self._ensure_loaded()
        user_message = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question: {question.strip()}"
        )
        prompt = format_chat_prompt(
            user_message,
            system_prompt=self.system_prompt,
            conversation_history=None,
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|assistant|>" in response:
            answer = response.split("<|assistant|>")[-1].strip()
        else:
            answer = response[len(prompt):].strip()
        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        return answer
