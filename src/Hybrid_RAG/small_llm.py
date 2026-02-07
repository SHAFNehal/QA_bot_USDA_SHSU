"""
Shared small LLM for restater and synthesizer. Load once, run_prompt(prompt) -> str.
"""

from typing import Optional

import torch

from src.utils.model_utils import load_model_and_tokenizer, format_chat_prompt

SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instruction precisely."
)


class SmallLLM:
    """Single small LLaMA used for restating questions and synthesizing final answers."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
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

    def run_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Run one turn: format as chat, generate, return assistant reply only.
        """
        self._ensure_loaded()
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        prompt = format_chat_prompt(
            user_message,
            system_prompt=system_prompt or SYSTEM_PROMPT,
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
                max_new_tokens=max_tokens,
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
