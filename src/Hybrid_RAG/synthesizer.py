"""
Synthesizer: small LLM produces one final answer from RAG answer + 6 FT answers.
"""

from typing import List

SYNTHESIZE_PROMPT_TEMPLATE = """You are given one question and several candidate answers from two sources: (1) a retrieval-augmented system (RAG) and (2) a fine-tuned model asked the same question in different phrasings.

Your task: Produce one correct, coherent final answer based on these candidates. Prefer agreement across sources; if they conflict, choose the most accurate or combine the best parts. Do not add meta-commentary; output only the final answer.

Question: {question}

RAG answer:
{rag_answer}

Fine-tuned model answers (same question, different phrasings):
{ft_answers_block}

Final answer:"""


class Synthesizer:
    """Uses shared SmallLLM to synthesize one answer from RAG + FT candidates."""

    def __init__(self, small_llm, max_new_tokens: int = 512):
        self.small_llm = small_llm
        self.max_new_tokens = max_new_tokens

    def synthesize(self, question: str, rag_answer: str, ft_answers: List[str]) -> str:
        """Produce one final answer from the original question, 1 RAG answer, and 6 FT answers."""
        ft_block = "\n\n".join(f"({i+1}) {a.strip()}" for i, a in enumerate(ft_answers) if a and a.strip())
        if not ft_block:
            ft_block = "(No fine-tuned answers provided.)"
        user_message = SYNTHESIZE_PROMPT_TEMPLATE.format(
            question=question.strip(),
            rag_answer=rag_answer.strip() if rag_answer else "(No RAG answer.)",
            ft_answers_block=ft_block,
        )
        return self.small_llm.run_prompt(user_message, max_new_tokens=self.max_new_tokens)