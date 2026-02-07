"""
Restater: small LLM restates the question in 5+ different ways (same meaning).
"""

import re
from typing import List

RESTATE_PROMPT_TEMPLATE = (
    "Restate the following question in exactly 5 different ways without changing its meaning. "
    "Output one question per line. Do not number them. Do not add any other text.\n\nQuestion: {question}"
)


def _parse_restatements(raw: str, min_count: int = 5) -> List[str]:
    """Parse model output into list of restatements. Tolerate numbered lines and bullets."""
    lines = raw.strip().split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading number/bullet (e.g. "1.", "1)", "- ", "* ")
        line = re.sub(r"^\s*[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s+", "", line)
        if line and not line.lower().startswith("question:"):
            out.append(line)
    return out


class Restater:
    """Uses shared SmallLLM to produce 5+ restatements of a question."""

    def __init__(self, small_llm, num_restatements: int = 5, max_new_tokens: int = 256):
        self.small_llm = small_llm
        self.num_restatements = num_restatements
        self.max_new_tokens = max_new_tokens

    def restate(self, question: str) -> List[str]:
        """
        Return at least num_restatements different phrasings of the question.
        If the model returns fewer, the list is padded with the original question.
        """
        user_message = RESTATE_PROMPT_TEMPLATE.format(question=question.strip())
        raw = self.small_llm.run_prompt(user_message, max_new_tokens=self.max_new_tokens)
        restatements = _parse_restatements(raw, min_count=self.num_restatements)

        # Ensure we have at least num_restatements (pad with original if needed)
        while len(restatements) < self.num_restatements:
            restatements.append(question.strip())
        return restatements[: self.num_restatements]
