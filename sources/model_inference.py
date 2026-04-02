"""
Model inference classes for API-based and local models.

Both classes expose a single method:

    infer(prompt_text, num_options=4) -> InferenceResult

where InferenceResult contains:
    predicted_letter : str          – 'A', 'B', 'C', or 'D'
    entropy          : float        – predictive entropy over answer options
    logit_margin     : float        – top-1 minus top-2 logit (or log-prob)
    answer_logits    : dict[str,float] – raw logit / log-prob for each letter

For Llama-3 (local) we extract logits directly from the model's first generated token.
For API models (GPT, Llama, Qwen via OpenRouter) we use the `logprobs` / `top_logprobs`
fields from the chat completion. Llama/GPT use top_logprobs=20; Qwen uses 5
(Alibaba provider hard-limit). The value does not affect returned logprob
magnitudes, only how many tokens are listed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

LETTERS = ["A", "B", "C", "D"]


@dataclass
class InferenceResult:
    predicted_letter: str
    entropy: float
    logit_margin: float
    answer_logits: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(values: List[float]) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    v -= v.max()  # numerical stability
    e = np.exp(v)
    return e / e.sum()


def _entropy_from_logits(logits_dict: Dict[str, float], letters: List[str]) -> float:
    logit_vals = [logits_dict[l] for l in letters]
    probs = _softmax(logit_vals)
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def _logit_margin(logits_dict: Dict[str, float], letters: List[str]) -> float:
    vals = sorted([logits_dict[l] for l in letters], reverse=True)
    return float(vals[0] - vals[1]) if len(vals) >= 2 else 0.0


def _predicted_letter(logits_dict: Dict[str, float], letters: List[str]) -> str:
    return max(letters, key=lambda l: logits_dict[l])


# ---------------------------------------------------------------------------
# Llama-3 (HuggingFace Transformers)
# ---------------------------------------------------------------------------

class LlamaModel:
    """
    Wraps a Llama-3 Instruct model loaded via HuggingFace Transformers.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.
        e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    load_in_8bit : bool
        Load in 8-bit quantisation (requires bitsandbytes).
    load_in_4bit : bool
        Load in 4-bit quantisation (requires bitsandbytes).
    """

    def __init__(
        self,
        model_name_or_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_kwargs: dict = {}
        if load_in_8bit:
            quantization_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            quantization_kwargs["load_in_4bit"] = True
        else:
            quantization_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            **quantization_kwargs,
        )
        self.model.eval()

        # Cache token IDs for " A", " B", " C", " D" (with leading space,
        # as generated after the "Answer:" suffix)
        self._answer_token_ids = self._build_answer_token_ids()

    def _build_answer_token_ids(self) -> Dict[str, List[int]]:
        """
        For each answer letter, collect all single-token representations
        (with/without space prefix). During inference we take the max logit
        across all candidate IDs for robustness.
        """
        ids: Dict[str, List[int]] = {}
        for letter in LETTERS:
            candidates = set()
            for prefix in ("", " ", "\n"):
                encoded = self.tokenizer.encode(
                    prefix + letter, add_special_tokens=False
                )
                if len(encoded) == 1:
                    candidates.add(encoded[0])
            ids[letter] = list(candidates) if candidates else [
                self.tokenizer.encode(letter, add_special_tokens=False)[0]
            ]
        return ids

    def infer(self, prompt_text: str, num_options: int = 4) -> InferenceResult:
        import torch

        active_letters = LETTERS[:num_options]

        messages = [{"role": "user", "content": prompt_text}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        # scores[0]: logits for the 1st generated token, shape [1, vocab_size]
        logits = outputs.scores[0][0]  # [vocab_size]

        # For each answer letter take the maximum logit among candidate token IDs
        answer_logits: Dict[str, float] = {}
        for letter in active_letters:
            token_ids = self._answer_token_ids[letter]
            answer_logits[letter] = max(logits[tid].item() for tid in token_ids)

        entropy = _entropy_from_logits(answer_logits, active_letters)
        margin = _logit_margin(answer_logits, active_letters)
        pred = _predicted_letter(answer_logits, active_letters)

        return InferenceResult(
            predicted_letter=pred,
            entropy=entropy,
            logit_margin=margin,
            answer_logits=answer_logits,
        )


# ---------------------------------------------------------------------------
# OpenAI GPT (API)
# ---------------------------------------------------------------------------

class OpenAIModel:
    """
    Wraps the OpenAI Chat Completions API (also used for OpenRouter).

    For uncertainty estimation we request top_logprobs on the first generated
    token. Llama/GPT use top_logprobs=20; Qwen uses 5 (Alibaba provider
    hard-limit). If a letter is not in the top-N, we assign a very low
    log-prob (LOG_PROB_FLOOR) as a conservative estimate.

    For Qwen models, thinking mode is automatically disabled by prepending
    "/no_think\n" to the user message so that the first output token is the
    answer letter (not a thinking token).
    """

    LOG_PROB_FLOOR = -20.0  # stand-in for letters missing from top_logprobs

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ):
        import openai as _openai
        import os

        client_kwargs = {"max_retries": max_retries}
        resolved_key = api_key or (
            os.environ.get("OPENROUTER_API_KEY") if base_url else None
        )
        if resolved_key:
            client_kwargs["api_key"] = resolved_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = _openai.OpenAI(**client_kwargs)
        self.model = model
        self._is_qwen = "qwen" in model.lower()
        # Qwen provider (Alibaba) hard-limits top_logprobs to 5;
        # Llama and GPT providers support up to 20.
        self._top_logprobs = 5 if self._is_qwen else 20

    def infer(self, prompt_text: str, num_options: int = 4) -> InferenceResult:
        import time

        active_letters = LETTERS[:num_options]

        # For Qwen models, prepend /no_think to disable thinking mode so the
        # first output token is the answer letter, not a reasoning token.
        user_content = f"/no_think\n{prompt_text}" if self._is_qwen else prompt_text

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user_content}],
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=self._top_logprobs,
                    extra_body={"provider": {"require_parameters": True}},
                )
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
                print(f"  API error ({exc}), retrying in {wait}s …")
                time.sleep(wait)

        if not response.choices:
            raise RuntimeError("API response has no choices.")

        choice = response.choices[0]

        # Extract log-probs from the first token's top_logprobs list
        logprobs_data = getattr(choice, "logprobs", None)
        answer_logits: Dict[str, float] = {}
        logprob_content = getattr(logprobs_data, "content", None)
        if logprob_content:
            top_lp = logprob_content[0].top_logprobs
            # Build a lookup: token_string -> log_prob
            lp_lookup: Dict[str, float] = {
                entry.token: entry.logprob for entry in top_lp
            }
            # Match each answer letter to its log-prob
            for letter in active_letters:
                # Check bare letter and space-prefixed letter
                lp = max(
                    lp_lookup.get(letter, self.LOG_PROB_FLOOR),
                    lp_lookup.get(f" {letter}", self.LOG_PROB_FLOOR),
                )
                answer_logits[letter] = lp
        else:
            # ⚠️  FALLBACK: provider did not return logprobs.
            # Assigns artificial logits (predicted → 0.0, others → -20.0).
            # Only prediction_flip remains valid in this case.
            message = getattr(choice, "message", None)
            text = ""
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text_parts: List[str] = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(str(block.get("text", "")))
                    text = "".join(text_parts)

            text = text.strip()
            if not text:
                raise RuntimeError(
                    "Provider returned neither token logprobs nor text content "
                    "(choice.message is missing/empty)."
                )

            pred = next((l for l in active_letters if l in text), active_letters[0])
            answer_logits = {
                l: (0.0 if l == pred else self.LOG_PROB_FLOOR)
                for l in active_letters
            }

        entropy = _entropy_from_logits(answer_logits, active_letters)
        margin = _logit_margin(answer_logits, active_letters)
        pred = _predicted_letter(answer_logits, active_letters)

        return InferenceResult(
            predicted_letter=pred,
            entropy=entropy,
            logit_margin=margin,
            answer_logits=answer_logits,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(model_name: str, **kwargs):
    """
    Convenience factory.

    model_name examples:
        "meta-llama/Meta-Llama-3-8B-Instruct"   → LlamaModel
        "meta-llama/Meta-Llama-3-70B-Instruct"  → LlamaModel
        "gpt-4o"                                  → OpenAIModel
        "gpt-5"                                   → OpenAIModel
        any model with base_url set              → OpenAIModel (e.g. OpenRouter)
    """
    base_url = kwargs.pop("base_url", None)
    openai_prefixes = ("gpt-", "o1", "o3", "o4")
    if base_url or any(model_name.lower().startswith(p) for p in openai_prefixes):
        # Strip local-only quantization kwargs irrelevant to API-based models
        kwargs.pop("load_in_8bit", None)
        kwargs.pop("load_in_4bit", None)
        return OpenAIModel(model=model_name, base_url=base_url, **kwargs)
    return LlamaModel(model_name_or_path=model_name, **kwargs)
