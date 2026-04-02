"""
Test which models on OpenRouter support logprobs.
Sends a simple MCQ prompt and checks if logprobs are returned.
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODELS = [
    # GPT series
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    # Qwen 2.5
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-14b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    # Qwen 3.5
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-122b-a10b",
    # Llama
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    # DeepSeek
    "deepseek/deepseek-chat-v3.1",
]

PROMPT = (
    "Answer the following multiple-choice question. "
    "Choose the correct answer.\n\n"
    "Question: What is the powerhouse of the cell?\n\n"
    "  A) Mitochondria\n"
    "  B) Nucleus\n"
    "  C) Ribosome\n"
    "  D) Golgi apparatus\n\n"
    "Respond with a single letter (A, B, C, or D).\n"
    "Answer:"
)


def test_model(model_name: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        choice = response.choices[0]
        text = choice.message.content.strip() if choice.message.content else ""

        logprobs_data = getattr(choice, "logprobs", None)
        content = getattr(logprobs_data, "content", None) if logprobs_data else None

        has_logprobs = False
        num_top = 0
        sample_tokens = []

        if content and len(content) > 0:
            top_lp = content[0].top_logprobs
            if top_lp and len(top_lp) > 0:
                has_logprobs = True
                num_top = len(top_lp)
                sample_tokens = [
                    (entry.token, round(entry.logprob, 4))
                    for entry in top_lp[:5]
                ]

        return {
            "model": model_name,
            "answer": text,
            "has_logprobs": has_logprobs,
            "num_top_logprobs": num_top,
            "top5_tokens": sample_tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "answer": None,
            "has_logprobs": False,
            "num_top_logprobs": 0,
            "top5_tokens": [],
            "error": str(e),
        }


if __name__ == "__main__":
    print(f"Testing {len(MODELS)} models on OpenRouter for logprobs support...\n")
    print(f"{'Model':<45} {'Answer':<8} {'Logprobs?':<12} {'#Top':<6} {'Top-5 Tokens'}")
    print("-" * 130)

    results = []
    for model in MODELS:
        r = test_model(model)
        results.append(r)

        if r["error"]:
            print(f"{r['model']:<45} {'ERROR':<8} {'-':<12} {'-':<6} {r['error'][:60]}")
        else:
            logprobs_str = "YES" if r["has_logprobs"] else "NO"
            tokens_str = str(r["top5_tokens"]) if r["top5_tokens"] else "-"
            print(f"{r['model']:<45} {r['answer']:<8} {logprobs_str:<12} {r['num_top_logprobs']:<6} {tokens_str[:60]}")

    print("\n\n=== SUMMARY ===\n")
    supported = [r for r in results if r["has_logprobs"]]
    unsupported = [r for r in results if not r["has_logprobs"] and not r["error"]]
    errors = [r for r in results if r["error"]]

    print(f"Logprobs SUPPORTED ({len(supported)}):")
    for r in supported:
        print(f"  [Y] {r['model']}")

    print(f"\nLogprobs NOT supported ({len(unsupported)}):")
    for r in unsupported:
        print(f"  [N] {r['model']}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for r in errors:
            print(f"  [!] {r['model']}: {r['error'][:80]}")
