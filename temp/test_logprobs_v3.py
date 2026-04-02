"""
Deeper test: use require_parameters=True on a broader set of models.
This was the only strategy that worked (for llama-3.1-8b).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODELS = [
    # Llama series
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    # Qwen
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen3-8b",
    "qwen/qwen3-14b",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-235b-a22b",
    # Mistral
    "mistralai/mistral-small-3.1-24b-instruct",
    "mistralai/ministral-8b",
    # DeepSeek
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-v3.2",
    # Others
    "microsoft/phi-4",
    "google/gemma-3-27b-it",
    # GPT (control group)
    "openai/gpt-4o-mini",
]

PROMPT = (
    "Answer the following multiple-choice question. "
    "Choose the correct answer.\n\n"
    "Question: What is the powerhouse of the cell?\n\n"
    "  A) Mitochondria\n  B) Nucleus\n  C) Ribosome\n  D) Golgi apparatus\n\n"
    "Respond with a single letter (A, B, C, or D).\nAnswer:"
)


def test(model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            extra_body={"provider": {"require_parameters": True}},
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if content and len(content) > 0:
            top = content[0].top_logprobs
            if top and len(top) > 0:
                return text, f"YES({len(top)})"
        return text, "NO"
    except Exception as e:
        msg = str(e)[:100]
        if "no endpoints" in msg.lower() or "not available" in msg.lower():
            return "-", "NO_PROVIDER"
        return "-", f"ERR: {msg[:60]}"


if __name__ == "__main__":
    print(f"{'Model':<50} {'Ans':<6} {'Logprobs (require_parameters=True)'}")
    print("-" * 100)

    yes_models = []
    for model in MODELS:
        ans, result = test(model)
        print(f"{model:<50} {ans:<6} {result}")
        if result.startswith("YES"):
            yes_models.append(model)

    print(f"\n=== MODELS WITH LOGPROBS ({len(yes_models)}) ===\n")
    for m in yes_models:
        print(f"  [Y] {m}")
