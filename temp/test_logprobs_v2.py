"""
Test logprobs on OpenRouter with provider routing.
Uses require_parameters=True and specific provider targeting.
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
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen3.5-9b",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.5-122b-a10b",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat-v3.1",
]

PROVIDER_CONFIGS = [
    ("require_params", {"require_parameters": True}),
    ("fireworks",      {"only": ["Fireworks"], "allow_fallbacks": False}),
    ("together",       {"only": ["Together"], "allow_fallbacks": False}),
    ("deepinfra",      {"only": ["DeepInfra"], "allow_fallbacks": False}),
]

PROMPT = (
    "Answer the following multiple-choice question. "
    "Choose the correct answer.\n\n"
    "Question: What is the powerhouse of the cell?\n\n"
    "  A) Mitochondria\n  B) Nucleus\n  C) Ribosome\n  D) Golgi apparatus\n\n"
    "Respond with a single letter (A, B, C, or D).\nAnswer:"
)


def test(model, provider_cfg):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            extra_body={"provider": provider_cfg},
        )
        choice = resp.choices[0]
        lp = getattr(choice, "logprobs", None)
        content = getattr(lp, "content", None) if lp else None
        if content and len(content) > 0:
            top = content[0].top_logprobs
            if top and len(top) > 0:
                return f"YES({len(top)})"
        return "NO"
    except Exception as e:
        msg = str(e)
        if "not available" in msg.lower() or "no endpoints" in msg.lower():
            return "N/A"
        return "ERR"


if __name__ == "__main__":
    labels = [l for l, _ in PROVIDER_CONFIGS]
    hdr = f"{'Model':<42}" + "".join(f" {l:<16}" for l in labels)
    print(hdr)
    print("-" * len(hdr))

    winners = []
    for model in MODELS:
        row = f"{model:<42}"
        won = []
        for label, cfg in PROVIDER_CONFIGS:
            r = test(model, cfg)
            row += f" {r:<16}"
            if r.startswith("YES"):
                won.append(label)
        print(row)
        if won:
            winners.append((model, won))

    print("\n=== WORKING COMBINATIONS ===\n")
    if winners:
        for m, provs in winners:
            print(f"  [Y] {m}  via  {', '.join(provs)}")
    else:
        print("  None found.")
