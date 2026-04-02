"""
Retry logprobs test for Llama models with require_parameters=True.
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]

PROMPT = (
    "Answer the following multiple-choice question. "
    "Choose the correct answer.\n\n"
    "Question: What is the powerhouse of the cell?\n\n"
    "  A) Mitochondria\n  B) Nucleus\n  C) Ribosome\n  D) Golgi apparatus\n\n"
    "Respond with a single letter (A, B, C, or D).\nAnswer:"
)


def test(model, attempt=1):
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
                tokens = [(t.token.strip(), round(t.logprob, 4)) for t in top[:5]]
                return text, f"YES({len(top)})", tokens
        return text, "NO", []
    except Exception as e:
        return "-", f"ERR: {str(e)[:80]}", []


if __name__ == "__main__":
    for model in MODELS:
        print(f"\n--- {model} ---")
        for attempt in range(3):
            ans, result, tokens = test(model)
            print(f"  Attempt {attempt+1}: ans={ans}  logprobs={result}")
            if tokens:
                print(f"  Top-5: {tokens}")
            if result.startswith("YES"):
                break
            if "429" in result:
                print("  Rate limited, waiting 5s...")
                time.sleep(5)
            else:
                break
