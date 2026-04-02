"""
Test llama-3.3-70b with longer retry intervals to bypass rate limit.
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

MODEL = "qwen/qwen3.5-27b"

PROMPT = (
    "Answer the following multiple-choice question. "
    "Choose the correct answer.\n\n"
    "Question: What is the powerhouse of the cell?\n\n"
    "  A) Mitochondria\n  B) Nucleus\n  C) Ribosome\n  D) Golgi apparatus\n\n"
    "Respond with a single letter (A, B, C, or D).\nAnswer:"
)


def test():
    try:
        resp = client.chat.completions.create(
            model=MODEL,
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
        if "429" in str(e):
            return "-", "429", []
        return "-", str(e)[:80], []


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Strategy: require_parameters=True, exponential backoff\n")

    waits = [10, 20, 30, 60, 60, 60]
    for i, wait in enumerate(waits):
        ans, result, tokens = test()
        print(f"  Attempt {i+1}: ans={ans}  result={result}")
        if tokens:
            print(f"  Top-5: {tokens}")
            print("\n  SUCCESS!")
            break
        if result == "429":
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        else:
            print(f"  Non-429 error, stopping.")
            break
    else:
        print("\n  All attempts exhausted.")
