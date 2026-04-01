"""
Semantically equivalent prompt variants for SciQ and TruthfulQA.

Design principles
-----------------
1. Wording anchor: all variants use the same core correctness word ("correct")
   so the task meaning stays identical across variants.
2. Minimal diff: each variant changes at most 1–2 surface dimensions
   (instruction wording, option format, instruction position, or response cue).
3. No dataset leakage: no words that hint at the dataset's design intent
   (e.g. avoid "science", "truthful", "factual" as task-defining adjectives).
4. Consistent response cue style: all variants end with a colon-terminated
   cue so the next token is always the answer letter.
5. No reasoning or explanation signals: no "think step by step", no "explain",
   no role-play framing, no hedging words like "most likely".

Variant dimensions varied across the four variants
---------------------------------------------------
  V0  Baseline     – instruction first, parenthesis option format, "Answer:"
  V1  Reworded     – different instruction phrasing, dot option format,
                     "Your answer:"
  V2  Question-led – question appears before instruction, full-paren format,
                     "Correct option:"
  V3  Minimal      – no instruction sentence, colon format, "Answer:"

The same four structural slots are used for both datasets; only the
instruction sentence is adapted to keep it natural for each dataset.

TruthfulQA note
---------------
All TruthfulQA variants use "correct" (not "truthful", "best", or "accurate")
to avoid leaking the dataset's evaluation philosophy to the model.
"Best" is intentionally avoided everywhere because it is ambiguous and can be
read as "most persuasive" rather than "factually correct".
"""

from typing import Dict, List


LETTERS = ["A", "B", "C", "D"]
NUM_VARIANTS = 4  # number of prompt variants per dataset


def _format_options_standard(options: List[str]) -> str:
    """A) opt  (parenthesis style, two-space indent)"""
    return "\n".join(f"  {LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def _format_options_dot(options: List[str]) -> str:
    """A. opt  (dot style)"""
    return "\n".join(f"  {LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def _format_options_paren(options: List[str]) -> str:
    """(A) opt  (full-parenthesis style)"""
    return "\n".join(f"  ({LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def _format_options_colon(options: List[str]) -> str:
    """A: opt  (colon style)"""
    return "\n".join(f"  {LETTERS[i]}: {opt}" for i, opt in enumerate(options))


# ---------------------------------------------------------------------------
# SciQ prompt variants
# ---------------------------------------------------------------------------

def build_sciq_prompt(variant: int, question: str, options: List[str]) -> str:
    """
    Returns a formatted prompt string for a SciQ MCQ sample.
    variant in {0, 1, 2, 3}

    Dimension matrix
    ----------------
    V0  instruction-first | "Choose the correct answer." | A) format  | "Answer:"
    V1  instruction-first | "Select the correct option." | A. format  | "Your answer:"
    V2  question-first    | "Which option is correct?"   | (A) format | "Correct option:"
    V3  no instruction    | —                            | A: format  | "Answer:"
    """
    assert 0 <= variant < NUM_VARIANTS, f"variant must be 0–{NUM_VARIANTS - 1}"

    if variant == 0:
        # Baseline: instruction → question → options → cue
        return (
            "Answer the following multiple-choice question. "
            "Choose the correct answer.\n\n"
            f"Question: {question}\n\n"
            f"{_format_options_standard(options)}\n\n"
            "Respond with a single letter (A, B, C, or D).\n"
            "Answer:"
        )

    elif variant == 1:
        # Reworded instruction, dot-style options, different cue
        return (
            "Read the question below and select the correct option.\n\n"
            f"{question}\n\n"
            f"{_format_options_dot(options)}\n\n"
            "Reply with only the letter of your answer.\n"
            "Your answer:"
        )

    elif variant == 2:
        # Question appears first, instruction follows, full-parenthesis options
        return (
            f"Question: {question}\n\n"
            f"{_format_options_paren(options)}\n\n"
            "Which option is correct? "
            "Reply with only the corresponding letter.\n"
            "Correct option:"
        )

    else:  # variant == 3
        # Minimal: no instruction sentence, colon options
        return (
            f"{question}\n\n"
            f"{_format_options_colon(options)}\n\n"
            "Answer:"
        )


# ---------------------------------------------------------------------------
# TruthfulQA prompt variants
# ---------------------------------------------------------------------------

def build_truthfulqa_prompt(variant: int, question: str, options: List[str]) -> str:
    """
    Returns a formatted prompt string for a TruthfulQA MCQ sample.
    variant in {0, 1, 2, 3}

    Dimension matrix
    ----------------
    V0  instruction-first | "Choose the correct answer."        | A) format  | "Answer:"
    V1  instruction-first | "Select the correct option."        | A. format  | "Your answer:"
    V2  question-first    | "Which of the options is correct?"  | (A) format | "Correct option:"
    V3  no instruction    | —                                   | A: format  | "Answer:"

    Note: "correct" is used uniformly across all variants (not "truthful",
    "best", or "most accurate") to avoid leaking the dataset's design intent
    and to keep the task framing identical to SciQ for cross-dataset comparison.
    """
    assert 0 <= variant < NUM_VARIANTS, f"variant must be 0–{NUM_VARIANTS - 1}"

    if variant == 0:
        # Baseline: instruction → question → options → cue
        return (
            "Answer the following multiple-choice question. "
            "Choose the correct answer.\n\n"
            f"Question: {question}\n\n"
            f"{_format_options_standard(options)}\n\n"
            "Respond with a single letter (A, B, C, or D).\n"
            "Answer:"
        )

    elif variant == 1:
        # Reworded instruction, dot-style options, different cue
        return (
            "Read the question below and select the correct option.\n\n"
            f"{question}\n\n"
            f"{_format_options_dot(options)}\n\n"
            "Reply with only the letter of your answer.\n"
            "Your answer:"
        )

    elif variant == 2:
        # Question appears first, instruction follows, full-parenthesis options
        return (
            f"Question: {question}\n\n"
            f"{_format_options_paren(options)}\n\n"
            "Which of the options is correct? "
            "Reply with only the corresponding letter.\n"
            "Correct option:"
        )

    else:  # variant == 3
        # Minimal: no instruction sentence, colon options
        return (
            f"{question}\n\n"
            f"{_format_options_colon(options)}\n\n"
            "Answer:"
        )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

PROMPT_BUILDERS: Dict[str, callable] = {
    "sciq": build_sciq_prompt,
    "truthfulqa": build_truthfulqa_prompt,
}


def build_prompt(dataset: str, variant: int, question: str, options: List[str]) -> str:
    """
    dataset : 'sciq' or 'truthfulqa'
    variant : 0 – 3
    """
    builder = PROMPT_BUILDERS.get(dataset)
    if builder is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {list(PROMPT_BUILDERS)}"
        )
    return builder(variant, question, options)


def get_variant_descriptions() -> Dict[int, str]:
    """Human-readable description of each variant (identical for both datasets)."""
    return {
        0: "baseline: instruction-first, A) format, 'Answer:'",
        1: "reworded instruction, A. format, 'Your answer:'",
        2: "question-first, (A) format, 'Correct option:'",
        3: "minimal (no instruction), A: format, 'Answer:'",
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_q = "What is the powerhouse of the cell?"
    sample_opts = ["Mitochondria", "Nucleus", "Ribosome", "Golgi apparatus"]

    for ds in ("sciq", "truthfulqa"):
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        for v in range(NUM_VARIANTS):
            desc = get_variant_descriptions()[v]
            print(f"\n--- Variant {v}: {desc} ---")
            print(build_prompt(ds, v, sample_q, sample_opts))
