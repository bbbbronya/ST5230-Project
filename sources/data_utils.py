"""
Data loading and preprocessing for SciQ and TruthfulQA.

Both datasets are formatted as 4-choice MCQ (options A/B/C/D).
Option order is shuffled per sample with a fixed per-sample seed for reproducibility.
The sampled subset is saved to disk so all models use the same questions.
"""

import json
import os
import random

from datasets import load_dataset

SCIQ_SAMPLE_SIZE = 250
TRUTHFULQA_SAMPLE_SIZE = 250
RANDOM_SEED = 42
LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# SciQ
# ---------------------------------------------------------------------------

def load_sciq(sample_size: int = SCIQ_SAMPLE_SIZE, seed: int = RANDOM_SEED):
    """
    Returns a list of dicts, each with:
        question       : str
        options        : list[str]  (length 4, A/B/C/D)
        correct_letter : str        ('A' | 'B' | 'C' | 'D')
        correct_answer : str
        support        : str        (gold context, not shown to the model)
    """
    dataset = load_dataset("allenai/sciq", split="test")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected_indices = indices[:sample_size]

    formatted = []
    for sample_idx, ds_idx in enumerate(selected_indices):
        item = dataset[ds_idx]
        options = [
            item["correct_answer"],
            item["distractor1"],
            item["distractor2"],
            item["distractor3"],
        ]
        # Shuffle options with a per-sample seed so ordering is reproducible
        item_rng = random.Random(seed + sample_idx * 1000)
        item_rng.shuffle(options)
        correct_letter = LETTERS[options.index(item["correct_answer"])]

        formatted.append(
            {
                "question": item["question"],
                "options": options,
                "correct_letter": correct_letter,
                "correct_answer": item["correct_answer"],
                "support": item.get("support", ""),
            }
        )

    return formatted


# ---------------------------------------------------------------------------
# TruthfulQA (multiple-choice, mc1_targets — exactly one correct answer)
# ---------------------------------------------------------------------------

def load_truthfulqa(sample_size: int = TRUTHFULQA_SAMPLE_SIZE, seed: int = RANDOM_SEED):
    """
    Returns a list of dicts with the same schema as load_sciq
    (without the 'support' field).

    mc1_targets has variable numbers of choices; we always keep the correct
    answer and sample 3 incorrect answers to form a fixed 4-option question.
    """
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    selected_indices = indices[:sample_size]

    formatted = []
    for sample_idx, ds_idx in enumerate(selected_indices):
        item = dataset[ds_idx]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]

        correct_texts = [c for c, l in zip(choices, labels) if l == 1]
        incorrect_texts = [c for c, l in zip(choices, labels) if l == 0]

        if not correct_texts:
            continue  # skip malformed entries

        correct_answer = correct_texts[0]

        # Sample up to 3 incorrect answers
        item_rng = random.Random(seed + sample_idx * 1000)
        k = min(3, len(incorrect_texts))
        chosen_incorrect = item_rng.sample(incorrect_texts, k)

        # Pad with placeholder if fewer than 3 incorrect answers exist
        while len(chosen_incorrect) < 3:
            chosen_incorrect.append("None of the above")

        options = [correct_answer] + chosen_incorrect
        # Shuffle options
        item_rng.shuffle(options)
        correct_letter = LETTERS[options.index(correct_answer)]

        formatted.append(
            {
                "question": item["question"],
                "options": options,
                "correct_letter": correct_letter,
                "correct_answer": correct_answer,
            }
        )

    return formatted


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def save_dataset(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples → {path}")


def load_dataset_from_disk(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entrypoint: cache both datasets to data/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "outputs", "data")

    sciq_path = os.path.join(base, "sciq_sample.json")
    tqa_path = os.path.join(base, "truthfulqa_sample.json")

    print("Loading SciQ …")
    sciq_data = load_sciq()
    save_dataset(sciq_data, sciq_path)

    print("Loading TruthfulQA …")
    tqa_data = load_truthfulqa()
    save_dataset(tqa_data, tqa_path)

    print("Done.")
