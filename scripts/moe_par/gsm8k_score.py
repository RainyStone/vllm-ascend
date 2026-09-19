"""GSM8K scoring: few-shot completion prompts and answer extraction, offline.

Behaviour is aligned with ``vllm/tests/evals/gsm8k/gsm8k_eval.py`` -- same
:func:`get_answer_value`, same 5-shot ``Question: .../Answer: ...`` builder
(``:113``), same scorer (``:142``) and the same stop sequences -- but that
script downloads train/test from GitHub on first use, which an air-gapped box
cannot do, so this module reads local jsonl files instead.

``moe_par_bench.py --score`` imports :func:`build_prompts` and :func:`score`
from here; ``python scripts/moe_par/gsm8k_score.py --selftest`` exercises the whole
path without touching a card. Both exist so the accuracy numbers next to the
PAR numbers in the report come from the same prompts and the same extraction
rule as the reference evaluation.

Two details that move the numbers if you get them wrong:

* The prompts are **raw completions**, not chat messages. They go to
  ``llm.generate`` as plain strings; running them through the chat template
  would change the token distribution, and which MoE communication path the
  engine picks keys on the step's token count.
* The label is the *last* integer of the reference answer, which for GSM8K is
  the ``#### N`` at the end of the worked solution. Extraction from the model
  text uses the same rule, so a model that rambles and then commits to a final
  number is scored the way the reference scorer would score it.
"""
import argparse
import ast
import json
import os
import re
import sys

# This module lives in scripts/moe_par/, next to the dataset copies it probes for.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)

# The reference scorer's sentinel for "no number in the text".
INVALID = -9999999

# The reference evaluation's stop strings, unchanged.
STOP = ["Question", "Assistant:", "<|separator|>"]

# Where the train split lives on this box. The first hit wins; a missing train
# split degrades to 0-shot with a warning rather than failing the run.
TRAIN_CANDIDATES = [
    "/home/y00899271/datasets/gsm8k/train.jsonl",
    "/mnt/share/gsm8k/train.jsonl",
    os.path.join(SCRIPT_DIR, "gsm8k_train.jsonl"),
]
TEST_CANDIDATES = [
    "/home/y00899271/datasets/gsm8k/test.jsonl",
    "/mnt/share/gsm8k/test.jsonl",
    os.path.join(SCRIPT_DIR, "gsm8k_test.jsonl"),
]


def read_jsonl(path):
    """Every json object in ``path``, one per non-blank line."""
    rows = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: not valid json ({exc})")
    return rows


def get_answer_value(answer_str):
    """The last integer in ``answer_str``, or :data:`INVALID`.

    Byte-for-byte the reference rule: commas are removed first, so ``1,234``
    is one number and not two, and the surviving digits are run through
    ``ast.literal_eval`` so a stray non-numeric tail cannot smuggle a float in.
    """
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def find_file(explicit, candidates, what):
    """``explicit`` if given, else the first candidate that exists, else None."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"{what} not found: {explicit}")
        return explicit
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_train(path=None):
    """The train split, or ``None`` when this box has no copy of it."""
    path = find_file(path, TRAIN_CANDIDATES, "gsm8k train split")
    return read_jsonl(path) if path else None


def build_prompts(num_questions=0, num_shots=5, train_file=None, test_file=None):
    """Few-shot GSM8K completion prompts plus their ground-truth labels.

    ``test_file`` defaults to the local test split; ``moe_par_bench.py`` passes
    its own ``--dataset`` so the scored questions are the ones the load pass
    ran. ``num_questions=0`` means "every row in the file".

    Returns ``(prompts, labels, meta)``; ``meta`` records which files were
    used and how many shots actually made it into the prompt, because a
    silent downgrade to 0-shot would otherwise look like a model that got
    worse rather than a missing file.
    """
    test_file = find_file(test_file, TEST_CANDIDATES, "gsm8k test split")
    if test_file is None:
        sys.exit("no gsm8k test split found; pass --test-file, or the candidates "
                 f"tried were {TEST_CANDIDATES}")
    test_data = read_jsonl(test_file)
    for i, row in enumerate(test_data):
        if "question" not in row or "answer" not in row:
            sys.exit(f"{test_file}:{i + 1}: needs both 'question' and 'answer'; "
                     f"has {sorted(row)}")

    if num_questions:
        test_data = test_data[:num_questions]

    train_data = load_train(train_file)
    effective_shots = 0
    few_shot_examples = ""
    if train_data:
        effective_shots = min(num_shots, len(train_data))
        for i in range(effective_shots):
            few_shot_examples += (f"Question: {train_data[i]['question']}\n"
                                  f"Answer: {train_data[i]['answer']}\n\n")
    elif num_shots:
        print(f"[score] WARNING: no train split at {TRAIN_CANDIDATES}; scoring 0-shot "
              f"instead of {num_shots}-shot -- accuracy is NOT comparable with a "
              f"5-shot reference", file=sys.stderr)

    prompts, labels = [], []
    for i, row in enumerate(test_data):
        prompts.append(f"{few_shot_examples}Question: {row['question']}\nAnswer:")
        labels.append(get_answer_value(row["answer"]))
    bad = [i for i, label in enumerate(labels) if label == INVALID]
    if bad:
        sys.exit(f"{test_file}: rows {bad[:5]} have no number in 'answer'; the "
                 f"reference scorer asserts on this, so the file is not GSM8K-shaped")

    return prompts, labels, {
        "test_file": os.path.abspath(test_file),
        "train_file": os.path.abspath(find_file(train_file, TRAIN_CANDIDATES, "train"))
        if train_data else None,
        "num_shots": effective_shots,
        "num_questions": len(prompts),
        "strip_comma": True,
    }


def score(preds, labels, latency=0.0, output_tokens=0, shots_meta=None):
    """Accuracy / invalid rate over already-extracted predictions."""
    if len(preds) != len(labels):
        sys.exit(f"score(): {len(preds)} predictions against {len(labels)} labels")
    if not labels:
        sys.exit("score(): nothing to score")
    hits = sum(1 for p, g in zip(preds, labels) if p == g)
    invalid = sum(1 for p in preds if p == INVALID)
    result = {
        "accuracy": hits / len(labels),
        "invalid_rate": invalid / len(labels),
        "n": len(labels),
        "n_correct": hits,
        "n_invalid": invalid,
        "latency_s": round(latency, 3),
        "output_tokens": output_tokens,
        "questions_per_second": round(len(labels) / latency, 3) if latency > 0 else None,
        "tokens_per_second": round(output_tokens / latency, 2) if latency > 0 else None,
    }
    if shots_meta:
        result.update(shots_meta)
    return result


# ---------------------------------------------------------------------- selftest -- #
def _selftest(test_file=None, train_file=None):
    """Check the prompt builder and the extractor on this box's files."""
    problems = []

    def check(name, ok, detail=""):
        print(f"[selftest] {'ok  ' if ok else 'FAIL'} {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            problems.append(name)

    # 1. Extraction matches the reference rule on the awkward cases.
    cases = [("#### 18", 18), ("The answer is 1,234 eggs.", 1234), ("no digits here", INVALID),
             ("42\n#### 7", 7), ("", INVALID)]
    for text, want in cases:
        got = get_answer_value(text)
        check(f"extract({text!r})", got == want, f"got {got}, want {want}")

    # 2. Prompts build, are 5-shot, and end on the dangling 'Answer:'.
    prompts, labels, meta = build_prompts(num_questions=3, num_shots=5,
                                          train_file=train_file, test_file=test_file)
    check("prompt count", len(prompts) == 3, f"got {len(prompts)}")
    check("5-shot examples present", meta["num_shots"] == 5, f"got {meta['num_shots']}")
    head = prompts[0]
    check("shot block + trailing 'Answer:'", head.count("Question:") == 6 and head.endswith("Answer:"),
          f"{head.count('Question:')} 'Question:' blocks, tail {head[-20:]!r}")
    check("labels are ints", all(isinstance(v, int) and v != INVALID for v in labels), f"{labels}")

    # 3. The scored questions are the ones the load pass runs: the repo's own
    #    copy of the test split and the box's dataset copy must agree row for row.
    repo_copy = os.path.join(SCRIPT_DIR, "gsm8k_test.jsonl")
    box_copy = "/home/y00899271/datasets/gsm8k/test.jsonl"
    if os.path.exists(repo_copy) and os.path.exists(box_copy):
        a = read_jsonl(repo_copy)
        b = read_jsonl(box_copy)
        check("repo copy == box copy", len(a) == len(b) and all(
            x["question"] == y["question"] for x, y in zip(a, b)),
            f"{len(a)} vs {len(b)} rows")

    print(f"[selftest] {'all checks passed' if not problems else 'FAILED: ' + ', '.join(problems)}")
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description="GSM8K prompt building and scoring, offline.")
    parser.add_argument("--selftest", action="store_true",
                        help="check extraction and prompt building against the local files")
    parser.add_argument("--train-file", default="", help="5-shot examples")
    parser.add_argument("--test-file", default="", help="questions to score")
    args = parser.parse_args()
    if args.selftest:
        return _selftest(args.test_file, args.train_file)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
