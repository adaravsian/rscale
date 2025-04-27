# eval.py

import argparse
import json
import os
import re
from datasets import load_dataset
import utils  # must define EVAL_DATASETS and extract_gold_answer
import pdb
import re
from sympy import sympify, simplify, SympifyError
from tqdm import tqdm

def extract_final_answer(text: str) -> str:
    """Pulls out the text inside \\boxed{} or after 'Answer:'."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Answer:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return None


def normalize_answer(ans: str) -> str:
    """
    1) Turn \frac{a}{b} → (a)/(b)
    2) Turn \pi → pi
    3) Strip leftover LaTeX
    4) Insert * for implicit multiplication everywhere
    5) Remove spaces
    """
    ans = ans.strip()

    # 1) Fractions
    ans = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', ans)
    # 2) Pi
    ans = ans.replace('\\pi', 'pi')
    # 3) Other LaTeX cruft
    ans = ans.replace('\\left', '').replace('\\right', '')
    ans = re.sub(r'\\[a-zA-Z]+', '', ans)
    ans = ans.replace('{', '').replace('}', '')
    # 4) Implicit multiplication: 
    ans = re.sub(r'(?<=[0-9A-Za-z])\(', r'*(', ans)  # x(
    ans = re.sub(r'\)\(', r')*(', ans)               # )(
    ans = re.sub(r'\)(?=[0-9A-Za-z])', r')*', ans)   # )x
    # 5) Whitespace
    ans = ans.replace(' ', '')

    return ans

def main():
    p = argparse.ArgumentParser(description="Evaluate saved generations")
    p.add_argument(
        "--name",
        type=str,
        required=True,
        help="experiment name (subfolder under `generations/`)",
    )
    args = p.parse_args()

    gen_dir = os.path.join("generations", args.name)
    if not os.path.isdir(gen_dir):
        raise FileNotFoundError(f"No generations folder at {gen_dir}")

    results = {}

    for ds_name, cfg in utils.EVAL_DATASETS.items():
        gen_path = os.path.join(gen_dir, f"{ds_name}.json")
        if not os.path.isfile(gen_path):
            print(f"⚠️  Missing generations for {ds_name}: {gen_path}")
            continue

        # Load predictions
        with open(gen_path, "r") as f:
            preds = json.load(f)

        # Load gold test split
        ds = load_dataset(cfg["path"], split="test")
        n = min(len(preds), len(ds))
        ds = ds.select(range(n))

        correct = 0
        for pred_text, example in tqdm(zip(preds, ds), total=n, desc=f"Evaluating {ds_name}"):
            pred = extract_final_answer(pred_text)
            if pred is None:
                continue
            gold = utils.extract_gold_answer(example, cfg)
            # if expressions_equal(pred, gold):
            if pred == gold or normalize_answer(pred) == normalize_answer(gold):
                correct += 1

        acc = correct / n if n else 0.0
        results[ds_name] = {"correct": correct, "total": n, "accuracy": acc}
        print(f"{ds_name:15}  {correct}/{n} = {acc:.4f}")

    # Save JSON report
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_path}")

if __name__ == "__main__":
    main()
