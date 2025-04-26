# no longer used

# import argparse
# import json
# import os
# import re
# from datasets import load_dataset
# import utils  # must define EVAL_DATASETS and extract_gold_answer

# def extract_final_answer(text: str) -> str:
#     """Pulls out the text inside \\boxed{} or after 'Answer:'."""
#     m = re.search(r"\\boxed\{([^}]*)\}", text)
#     if m:
#         return m.group(1).strip()
#     m = re.search(r"Answer:\s*(.+)", text)
#     if m:
#         return m.group(1).strip()
#     return text.strip()

# def main():
#     p = argparse.ArgumentParser(description="Evaluate saved generations")
#     p.add_argument(
#         "--name",
#         type=str,
#         required=True,
#         help="experiment name (subfolder under `generations/`)",
#     )
#     args = p.parse_args()

#     gen_dir = os.path.join("generations", args.name)
#     if not os.path.isdir(gen_dir):
#         raise FileNotFoundError(f"No generations folder at {gen_dir}")

#     results = {}

#     ds_name = "LIMO"
#     cfg = {"path": "GAIR/LIMO", "answer_key": "answer", "question_key": "question"}

#     # for ds_name, cfg in utils.EVAL_DATASETS.items():
#     gen_path = os.path.join(gen_dir, f"{ds_name}.json")
#     if not os.path.isfile(gen_path):
#         print(f"⚠️  Missing generations for {ds_name}: {gen_path}")
#         return

#     # Load predictions
#     with open(gen_path, "r") as f:
#         preds = json.load(f)

#     # Load gold test split
#     ds = load_dataset(cfg["path"], split="test")
#     n = min(len(preds), len(ds))
#     ds = ds.select(range(n))

#     correct = 0
#     for pred_text, example in zip(preds, ds):
#         pred = extract_final_answer(pred_text)
#         gold = utils.extract_gold_answer(example, cfg)
#         if pred == gold:
#             correct += 1

#     acc = correct / n if n else 0.0
#     results[ds_name] = {"correct": correct, "total": n, "accuracy": acc}
#     print(f"{ds_name:15}  {correct}/{n} = {acc:.4f}")

#     # Save JSON report
#     os.makedirs("results", exist_ok=True)
#     out_path = os.path.join("results", f"{args.name}.json")
#     with open(out_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"\nSaved evaluation results to {out_path}")

# if __name__ == "__main__":
#     main()
