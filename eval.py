import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

EVAL_DATASETS = {
    "AIME24": "math-ai/aime24",
    "MATH500": "HuggingFaceH4/MATH-500",
    "AMC23": "math-ai/amc23",
    "OlympiadBench": "math-ai/olympiadbench",
    "GPQA": "math-ai/gpqa",
    "Minerva": "math-ai/minerva",
}

@torch.no_grad()
def evaluate_model(model, tokenizer):
    model.eval()
    results = {}

    for name, dataset_name in EVAL_DATASETS.items():
        print(f"Evaluating on {name}...")
        dataset = load_dataset(dataset_name, split="test")

        correct = 0
        total = 0
        for example in dataset:
            if "question" in example:
                question = example["question"]
                answer = str(example["answer"]).strip()
            elif "problem" in example:
                question = example["problem"]
                answer = str(example["solution"]).strip()
            else:
                continue  # Skip examples without question/problem

            prompt = f"Solve the following problem with chain of thought reasoning:\nQuestion: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if answer in decoded:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        results[name] = accuracy
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LLaMA model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation results.")
    args = parser.parse_args()

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = os.getenv("HF_KEY")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, args.model_dir)

    print("Evaluating model...")
    eval_results = evaluate_model(model, tokenizer)

    print("\n=== Evaluation Results ===")
    for dataset_name, acc in eval_results.items():
        print(f"{dataset_name}: {acc:.4f}")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(eval_results, f, indent=4)
        print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
