# inference_vllm.py – batched generation to control GPU memory usage

import argparse
import os
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm
import utils


def extract_question(datum, config):
    """Return the raw question string given a dataset example and its config."""
    return datum.get(config["question_key"], "").strip()


def generate_and_save(llm, exp_name, max_new_tokens):
    save_dir = os.path.join("generations", exp_name)
    adapter_path = os.path.join("models", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    for ds_name, config in utils.EVAL_DATASETS.items():
        print(f"\nGenerating for {ds_name}")

        ds = load_dataset(config["path"], split="test")
        prompts = [
            f"[INST] Solve the following math problem with clear step‑by‑step reasoning. "
            f"Write the FINAL numerical answer wrapped inside \\boxed{{}} at the end followd by a <STOP> token. "
            f"Example: Problem: Solve 1+1. Answer: 1 + 1 is 2. so the final answer is \\boxed{{2}} <STOP> "
            f"Problem: {extract_question(datum, config)} [/INST]"
            for datum in ds
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            n=1,
            max_tokens=max_new_tokens,
            stop=["<STOP>"],
        )

        all_outputs = llm.generate(
            prompts, 
            sampling_params, 
            use_tqdm=True, 
            lora_request=LoRARequest(exp_name, 1, adapter_path)
        )

        gens = [out.outputs[0].text.strip() for out in all_outputs]

        save_path = os.path.join(save_dir, f"{ds_name}.json")
        with open(save_path, "w") as f:
            json.dump(gens, f, indent=2)
        print(f"Saved {len(gens)} generations to {save_path}")


def main():
    parser = argparse.ArgumentParser("vLLM inference with batching")
    parser.add_argument("--name", type=str, required=True, help="experiment/adapter name")
    parser.add_argument("--gpu_util", type=float, default=0.8, help="fraction of GPU RAM to use")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="generation length budget")
    args = parser.parse_args()

    # Initialise vLLM engine
    llm = LLM(
        model=utils.BASE_MODEL,
        tensor_parallel_size=8,
        gpu_memory_utilization=args.gpu_util,
        enable_lora=True,
    )

    generate_and_save(
        llm,
        exp_name=args.name,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
