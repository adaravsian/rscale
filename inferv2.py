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


def generate_and_save(llm, exp_name, batch_size, max_new_tokens):
    """Generate answers in batches and save them as JSON files.

    Args:
        llm: vLLM LLM instance.
        exp_name: Name of the fine‑tuning/adapter experiment (used for LoRA path).
        batch_size: Number of prompts to send to vLLM at once.
        max_new_tokens: Decoding budget per completion.
    """
    save_dir = os.path.join("generations", exp_name)
    adapter_path = os.path.join("models", exp_name)
    os.makedirs(save_dir, exist_ok=True)

    ds_name = "LIMO"
    config = {"path": "GAIR/LIMO", "answer_key": "answer", "question_key": "question"}
    # for ds_name, config in utils.EVAL_DATASETS.items():
    print(f"\nGenerating for {ds_name}… (batch={batch_size})")

    # ds = load_dataset(config["path"], split="test")
    ds = load_dataset(config["path"], split="train")
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

    all_outputs = []
    # ------------------------------------------------------------------
    # Batched generation loop – keeps KV‑cache size bounded
    # ------------------------------------------------------------------
    for i in tqdm(range(0, len(prompts), batch_size)):
        prompt_chunk = prompts[i : i + batch_size]
        chunk_outputs = llm.generate(
            prompt_chunk,
            sampling_params,
            use_tqdm=False,  # outer tqdm already shows progress
            lora_request=LoRARequest(exp_name, 1, adapter_path),
        )
        all_outputs.extend(chunk_outputs)

    gens = [out.outputs[0].text.strip() for out in all_outputs]

    save_path = os.path.join(save_dir, ".json")
    with open(save_path, "w") as f:
        json.dump(gens, f, indent=2)
    print(f"Saved {len(gens)} generations to {save_path}")


def main():
    parser = argparse.ArgumentParser("vLLM inference with batching")
    parser.add_argument("--name", type=str, required=True, help="experiment/adapter name")
    parser.add_argument("--model", type=str, default="7b", help="llama2 model size: 7b, 13b, 70b")
    parser.add_argument("--batch_size", type=int, default=64, help="prompts per batch")
    parser.add_argument("--gpu_util", type=float, default=0.8, help="fraction of GPU RAM to use")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="generation length budget")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-2-" + args.model + "-chat-hf"

    # Initialise vLLM engine
    llm = LLM(
        model=model_name,
        tensor_parallel_size=8,
        gpu_memory_utilization=args.gpu_util,
        enable_lora=True,
    )

    generate_and_save(
        llm,
        exp_name=args.name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
