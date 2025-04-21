# train.py

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
import utils

load_dotenv("./.env")

def load_limo_subset():
    ds = load_dataset("GAIR/LIMO", split="train")
    # ds = ds.shuffle(seed=42).select(range(subset_size))
    return ds

def main():
    parser = argparse.ArgumentParser("Fine‑tune + merge LoRA")
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    model_id   = utils.BASE_MODEL
    hf_token   = os.getenv("HF_KEY")
    output_dir   = f"models/{args.name}"

    # ─── 1) TOKENIZER ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ─── 2) DATASET ─────────────────────────────────────────────────────────
    train_ds = load_limo_subset()

    # ─── 3) BASE MODEL (QLORA) ──────────────────────────────────────────────
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        quantization_config=quant_cfg,
    )

    # ─── 4) LoRA CONFIG ─────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ─── 5) FORMAT FUNCTION ─────────────────────────────────────────────────
    def formatting_func(example):
        return [
            "<s>[INST] Solve the following problem with chain of thought reasoning:\n"
            f"Question: {example['question']}\n"
            "Answer: [/INST] "
            f"{example['solution']} </s>"
        ]

    # ─── 6) TRAINING ARGS ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=50,
        push_to_hub=False,
        report_to=[]
    )

    # ─── 7) INITIALIZE & RUN TRAINER ─────────────────────────────────────────
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
