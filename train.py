import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, get_peft_config, get_quantization_config
from trl import SFTConfig
from dotenv import load_dotenv

load_dotenv("./.env")

def load_limo_subset(subset_size):
    dataset = load_dataset("GAIR/LIMO", split="train")
    if subset_size < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(subset_size))
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.1 on LIMO dataset.")
    parser.add_argument("--subset_size", type=int, required=True, help="Number of training examples to use.")
    args = parser.parse_args()

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    hf_token = os.getenv("HF_KEY")
    output_dir = f"models/{args.subset_size}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    limo_dataset = load_limo_subset(args.subset_size)

    # Load model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        quantization_config=quantization_config
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"   # VERY important for LLaMA causal language modeling
    )

    def formatting_func(example):
        return [f"""<s>[INST] Solve the following problem with chain of thought reasoning:
                Question: {example['question']}
                Answer: [/INST] {example['solution']} </s>"""]



    # Configure training
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

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=limo_dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        formatting_func=formatting_func,  
    )

    # Fine-tune model
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
