# train.py

import os
import argparse
import torch
from data_selection import DatasetSelector
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
import utils

load_dotenv("./.env")

def load_limo_subset(proportion=1.0, method='rand'):
    ds_sel = DatasetSelector() 
    ds = ds_sel.select(method=method, proportion=proportion)
    return ds

def main():
    # to use DDP instead of DP 
    # https://huggingface.co/docs/trl/en/sft_trainer#multi-gpu-training
    from accelerate import PartialState
    device_string = PartialState().process_index

    parser = argparse.ArgumentParser("Fine‑tune + merge LoRA")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--proportion", type=float, default=1.0, help="Proportion of the dataset to use (0.0 to 1.0)")
    parser.add_argument("--method", type=str, default="rand", help="Method for dataset selection (e.g., 'rand')")
    args = parser.parse_args()

    model_id   = utils.BASE_MODEL
    hf_token   = os.getenv("HF_KEY")
    output_dir = f"models/{args.name}"
    print(f"Saving model to {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_limo_subset(proportion=args.proportion, method=args.method)

    def to_prompt_completion(examples):
        return {
            "prompt": [
                # you can include whatever instruction prefix you'd like
                "Solve the following problem with chain-of-thought reasoning:\n"
                f"Question: {q}"
                for q in examples["question"]
            ],
            "completion": [
                # the trainer will by default add an <eos> or pad as needed
                f"{sol}"
                for sol in examples["solution"]
            ],
        }

    train_ds = train_ds.map(
        to_prompt_completion,
        batched=True,
        remove_columns=["question", "solution", "answer"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", 
        device_map=None,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        bf16=True,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        group_by_length=True,
        max_steps=-1,
        lr_scheduler_type="constant",
        save_strategy="steps",
        save_steps=500,
        report_to="none",
        push_to_hub=False,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=training_args,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    if device_string == 0:
        print("Training stopped. Saving model.")
        # merge lora weights and save model
        try:
            # save adapter config
            trainer.save_model(output_dir)
            print(f"Saved model to: {output_dir}")
        except:
            import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    main()
