"""
Training script for LoRA fine-tuning of AI Security Gateway.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def prepare_dataset(data_path: str, tokenizer, max_length: int = 512):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_instruction(example):
        instruction = "Analyze query for confidential information."
        input_text = example["input"]
        output_text = example["output"]
        prompt = f"{instruction}\n\nQuery: {input_text}\n\nResponse:"
        full_text = prompt + " " + output_text
        return {"text": full_text}

    dataset = dataset.map(format_instruction)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset


def main():
    print("=" * 60)
    print("AI Security Gateway - LoRA Training")
    print("=" * 60)

    config_path = Path(__file__).parent / "lora_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    data_path = Path(__file__).parent.parent / "examples" / "sample_queries.jsonl"
    print(f"\nLoading dataset: {data_path}")

    dataset = prepare_dataset(
        str(data_path),
        tokenizer,
        max_length=config["training_args"].get("max_length", 512)
    )

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")

    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        bias=config["lora_config"]["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = "./lora-security-classifier"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training_args"]["num_epochs"],
        per_device_train_batch_size=config["training_args"]["batch_size"],
        per_device_eval_batch_size=config["training_args"]["batch_size"],
        gradient_accumulation_steps=config["training_args"]["gradient_accumulation_steps"],
        learning_rate=config["training_args"]["learning_rate"],
        weight_decay=config["training_args"]["weight_decay"],
        warmup_steps=config["training_args"]["warmup_steps"],
        logging_steps=config["training_args"]["logging_steps"],
        save_steps=config["training_args"]["save_steps"],
        eval_strategy="steps",
        eval_steps=config["training_args"]["eval_steps"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )

    print("\nStarting training...")
    train_result = trainer.train()

    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
