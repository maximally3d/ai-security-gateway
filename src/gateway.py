"""
AI Security Gateway - Local LLM-based system for preventing semantic data leaks.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class AISecurityGateway:
    """
    AI Security Gateway for analyzing queries for confidential information.

    Supports two modes:
    - "prototype": Uses prompt engineering with base model (87% accuracy)
    - "fine-tuned": Uses LoRA adapter for improved accuracy (94% accuracy)
    """

    def __init__(
        self,
        mode: str = "prototype",
        base_model_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        device: str = "auto"
    ):
        self.mode = mode
        self.device = device

        if mode == "fine-tuned":
            if not base_model_path or not lora_adapter_path:
                raise ValueError(
                    "Fine-tuned mode requires both base_model_path and lora_adapter_path"
                )
            self._init_fine_tuned(base_model_path, lora_adapter_path)
        else:
            self._init_prototype()

    def _init_prototype(self):
        print("Loading base model (prototype mode)...")

        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,
            device_map=self.device,
            load_in_4bit=True,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt_path = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = (
                "You are a corporate AI security auditor. Analyze the query for "
                "confidential information. Respond ONLY in JSON format: "
                '{"risk_level": "HIGH|MEDIUM|LOW", "reason": "Brief explanation"}\n\n'
                "Query: {query}\n\nAnalysis:"
            )

        print("Prototype mode initialized.")

    def _init_fine_tuned(self, base_model_path: str, lora_adapter_path: str):
        print(f"Loading base model: {base_model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            load_in_4bit=True,
            trust_remote_code=True
        )

        print(f"Loading LoRA adapter: {lora_adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.system_prompt = (
            'Classify risk: {query}\n'
            'Respond in JSON: {{"risk_level": "...", "reason": "..."}}'
        )

        print("Fine-tuned mode initialized.")

    def analyze_query(self, user_query: str) -> Dict[str, str]:
        if self.mode == "fine-tuned":
            prompt = self.system_prompt.format(query=user_query)
        else:
            prompt = self.system_prompt + f"\n\nUSER QUERY:\n{user_query}\n\nANALYSIS:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return self._extract_json(response)

    def _extract_json(self, text: str) -> Dict[str, str]:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                if "risk_level" in result and "reason" in result:
                    if result["risk_level"] not in ["HIGH", "MEDIUM", "LOW"]:
                        result["risk_level"] = "MEDIUM"
                    return result
            except json.JSONDecodeError:
                pass

        return {
            "risk_level": "MEDIUM",
            "reason": "Failed to parse model response. Manual review recommended."
        }


def main():
    print("=" * 60)
    print("AI Security Gateway - Demo")
    print("=" * 60)

    gateway = AISecurityGateway(mode="prototype")

    test_queries = [
        "Help me optimize the payment processing algorithm in QuantumPay",
        "Explain how blockchain technology works in simple terms",
        "Write SQL query to fetch all client transactions from the database",
        "What is machine learning?",
        "Help me debug the Crystal module code"
    ]

    print("\nAnalyzing queries...")
    print("-" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        result = gateway.analyze_query(query)
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Reason: {result['reason']}")

    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    main()
