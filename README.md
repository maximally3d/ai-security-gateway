# AI Security Gateway

A local LLM-based system for preventing semantic data leaks to external AI services (ChatGPT, Claude, Copilot) using LoRA fine-tuned models.

## Overview

This project implements a multi-level security gateway that analyzes employee queries before sending them to external AI services. Unlike traditional DLP filters that rely on keyword matching, our system uses **semantic analysis** to detect confidential information, understanding context and paraphrasing.

## Key Features

- **Two-stage architecture**: Fast regex filter + semantic neural analysis
- **Two deployment modes**: Prompt-engineering prototype (87% accuracy) and LoRA fine-tuned (94% accuracy)
- **Local deployment**: No external dependencies, full data control
- **Enterprise integration**: Docker, SIEM, audit logging support

## Performance Metrics

| Metric | Prototype (Prompt Engineering) | Fine-tuned (LoRA) |
|--------|-------------------------------|-------------------|
| Accuracy | 87% | 94% |
| F1-score (HIGH risk) | 0.82 | 0.93 |
| Inference time | 1.8s | 0.9s |
| False positives | 9% | 4% |

## Tech Stack

- **Base Model**: Meta Llama 3.2 3B-Instruct
- **Fine-tuning**: LoRA (r=16, lora_alpha=32)
- **Quantization**: 4-bit (Q4_K_M) for CPU-efficient inference
- **Framework**: Hugging Face Transformers + PEFT

## Installation

    git clone https://github.com/your-username/ai-security-gateway.git
    cd ai-security-gateway
    pip install -r requirements.txt

## Usage

### Prototype Mode (Prompt Engineering)

    from src.gateway import AISecurityGateway
    
    gateway = AISecurityGateway(mode="prototype")
    result = gateway.analyze_query("Help me optimize the payment processing algorithm in QuantumPay")
    print(result)
    # Output: {"risk_level": "HIGH", "reason": "Mention of internal product name..."}

### Fine-tuned Mode (LoRA)

    from src.gateway import AISecurityGateway
    
    gateway = AISecurityGateway(
        mode="fine-tuned",
        base_model_path="meta-llama/Llama-3.2-3B-Instruct",
        lora_adapter_path="./lora-security-classifier"
    )
    result = gateway.analyze_query("Your query here")
    print(result)

## Training

To fine-tune the model on your corporate data:

    # Prepare data (optional, if you have raw_queries.txt)
    python training/prepare_data.py
    
    # Train the model
    python training/train.py

## Project Structure

    ai-security-gateway/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── src/
    │   ├── __init__.py
    │   └── gateway.py
    ├── prompts/
    │   └── system_prompt.txt
    ├── training/
    │   ├── lora_config.json
    │   ├── train.py
    │   └── prepare_data.py
    ├── examples/
    │   └── sample_queries.jsonl
    └── docs/
        └── AI_Security_Gateway_Report.pdf

## Full Project Report

Detailed technical documentation: [docs/AI_Security_Gateway_Report.pdf](docs/AI_Security_Gateway_Report.pdf)

## License

MIT

```bash
pip install -r requirements.txt
