# AI Security Gateway

Local AI gateway for preventing semantic data leaks to external LLM services using LoRA fine-tuned models.

## Overview

This project implements a multi-level security gateway that analyzes employee queries before sending them to external AI services (ChatGPT, Claude, etc.). The system uses semantic analysis to detect confidential information, outperforming traditional keyword-based DLP filters.

## Key Features

- **Two-stage architecture**: Fast regex filter + semantic neural analysis
- **LoRA fine-tuning**: 94% accuracy with 0.9s inference time
- **Local deployment**: No external dependencies, full data control
- **Enterprise integration**: Docker, SIEM, audit logging

## Performance Metrics

| Metric | Prototype (Prompt Engineering) | Fine-tuned (LoRA) |
|--------|-------------------------------|-------------------|
| Accuracy | 87% | 94% |
| F1-score (HIGH risk) | 0.82 | 0.93 |
| Inference time | 1.8s | 0.9s |
| False positives | 9% | 4% |

## Tech Stack

- **Base Model**: Llama 3.2 3B-Instruct
- **Fine-tuning**: LoRA (r=16, lora_alpha=32)
- **Quantization**: 4-bit (Q4_K_M)
- **Framework**: Hugging Face Transformers + PEFT

## Installation

```bash
pip install -r requirements.txt
