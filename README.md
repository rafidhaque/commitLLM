# CommitLLM: Automated Git Commit Message Generation

> **"Stop writing `git commit -m 'fix'`. Let AI do the talking."**

## Overview

**CommitLLM** is a research initiative aimed at enhancing developer productivity by automating the generation of semantic git commit messages. This project leverages **Small Language Models (SLMs)**—specifically **Mistral-7B** and **Llama-3**—customized via **Parameter-Efficient Fine-Tuning (PEFT)** techniques.

By utilizing **QLoRA (Quantized Low-Rank Adaptation)**, we demonstrate that high-fidelity code summarization tasks can be performed on consumer-grade hardware (e.g., T4/P100 GPUs) without the need for massive computational clusters or proprietary APIs.

## Objectives

1.  **Customize** an open-source LLM to understand code diff syntax.
2.  **Generate** concise, conventional commit messages from staged changes.
3.  **Evaluate** the performance improvement of fine-tuned models over baseline vanilla models using BLEU and Semantic Similarity metrics.

## Tech Stack & Methodology

* **Core Model:** `Mistral-7B-v0.3` / `Llama-3-8B-Instruct`
* **Optimization:** 4-bit Quantization (NF4) via `bitsandbytes`.
* **Fine-Tuning:** LoRA adapters targeting attention modules (`q_proj`, `v_proj`).
* **Frameworks:** PyTorch, Hugging Face Transformers, PEFT, TRL (Transformer Reinforcement Learning).

## Dataset (Curated)

The model is being trained on a filtered subset of the **[BigCode/CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft)** dataset, specifically focusing on:
* **Languages:** Python, JavaScript, Java.
* **Volume:** ~10,000 - 15,000 high-quality instruction-response pairs.
* **Preprocessing:** Removed merge commits, auto-generated files, and noisy/trivial messages (e.g., "update", "typo").

## Installation

To replicate this experiment, clone the repository and install the required dependencies:

```bash
git clone [https://github.com/rafidhaque/CommitLLM.git](https://github.com/rafidhaque/CommitLLM.git)
cd CommitLLM

# It is recommended to create a virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt