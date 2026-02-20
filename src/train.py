
import logging
from pathlib import Path
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- Configuration ---
# Model and Tokenizer
MODEL_ID = "mistralai/Mistral-7B-v0.1"

# Dataset
DATA_PATH = Path("data/processed_commits.csv")
PROMPT_TEMPLATE = """Instruction: Write a git commit message for this diff.
Input:
{diff}

Output:
{message}"""

# QLoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LORA_DROPOUT = 0.05

# Training Arguments
TRAIN_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1
OPTIMIZER = "paged_adamw_32bit"
MAX_SEQ_LENGTH = 512
OUTPUT_DIR = Path("checkpoints/training_run") # Temporary dir for trainer
FINAL_ADAPTER_DIR = Path("checkpoints/commitllm_lora")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def format_sample(sample):
    """Formats a sample into the prompt template."""
    return PROMPT_TEMPLATE.format(diff=sample["input"], message=sample["target"])

def main():
    """Main function to execute the fine-tuning pipeline."""
    logging.info("Starting model fine-tuning script.")

    # 1. Load Dataset
    logging.info(f"Loading dataset from '{DATA_PATH}'...")
    try:
        dataset = load_dataset("csv", data_files=str(DATA_PATH))['train']
    except FileNotFoundError:
        logging.error(f"Dataset file not found at '{DATA_PATH}'. Please run the data_pipeline.py script first.")
        return
    logging.info(f"Dataset loaded successfully with {len(dataset)} samples.")

    # 2. Configure BitsAndBytes for 4-bit quantization
    logging.info("Configuring BitsAndBytes for 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Load Tokenizer and Model
    logging.info(f"Loading model '{MODEL_ID}' and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically map model layers to available devices
    )
    logging.info("Model and tokenizer loaded successfully.")
    
    # 4. Configure LoRA (PEFT)
    logging.info("Configuring LoRA adapter...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Set up Training Arguments
    logging.info("Defining training arguments...")
    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        fp16=False, # fp16 is not compatible with bfloat16
        bf16=True,  # Use bfloat16 for training
        logging_steps=50,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none", # can be set to "wandb"
    )

    # 6. Initialize SFTTrainer
    logging.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=format_sample,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    
    # Enable gradient checkpointing
    model.config.use_cache = False

    # 7. Start Training
    logging.info("Starting training...")
    try:
        trainer.train()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        return

    # 8. Save Final LoRA Adapter
    logging.info(f"Saving LoRA adapter to '{FINAL_ADAPTER_DIR}'...")
    try:
        FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(FINAL_ADAPTER_DIR))
        logging.info("LoRA adapter saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save LoRA adapter: {e}")

if __name__ == "__main__":
    main()
