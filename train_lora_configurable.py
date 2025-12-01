#!/usr/bin/env python3
"""
Configurable LoRA Training Script for EMBER Malware Detection
Easily adjust LORA_RANK to train models with different parameter percentages

CONFIGURATION GUIDE:
-------------------
Change LORA_RANK below to train different models:

LORA_RANK = 16    →  1.15% parameters  (minimal)
LORA_RANK = 96    →  6.44% parameters  (optimal - recommended)
LORA_RANK = 256   → 15.50% parameters  (good balance)
LORA_RANK = 512   → 26.85% parameters  (high capacity)
LORA_RANK = 896   → 39.11% parameters  (very high)
LORA_RANK = 1024  → 42.00% parameters  (near full)
LORA_RANK = 1280  → 50.00% parameters  (half of full model)

NOTE: Training data does NOT use SHAP. SHAP is only used during evaluation.
"""

import os
import json
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
import gc

# Clear GPU cache before starting
torch.cuda.empty_cache()
gc.collect()

# ============================================================================
# CONFIGURATION - CHANGE THESE VALUES TO TRAIN DIFFERENT MODELS
# ============================================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ⚙️ CHANGE THIS to train different parameter percentages
LORA_RANK = 96  # Default: 96 for ~6.44% parameters (optimal)

LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training data (generated WITHOUT SHAP - uses real PE features)
TRAINING_DATA_FILE = "real_ember_llama_training_20251019_182837.jsonl"

# Output directory with timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"{TIMESTAMP}_LoRA_r{LORA_RANK}_ember_llama"

# ============================================================================

print("=" * 80)
print("🔬 Training Optimal 6-7% LoRA Model on Real EMBER Data")
print("=" * 80)
print(f"📋 Configuration:")
print(f"   LoRA Rank: {LORA_RANK}")
print(f"   Target: ~6-7% parameters")
print(f"   Training Data: {TRAINING_DATA_FILE}")
print(f"   Using DataCollatorForSeq2Seq (same as successful rank 256)")
print("=" * 80)

# Load training data
print("\n" + "=" * 80)
print("📂 Loading Real EMBER Training Data")
print("=" * 80)

if not os.path.exists(TRAINING_DATA_FILE):
    raise FileNotFoundError(f"Training data file not found: {TRAINING_DATA_FILE}")

training_data = []
with open(TRAINING_DATA_FILE, 'r') as f:
    for line in f:
        training_data.append(json.loads(line))

print(f"✅ Loaded {len(training_data)} training examples from real EMBER 2018 dataset")

# Setup model and tokenizer
print("\n" + "=" * 80)
print("🧠 Setting Up Optimal LoRA Model")
print("=" * 80)

print("\n📝 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Tokenizer loaded")

print("🧠 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Base model loaded")

# Configure LoRA
print(f"⚙️  Applying LoRA configuration (rank {LORA_RANK})...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none"
)

model = get_peft_model(model, lora_config)
print("✅ LoRA configuration applied")

# Print model statistics
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
percentage = (trainable_params / total_params) * 100

print(f"\n📊 Model Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Actual percentage: {percentage:.4f}%")
print(f"   GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"   GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# Prepare dataset
print("\n" + "=" * 80)
print("🔧 Preparing Training Dataset")
print("=" * 80)

# Format examples with chat template (same as successful rank 256 training)
formatted_texts = []
for item in training_data:
    formatted_text = f"<|user|>\n{item['input']}\n<|assistant|>\n{item['output']}<|end|>"
    formatted_texts.append(formatted_text)

dataset_dict = {"text": formatted_texts}
dataset = Dataset.from_dict(dataset_dict)

def tokenize_function(examples):
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # Don't pad during tokenization
        max_length=2048,
        return_tensors=None
    )
    # Add labels (same as input_ids for causal LM)
    tokenized['labels'] = [ids[:] for ids in tokenized['input_ids']]
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Split into train/val
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(eval_dataset)}")

# Training arguments - same as successful rank 256 training
print("\n" + "=" * 80)
print("🚀 Starting Optimal LoRA Fine-Tuning")
print("=" * 80)
print(f"📁 Output directory: {OUTPUT_DIR}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=40,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=10,
    fp16=torch.cuda.is_available(),
    report_to=None,
    dataloader_pin_memory=False,
    remove_unused_columns=False
)

# Data collator - using DataCollatorForSeq2Seq like the successful rank 256 training
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Train
print("\n🏋️  Training optimal LoRA model...")
print("⏱️  Monitoring first few steps for speed check...")
trainer.train()

# Save model
print(f"\n💾 Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save metadata
metadata = {
    "model_type": "lora",
    "base_model": BASE_MODEL,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "target_modules": TARGET_MODULES,
    "trainable_params": trainable_params,
    "total_params": total_params,
    "percentage": percentage,
    "training_data": TRAINING_DATA_FILE,
    "num_training_samples": len(train_dataset),
    "num_eval_samples": len(eval_dataset),
    "timestamp": TIMESTAMP
}

with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata saved")

print("\n" + "=" * 80)
print("🎉 Optimal LoRA Model Training Complete!")
print("=" * 80)
print(f"📁 Model saved to: {OUTPUT_DIR}")
print(f"📊 Actual percentage: {percentage:.4f}%")
print(f"🔢 Rank: {LORA_RANK}")
print("\n✅ All done!")

