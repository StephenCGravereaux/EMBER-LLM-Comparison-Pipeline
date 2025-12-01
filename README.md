# EMBER LLM Pipeline

A machine learning pipeline for malware detection using the EMBER 2018 dataset with LLaMA-based language models for explainable AI analysis.

## Overview

This project combines:
- **EMBER 2018 Dataset**: 1M+ PE file samples for malware detection
- **LightGBM Model**: Traditional ML model for malware classification
- **LoRA Fine-tuned LLaMA**: Language model for generating human-readable explanations
- **SHAP Analysis**: Feature importance extraction for model interpretability

## Features

- Train LoRA models with configurable rank (1-40% of parameters)
- Compare LoRA vs Full Fine-tuned model performance
- Generate detailed malware analysis explanations
- SHAP-based feature importance visualization

## Quick Start (Docker - Recommended)

The complete pipeline with all models and data is available on Docker Hub:

```bash
# Pull the image (includes dataset and trained models)
docker pull stephencgravereaux/ember-llm-pipeline:latest

# Run the pipeline
docker run stephencgravereaux/ember-llm-pipeline:latest

# Run interactively
docker run -it stephencgravereaux/ember-llm-pipeline:latest /bin/bash
```

## Manual Installation

If you want to run from source, you'll need to download the EMBER 2018 dataset separately.

```bash
# Install dependencies
pip install -r requirements.txt

# Install EMBER package
pip install ember/

# Run model comparison
python model_comparison_framework.py
```

## Project Structure

```
├── model_comparison_framework.py  # Main comparison script
├── train_lora_configurable.py     # LoRA training script
├── generate_real_ember_training_data.py  # Training data generator
├── ember/                         # EMBER library
├── ember2018/                     # EMBER 2018 dataset
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── docker-compose.yml             # Docker Compose configuration
```

## Configuration

### LoRA Training (`train_lora_configurable.py`)

| Line | Parameter | Description |
|------|-----------|-------------|
| 47 | `LORA_RANK` | LoRA rank (16=1%, 96=6%, 256=15%, 512=27%) |
| 49 | `LORA_ALPHA` | LoRA alpha scaling factor |
| 50 | `LORA_DROPOUT` | Dropout rate |
| 180 | `num_train_epochs` | Training epochs |
| 181 | `per_device_train_batch_size` | Batch size |

### Model Comparison (`model_comparison_framework.py`)

| Line | Parameter | Description |
|------|-----------|-------------|
| 1196 | `lora_pattern` | Pattern to find LoRA model |
| 1231 | `num_samples` | Number of test samples |
| 1231 | `start_index` | Starting index in dataset |

## Trained Models

| Model | Parameters | Description |
|-------|------------|-------------|
| LoRA r=16 | 1.15% | Minimal adapter |
| LoRA r=96 | 6.44% | Optimal balance (recommended) |
| LoRA r=256 | 15.50% | Good capacity |
| LoRA r=512 | 26.85% | High capacity |
| Full Fine-tuned | 100% | Complete model |

## Dataset

This project uses the [EMBER 2018 dataset](https://github.com/elastic/ember):
- 800K training samples
- 200K test samples
- 2,381 features per sample

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended)
- 8GB+ GPU memory

## License

See individual component licenses in `ember/licenses/`.

## Citation

If you use EMBER in your research, please cite:
```
@article{anderson2018ember,
  title={EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models},
  author={Anderson, Hyrum S and Roth, Phil},
  journal={arXiv preprint arXiv:1804.04637},
  year={2018}
}
```

