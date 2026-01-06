# Qwen PPO Tuning

PPO (Proximal Policy Optimization) fine-tuning implementation for Qwen2.5-1.5B language model using torchtune and sentiment-based rewards.

This project demonstrates how to use reinforcement learning to align a language model's outputs toward a specific behavior - in this case, generating positive-sentiment text completions that terminate properly.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Reward Function](#reward-function)
- [Training](#training)
- [Configuration](#configuration)
- [Results](#results)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

## Overview

This implementation trains a Qwen2.5-1.5B model using PPO to:

1. **Generate positive sentiment** text completions
2. **Terminate properly** with an EOS token
3. **Keep responses short** (implicitly learned through the reward structure)

The training uses three model instances:
- **Policy Model**: Trainable model that generates text
- **Reference Model**: Frozen copy for KL divergence computation
- **Value Model**: Predicts state values for advantage estimation

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended) or CPU
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/suyashh94/qwen-ppo-tuning.git
cd qwen-ppo-tuning

# Install dependencies
uv sync
uv sync --group dev  # Include development dependencies

# Install PyTorch (choose one based on your hardware)
# For GPU (CUDA 12.1)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU only
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Download Base Model

```bash
python src/scripts/download_base_model.py --output-dir /workspace/base_models
```

This downloads the Qwen2.5-1.5B base model from HuggingFace.

## Quick Start

### Run PPO Training

```bash
python -m qwen_ppo_tuning.ppo_trainer
```

### Test Model Generation

```bash
python -m qwen_ppo_tuning.models
```

### Test Reward Model

```bash
python -m qwen_ppo_tuning.rewarder
```

## Architecture

### Model Hierarchy

```
SetupQwenModel (base: loads weights, creates tokenizer/optimizer)
├── QwenModel (adds generation logic with/without KV cache)
│   ├── PolicyQwenModel (trainable, requires_grad=True)
│   └── ReferenceQwenModel (frozen, provides KL baseline)
└── QwenModelValueHead (adds linear value head for advantage estimation)
```

### Training Flow

1. **Rollout Collection**: Policy model generates responses for input prompts
2. **Reward Computation**: Sentiment model scores the generated text
3. **Advantage Estimation**: GAE (Generalized Advantage Estimation) computes advantages
4. **PPO Update**: Multiple epochs of clipped surrogate objective optimization
5. **KL Penalty**: Prevents policy from diverging too far from reference

### Key Components

| Module | Description |
|--------|-------------|
| `models.py` | Model classes with generation logic (temperature, top-k, KV cache) |
| `ppo_trainer.py` | PPO training loop with GAE, clipping, mixed precision |
| `rewarder.py` | Sentiment-based reward using RoBERTa |
| `utils.py` | Log probability extraction, padding, entropy, advantage whitening |
| `config.py` | Dataclass configurations for all hyperparameters |

## Reward Function

The reward function encourages positive sentiment and proper termination:

### EOS Token Check (Primary)

| Condition | Reward |
|-----------|--------|
| EOS token missing | -10 (sentiment ignored) |
| EOS token found | Sentiment reward (see below) |

### Sentiment Reward (If EOS Found)

| Sentiment | Reward Formula |
|-----------|----------------|
| Positive | +10 × confidence |
| Neutral | +0.1 × confidence |
| Negative | -1 × confidence |

The sentiment analysis uses [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).

### Why This Works

- **EOS penalty dominates**: Model first learns to terminate properly
- **Positive reward is high**: Once terminating, model optimizes for positive sentiment
- **Short responses emerge**: Fewer tokens = fewer chances to introduce negative sentiment

## Training

### PPO Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `vf_coef` | 0.5 | Value loss coefficient |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `gamma` | 1.0 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `num_ppo_epochs` | 4 | PPO epochs per update |
| `kl_coef` | 0.02 | KL penalty coefficient |
| `target_kl` | 0.015 | Target KL for early stopping |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5e-6 | Policy/value learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `total_steps` | 10000 | Total training steps |
| `num_rollouts_per_update` | 8 | Rollouts before PPO update |
| `minibatch_size` | 4 | Minibatch size for PPO epochs |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 50 | Top-k filtering |
| `max_new_tokens` | 50 | Maximum response length |
| `repetition_penalty` | 1.0 | Repetition penalty |
| `use_kv_cache` | False | Enable KV cache for faster generation |

## Configuration

All configurations are defined as dataclasses in `src/qwen_ppo_tuning/config.py`:

- `PPOConfig` - Core PPO algorithm hyperparameters
- `TrainingConfig` - Training loop parameters
- `GenerationConfig` - Text generation settings
- `LoggingConfig` - Output directories and logging frequencies
- `ReferenceModelConfig` - Reference model synchronization
- `PPOMetrics` - Training metrics container

### Output Structure

Training outputs are saved to `./ppo_outputs/{experiment_name}/`:

```
ppo_outputs/
└── {experiment_name}/
    ├── policy_model.pt      # Trained policy weights
    ├── value_model.pt       # Value model weights
    ├── metrics.jsonl        # Training metrics (JSON lines)
    └── checkpoints/         # Periodic checkpoints
```

## Results

### Expected Training Outcomes

| Metric | Base Model | After PPO Training |
|--------|------------|-------------------|
| Mean Reward | -4 to -10 | ~+9.8 |
| EOS Rate | 0-20% | ~100% |
| Response Length | ~50 tokens | 4-5 tokens |
| Dominant Sentiment | Mixed | Positive |

### Learning Dynamics

1. **Early training**: Model learns to produce EOS token (escapes -10 penalty)
2. **Mid training**: Model optimizes for positive sentiment
3. **Late training**: Policy stabilizes with short, positive completions

## Demo

A live demo is available on HuggingFace Spaces:

**[Base vs PPO-Aligned Model Comparison](https://huggingface.co/spaces/suyash94/qwen-ppo-sentiment-demo)**

The demo allows you to:
- Compare base model vs aligned model responses
- See sentiment scores and reward breakdowns
- Try custom prompts or use examples

### Trained Weights

The PPO-aligned weights are available at:
- [suyash94/qwen-ppo-sentiment-aligned](https://huggingface.co/suyash94/qwen-ppo-sentiment-aligned)

## Project Structure

```
qwen-ppo-tuning/
├── src/
│   ├── qwen_ppo_tuning/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration dataclasses
│   │   ├── models.py          # Model definitions
│   │   ├── ppo_trainer.py     # PPO training loop
│   │   ├── rewarder.py        # Sentiment reward model
│   │   └── utils.py           # Utility functions
│   └── scripts/
│       └── download_base_model.py
├── tests/
│   └── test_placeholder.py
├── notebooks/
│   └── ppo_walkthrough.ipynb  # Training walkthrough notebook
├── spaces/                     # HuggingFace Spaces demo
│   ├── app.py
│   ├── models.py
│   ├── rewarder.py
│   ├── requirements.txt
│   └── README.md
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Development

### Running Tests

```bash
pytest
pytest tests -v
pytest tests/test_file.py::test_function -v  # Single test
```

### Linting

```bash
ruff check src tests
ruff format src tests
```

### Type Checking

```bash
mypy src
```

### Code Style

- Line length: 100 characters
- Formatter: Ruff
- Type hints: Required (mypy strict mode)

## Dependencies

### Core

- PyTorch
- torchtune - Model architecture and checkpointing
- transformers - Tokenizers and sentiment model
- huggingface_hub - Model downloads

### Training

- accelerate - Mixed precision training
- trl - RLHF utilities
- peft - Parameter-efficient fine-tuning

### Experiment Tracking

- wandb
- tensorboard
- mlflow

### Visualization

- matplotlib
- seaborn
- plotly

See `pyproject.toml` for the complete dependency list.

## Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen2.5-1.5B base model
- [Cardiff NLP](https://huggingface.co/cardiffnlp) for the sentiment analysis model
- [torchtune](https://github.com/pytorch/torchtune) for the model architecture and training utilities
- [Anthropic](https://www.anthropic.com/) for Claude, which assisted in developing this codebase

## License

MIT License

## Author

Suyash Harlalka ([@suyashh94](https://github.com/suyashh94))
