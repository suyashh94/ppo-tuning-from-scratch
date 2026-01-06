---
title: Base vs PPO-Aligned Qwen Comparison
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🔄 Base vs PPO-Aligned Model Comparison

Compare responses from a **base Qwen2.5-1.5B** model versus a **PPO-aligned** version trained with sentiment-based rewards.

## 🎯 Alignment Objective

The aligned model was trained using **Proximal Policy Optimization (PPO)** to:

| Behavior | Reward |
|----------|--------|
| Positive sentiment | +10 × confidence |
| Neutral sentiment | +0.1 × confidence |
| Negative sentiment | -1 × confidence |
| Missing EOS token | -10 penalty |

**Goal:** Generate short, positive responses that properly terminate.

## ⚙️ Generation Settings

Settings match those used during training:
- `temperature=1.0`
- `top_k=50`
- `max_new_tokens=50`
- `repetition_penalty=1.0`

## 📊 Expected Results

| Metric | Base Model | Aligned Model |
|--------|------------|---------------|
| Mean Reward | ~-4 to -10 | ~+9.8 |
| EOS Rate | ~0-20% | ~100% |
| Response Length | ~50 tokens | ~4-5 tokens |
| Dominant Sentiment | Mixed | Positive |

## 🏗️ Training Details

- **Base Model:** [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Training Method:** PPO with GAE (λ=0.95, γ=1.0)
- **Reward Model:** [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Training Steps:** ~10,000

## 🚀 Usage

1. Enter a prompt or click one of the example prompts
2. Click "Generate" to see responses from both models
3. Compare the sentiment scores and reward values

## 📁 Files

- `app.py` - Main Gradio application
- `models.py` - Qwen model loading and generation
- `rewarder.py` - Sentiment-based reward computation
- `requirements.txt` - Python dependencies

## 📜 License

MIT License
