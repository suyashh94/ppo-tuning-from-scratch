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

# Base vs PPO-Aligned Model Comparison

This demo compares responses from a **base Qwen2.5-1.5B** language model versus a **PPO-aligned** version trained with sentiment-based rewards. It demonstrates how Reinforcement Learning from Human Feedback (RLHF) style training can steer a language model's behavior.

## Alignment Objective

The aligned model was trained using **Proximal Policy Optimization (PPO)** to generate short, positive responses that properly terminate with an EOS token.

### Reward Function

The reward is computed based on two factors:

1. **EOS Token Presence** (checked first):
   - If EOS token is **missing**: Reward = **-10** (sentiment is ignored)
   - If EOS token is **found**: Reward = Sentiment Reward (see below)

2. **Sentiment Reward** (only applies if EOS is found):

   | Sentiment | Reward Formula |
   |-----------|----------------|
   | Positive | +10 × confidence |
   | Neutral | +0.1 × confidence |
   | Negative | -1 × confidence |

This reward structure teaches the model to:
- Always terminate responses properly (EOS token)
- Generate positive sentiment text
- Keep responses short (fewer chances to lose the positive sentiment)

## How It Works

This Space runs on CPU with limited memory. To handle the ~3GB model size, we load models **sequentially**:

1. **Load Base Model** → Generate response → Unload
2. **Load Aligned Model** → Generate response → Unload
3. **Load Reward Model** → Compute sentiment scores → Unload

The UI shows real-time status updates as each model loads and generates.

## Generation Settings

Settings match those used during PPO training:

| Parameter | Value |
|-----------|-------|
| temperature | 1.0 |
| top_k | 50 |
| max_new_tokens | 50 |
| repetition_penalty | 1.0 |

## Expected Results

| Metric | Base Model | Aligned Model |
|--------|------------|---------------|
| Mean Reward | -4 to -10 | ~+9.8 |
| EOS Rate | 0-20% | ~100% |
| Response Length | ~50 tokens | 4-5 tokens |
| Dominant Sentiment | Mixed | Positive |

The aligned model learns to generate very short, positive completions (often just a few words like "great!" or "amazing!") that reliably terminate. The base model generates longer, more varied text but often fails to produce an EOS token within the 50-token limit.

## Training Details

| Component | Details |
|-----------|---------|
| Base Model | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |
| Training Method | PPO with GAE (λ=0.95, γ=1.0) |
| Reward Model | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) |
| Training Steps | ~10,000 |
| Aligned Weights | [suyash94/qwen-ppo-sentiment-aligned](https://huggingface.co/suyash94/qwen-ppo-sentiment-aligned) |

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-6 |
| Batch Size | 4 |
| PPO Epochs | 4 |
| Clip Range | 0.2 |
| GAE Lambda | 0.95 |
| Discount (γ) | 1.0 |
| Value Loss Coefficient | 0.5 |
| Entropy Coefficient | 0.01 |

## Usage

1. Enter a prompt in the text box or click one of the example prompts
2. Click "Generate" to start the comparison
3. Watch the status fields as each model loads and generates
4. Compare the responses and reward breakdowns

**Note:** Total generation time is approximately 60-90 seconds due to sequential CPU loading.

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Gradio application with sequential model loading |
| `models.py` | Qwen model loading (supports HuggingFace Hub download) and text generation |
| `rewarder.py` | Sentiment-based reward computation using RoBERTa |
| `requirements.txt` | Python dependencies |

## Technical Notes

- Models are downloaded from HuggingFace Hub on first run and cached
- The app uses `torch.device("cpu")` to ensure compatibility with CPU-only environments
- Memory is explicitly cleared between model loads using `gc.collect()`
- The aligned weights (~6GB) are loaded on top of the base model architecture

## License

MIT License

## Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen2.5-1.5B base model
- [Cardiff NLP](https://huggingface.co/cardiffnlp) for the sentiment analysis model
- [torchtune](https://github.com/pytorch/torchtune) for the model architecture and checkpointing utilities
