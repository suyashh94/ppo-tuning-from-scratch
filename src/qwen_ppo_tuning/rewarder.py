"""Sentiment-based reward model for PPO training."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentRewarder(nn.Module):
    """Reward model based on sentiment analysis.

    Uses cardiffnlp/twitter-roberta-base-sentiment-latest to score responses.
    - Positive sentiment → positive reward
    - Negative sentiment → negative reward
    - Neutral sentiment → near-zero reward

    Rewards are scaled by the model's confidence (probability of dominant class).
    """

    # Label mapping for the cardiffnlp model
    LABEL_MAP = {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
    }

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: torch.device | str | None = None,
        max_length: int = 512,
    ):
        """Initialize the sentiment rewarder.

        Args:
            model_name: HuggingFace model identifier for sentiment analysis.
            device: Device to run the model on. Defaults to CUDA if available.
            max_length: Maximum sequence length for tokenization.
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Get label names from model config
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def get_reward(self, text: str) -> dict:
        """Compute reward for a single text response.

        Args:
            text: The generated response text to score.

        Returns:
            Dictionary containing:
                - reward: The scaled sentiment reward
                - label: The predicted sentiment label
                - confidence: The model's confidence (probability)
                - scores: Raw probabilities for all classes
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Get dominant class
        dominant_idx = probs.argmax().item()
        dominant_label = self.id2label[dominant_idx].lower()
        confidence = probs[dominant_idx].item()

        # Compute scaled reward: sign from sentiment, magnitude from confidence
        base_reward = self.LABEL_MAP.get(dominant_label, 0.0)
        reward = base_reward * confidence

        return {
            "reward": reward,
            "label": dominant_label,
            "confidence": confidence,
            "scores": {self.id2label[i].lower(): probs[i].item() for i in range(len(probs))},
        }

    @torch.no_grad()
    def get_rewards_batch(self, texts: list[str]) -> dict:
        """Compute rewards for a batch of text responses.

        Args:
            texts: List of generated response texts to score.

        Returns:
            Dictionary containing:
                - rewards: Tensor of scaled sentiment rewards
                - labels: List of predicted sentiment labels
                - confidences: Tensor of model confidences
                - scores: List of score dictionaries for each text
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        # Get dominant class for each sample
        dominant_indices = probs.argmax(dim=-1)
        confidences = probs.gather(1, dominant_indices.unsqueeze(1)).squeeze(1)

        rewards = []
        labels = []
        all_scores = []

        for i, (idx, conf) in enumerate(zip(dominant_indices, confidences)):
            label = self.id2label[idx.item()].lower()
            base_reward = self.LABEL_MAP.get(label, 0.0)
            reward = base_reward * conf.item()

            rewards.append(reward)
            labels.append(label)
            all_scores.append(
                {self.id2label[j].lower(): probs[i, j].item() for j in range(probs.size(1))}
            )

        return {
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "labels": labels,
            "confidences": confidences.cpu(),
            "scores": all_scores,
        }


if __name__ == "__main__":
    rewarder = SentimentRewarder()
    print("Sentiment rewarder initialized successfully.")

    test_texts = [
        "I absolutely love this! It's amazing and wonderful!",
        "This is terrible. I hate it so much.",
        "The weather is okay today. Nothing special.",
        "I'm feeling pretty good about the results.",
        "This is the worst experience I've ever had.",
    ]

    print("\nSingle text rewards:")
    for text in test_texts:
        result = rewarder.get_reward(text)
        print(f"  Text: {text[:50]}...")
        print(f"  Label: {result['label']}, Confidence: {result['confidence']:.3f}")
        print(f"  Reward: {result['reward']:.3f}")
        print()

    print("Batch rewards:")
    batch_results = rewarder.get_rewards_batch(test_texts)
    for i, text in enumerate(test_texts):
        print(f"  [{batch_results['labels'][i]:>8}] reward={batch_results['rewards'][i]:>6.3f}  {text[:40]}...")
