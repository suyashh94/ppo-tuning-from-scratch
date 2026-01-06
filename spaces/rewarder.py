"""
Sentiment-based reward model for scoring generated responses.
Adapted from qwen_ppo_tuning.rewarder for Hugging Face Spaces deployment.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentRewarder(nn.Module):
    """
    Reward model based on sentiment analysis.

    Uses cardiffnlp/twitter-roberta-base-sentiment-latest to score responses.
    - Positive sentiment -> high positive reward (+10 * confidence)
    - Negative sentiment -> small negative reward (-1 * confidence)
    - Neutral sentiment -> small positive reward (+0.1 * confidence)

    This is the same reward function used during PPO training.
    """

    LABEL_MAP = {
        "negative": -1.0,
        "neutral": 0.1,
        "positive": 10.0,
    }

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: torch.device | str | None = None,
        max_length: int = 512,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def get_reward(self, text: str) -> dict:
        """
        Compute reward for a text response.

        Args:
            text: The full text (prompt + response) to score.

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

        # Compute scaled reward
        base_reward = self.LABEL_MAP.get(dominant_label, 0.0)
        reward = base_reward * confidence

        return {
            "reward": reward,
            "label": dominant_label,
            "confidence": confidence,
            "scores": {self.id2label[i].lower(): probs[i].item() for i in range(len(probs))},
        }
