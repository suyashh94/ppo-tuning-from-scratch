"""
Self-contained model code for Qwen2.5-1.5B base and PPO-aligned models.
Adapted from qwen_ppo_tuning.models for Hugging Face Spaces deployment.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torchtune.models.qwen2_5 import qwen2_5_1_5b_base
from torchtune.training import FullModelHFCheckpointer
from transformers import AutoTokenizer


@dataclass
class GenerationConfig:
    """Sampling parameters for text generation - matches training config."""
    temperature: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 50
    repetition_penalty: float = 1.0


class QwenModel(nn.Module):
    """Qwen2.5-1.5B model with text generation capabilities."""

    def __init__(self, model_path: str, device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_path
        self.block_size = 128000
        self._load_model()

    def _get_local_model_path(self) -> str:
        """Get local path to model, downloading from Hub if necessary."""
        # If it's already a local path that exists, use it directly
        if os.path.isdir(self.model_id):
            return self.model_id

        # Otherwise, download from HuggingFace Hub
        print(f"Downloading model from HuggingFace Hub: {self.model_id}")
        local_path = snapshot_download(
            repo_id=self.model_id,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*", "vocab*", "merges*"],
        )
        print(f"Model downloaded to: {local_path}")
        return local_path

    def _load_model(self):
        """Load tokenizer and model weights."""
        # Get local path (download if needed)
        local_path = self._get_local_model_path()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        # Initialize model architecture
        self.model = qwen2_5_1_5b_base()

        # Load checkpoint weights
        ckpt_files = sorted(f for f in os.listdir(local_path) if f.endswith(".safetensors"))
        if not ckpt_files:
            raise ValueError(f"No .safetensors files found in {local_path}")

        # output_dir must differ from checkpoint_dir (we use /tmp since we only load, not save)
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=local_path,
            checkpoint_files=ckpt_files,
            model_type="QWEN2",
            output_dir="/tmp/checkpointer_output",
        )
        state = checkpointer.load_checkpoint()["model"]
        self.model.load_state_dict(state)
        del state

        self.model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

    def load_aligned_weights(self, weights_path: str):
        """Load PPO-aligned weights on top of base model."""
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)
        self.eval()
        print("Aligned weights loaded successfully")

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        temperature: float,
        top_k: int | None,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Sample the next token from logits."""
        next_token_logits = logits / temperature

        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated_tokens.view(-1).tolist()):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= repetition_penalty
                else:
                    next_token_logits[0, token_id] /= repetition_penalty

        # Apply top-k filtering
        if top_k is not None:
            top_k_val = min(top_k, next_token_logits.size(-1))
            indices_to_remove = (
                next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
            )
            next_token_logits[indices_to_remove] = -float("Inf")

        # Handle numerical stability
        next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=1e20, neginf=-1e20)

        # Sample
        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
        probabilities = torch.clamp(probabilities, min=1e-20)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        return next_token

    @torch.no_grad()
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> dict:
        """Generate text from a prompt."""
        config = config or GenerationConfig()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_len = input_ids.size(1)
        total_len = min(input_len + config.max_new_tokens, self.block_size)
        max_new_tokens = total_len - input_len

        generated_tokens = torch.zeros((1, total_len), dtype=torch.long, device=self.device)
        generated_tokens[0, :input_len] = input_ids[0]

        end_token_found = False
        n_tokens_generated = 0
        current_len = input_len

        while not end_token_found and n_tokens_generated < max_new_tokens:
            valid_tokens = generated_tokens[:, :current_len]
            context_input = (
                valid_tokens[:, -self.block_size:]
                if current_len > self.block_size
                else valid_tokens
            )

            logits = self.model(context_input)
            next_logits = logits[:, -1, :]

            next_token = self._sample_next_token(
                next_logits,
                valid_tokens,
                config.temperature,
                config.top_k,
                config.repetition_penalty,
            )

            if next_token.item() == self.tokenizer.eos_token_id:
                end_token_found = True

            generated_tokens[0, current_len] = next_token
            current_len += 1
            n_tokens_generated += 1

        final_tokens = generated_tokens[:, :current_len]
        generated_ids = final_tokens[0][input_len:]
        response_text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)

        return {
            "prompt": prompt,
            "response": response_text,
            "full_text": prompt + response_text,
            "response_length": len(generated_ids),
            "eos_found": end_token_found,
        }
