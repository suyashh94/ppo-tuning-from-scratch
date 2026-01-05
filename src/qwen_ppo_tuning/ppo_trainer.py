import json
import math
import os
import random

import torch
from torch.optim import lr_scheduler

from qwen_ppo_tuning.config import (
    GenerationConfig,
    LoggingConfig,
    PPOConfig,
    PPOMetrics,
    ReferenceModelConfig,
    TrainingConfig,
)
from qwen_ppo_tuning.models import PolicyQwenModel, QwenModelValueHead, ReferenceQwenModel
from qwen_ppo_tuning.rewarder import SentimentRewarder
from qwen_ppo_tuning.utils import (
    calculate_clip_fraction,
    calculate_entropy,
    calculate_value_clip_fraction,
    logprobs_from_logits,
    whiten,
)


class PPOTrainer:
    def __init__(
        self,
        policy_model: PolicyQwenModel,
        value_model: QwenModelValueHead,
        reference_model: ReferenceQwenModel,
        reward_model: SentimentRewarder,
        ppo_config: PPOConfig | None = None,
        training_config: TrainingConfig | None = None,
        generation_config: GenerationConfig | None = None,
        logging_config: LoggingConfig | None = None,
        ref_model_config: ReferenceModelConfig | None = None,
    ):
        # Models
        self.policy_model = policy_model
        self.value_model = value_model
        self.reference_model = reference_model
        self.reward_model = reward_model

        # Config objects (use defaults if not provided)
        self.ppo_config = ppo_config or PPOConfig()
        self.training_config = training_config or TrainingConfig()
        self.generation_config = generation_config or GenerationConfig()
        self.logging_config = logging_config or LoggingConfig()
        self.ref_model_config = ref_model_config or ReferenceModelConfig()

        # Setup output directories
        os.makedirs(self.logging_config.experiment_dir, exist_ok=True)

        # Device and training state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.train_steps_taken = 0
        self.experience_collection_counter = 0
        self.rewards_since_last_save = []
        self.last_saved_mean_reward = -float("inf")

        # Metrics tracking
        self.metrics_history: list[dict] = []
        self.metrics_file = os.path.join(self.logging_config.experiment_dir, "metrics.jsonl")

        self._setup_schedulers()
        self.save_configs()
        self.dataset = [
            "I think the book was",
            "The movie was",
            "Overall, the product is",
            "The restaurant experience was",
            "The service at the hotel was",
            "The game was",
            "The concert was",
            "The play was",
        ]

    def save_configs(self):
        """Save configuration dataclasses to JSON files."""
        configs = {
            "ppo_config": self.ppo_config,
            "training_config": self.training_config,
            "generation_config": self.generation_config,
            "logging_config": self.logging_config,
            "ref_model_config": self.ref_model_config,
        }
        for name, config in configs.items():
            config_path = os.path.join(self.logging_config.experiment_dir, f"{name}.json")
            with open(config_path, "w") as f:
                json.dump(config.__dict__, f, indent=4)

    def _create_cosine_warmup_scheduler(self, optimizer, min_lr_ratio):
        """Create a cosine annealing scheduler with linear warmup."""
        total_steps = self.training_config.total_steps
        warmup_steps = int(self.training_config.warmup_steps_frac * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = min(progress, 1.0)  # Clamp to 1.0
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _create_linear_warmup_scheduler(self, optimizer):
        """Create a linear warmup scheduler followed by linear decay."""
        total_steps = self.training_config.total_steps
        warmup_steps = int(self.training_config.warmup_steps_frac * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _setup_schedulers(self):
        min_lr_ratio = 0.1
        if self.training_config.lr_scheduler_type == "cosine":
            self.policy_scheduler = self._create_cosine_warmup_scheduler(
                self.policy_model.optimizer, min_lr_ratio
            )
            self.value_scheduler = self._create_cosine_warmup_scheduler(
                self.value_model.optimizer, min_lr_ratio
            )
        elif self.training_config.lr_scheduler_type == "linear":
            self.policy_scheduler = self._create_linear_warmup_scheduler(
                self.policy_model.optimizer
            )
            self.value_scheduler = self._create_linear_warmup_scheduler(self.value_model.optimizer)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {self.training_config.lr_scheduler_type}"
            )

    def collect_experience(self):
        """Collect experience using the current policy model."""
        self.policy_model.eval()
        self.value_model.eval()
        self.experience_collection_counter += 1
        i = 0
        policy_model_outputs = []

        while i < self.training_config.num_rollouts_per_update:
            policy_model_output = self.policy_model.generate(
                prompt=random.choice(self.dataset), config=self.generation_config
            )
            policy_model_outputs.append(policy_model_output)
            i += 1

        policy_response_properties = self.policy_model.get_batched_response_properties(
            policy_model_outputs, pad_to=128
        )
        ref_response_properties = self.reference_model.get_batched_response_properties(
            policy_model_outputs, pad_to=128
        )
        # Store policy model's logprobs as old_logprobs (for PPO ratio calculation)
        # These are the logprobs from the policy at generation time, before any updates
        policy_response_properties["old_logprobs"] = (
            policy_response_properties["logprobs"].detach().clone()
        )
        # Store reference model's logprobs separately for KL penalty calculation
        policy_response_properties["ref_logprobs"] = ref_response_properties["logprobs"].detach()

        with torch.no_grad():
            value_preds = self.value_model(
                input_ids=policy_response_properties["padded_generated_ids"].to(self.device)
            )
        value_preds = value_preds[:, :-1, :].squeeze(-1).to(self.device)

        rewards = self.get_rewards(policy_model_outputs)
        self.rewards_since_last_save.extend(rewards.clone().cpu().tolist())
        # KL penalty is between policy and reference model (not old policy)
        kl_diff = policy_response_properties["logprobs"].to(
            self.device
        ) - policy_response_properties["ref_logprobs"].to(self.device)
        kl_penalty = self.ppo_config.kl_coef * kl_diff

        total_rewards_aligned = self.collate_rewards_tensor(
            rewards,
            policy_response_properties["response_id_lengths"],
            policy_response_properties["response_mask"].size(1),
        )

        total_rewards_aligned = total_rewards_aligned - kl_penalty

        return {
            "policy_response_properties": policy_response_properties,
            "value_preds": value_preds,
            "rewards": rewards,
            "kl_penalty": kl_penalty,
            "total_rewards": rewards,
            "policy_model_outputs": policy_model_outputs,
            "rewards_tensor": total_rewards_aligned,
        }

    def get_rewards(self, policy_model_outputs: list[dict]) -> torch.Tensor:
        """Compute rewards for the generated responses.

        Args:
            policy_model_outputs: List of dictionaries containing model outputs for each rollout.
        Returns:
            Tensor of rewards for each generated response.
        """
        rewards = []
        for output in policy_model_outputs:
            response_ids = output["response_ids"][0].tolist()
            if self.policy_model.tokenizer.eos_token_id in response_ids:
                response_text = output["input_text"] + output["response_text"]
                reward_dict = self.reward_model.get_reward(response_text)
            else:
                reward_dict = {"reward": -10.0}
            rewards.append(reward_dict["reward"])
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def collate_rewards_tensor(
        self, rewards: torch.Tensor, response_id_lengths: list[int], max_len: int
    ):
        """Collate rewards into a tensor aligned with generated sequences.

        Args:
            rewards: Tensor of shape (batch_size,) containing rewards for each response.
            response_id_lengths: List of lengths of each response in the batch.
            max_len: Maximum length of generated sequences after padding.

        Returns:
            Tensor of shape (batch_size, max_len) with rewards aligned to response tokens.

        Note:
            The reward is placed at position response_length - 2 (not response_length - 1)
            because the masks/logits are shifted by 1. The action at position t generates
            token t+1, so the last response action is at response_length - 2 in the
            shifted indexing used by response_mask.
        """
        batch_size = rewards.size(0)
        rewards_tensor = torch.zeros((batch_size, max_len), device=rewards.device)

        for i in range(batch_size):
            response_length = response_id_lengths[i]
            # Place reward at last response action position (aligned with response_mask)
            # response_mask covers positions prompt_length-1 to response_length-2
            reward_position = max(0, response_length - 2)
            rewards_tensor[i, reward_position] = rewards[i]

        return rewards_tensor

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Tensor of shape (batch_size, seq_len) containing rewards.
            values: Tensor of shape (batch_size, seq_len) containing value predictions.
            masks: Tensor of shape (batch_size, seq_len) indicating valid tokens.

        Returns:
            Tensor of shape (batch_size, seq_len) containing advantage estimates.
        """
        batch_size, seq_len = rewards.size()
        advantages = torch.zeros((batch_size, seq_len), device=rewards.device)
        last_gae_lam = torch.zeros(batch_size, device=rewards.device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_non_terminal = 0.0
                next_values = 0.0
            else:
                next_non_terminal = masks[:, t + 1]
                next_values = values[:, t + 1]

            delta = (
                rewards[:, t]
                + self.ppo_config.gamma * next_values * next_non_terminal
                - values[:, t]
            )
            advantages[:, t] = last_gae_lam = (
                delta
                + self.ppo_config.gamma
                * self.ppo_config.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )

        return advantages

    def train_step(self) -> PPOMetrics:
        """Main PPO training loop. Returns metrics for this update."""

        rollout_data = self.collect_experience()
        policy_response_properties = rollout_data["policy_response_properties"]
        rewards_tensor = rollout_data["rewards_tensor"]  # aligned rewards with kl penalty
        raw_rewards = rollout_data["rewards"]  # per-sample rewards
        kl_penalty = rollout_data["kl_penalty"]
        value_preds = rollout_data["value_preds"].to(self.device)
        policy_model_outputs = rollout_data["policy_model_outputs"]

        advantages = self.compute_gae(
            rewards_tensor,
            value_preds.squeeze(-1),
            policy_response_properties["response_mask"].to(self.device),
        )

        # Compute response stats before normalization
        response_lengths = [out["generated_response_length"] for out in policy_model_outputs]
        eos_count = sum(1 for out in policy_model_outputs if out["end_token_found"])
        raw_advantages = advantages.clone()

        if self.ppo_config.normalize_advantages:
            advantages = whiten(
                advantages, policy_response_properties["response_mask"].to(self.device)
            )

        self.policy_model.train()
        self.value_model.train()

        # Accumulators for metrics
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_approx_kl = []
        all_clip_fractions = []
        all_value_clip_fractions = []
        all_ratios = []

        for _ in range(self.ppo_config.num_ppo_epochs):
            epoch_kl_divs = []
            for start_idx in range(
                0, self.training_config.num_rollouts_per_update, self.training_config.minibatch_size
            ):
                end_idx = start_idx + self.training_config.minibatch_size
                mb_indices = torch.arange(start_idx, end_idx)

                mb_padded_generated_ids = policy_response_properties["padded_generated_ids"][
                    mb_indices
                ].to(self.device)
                mb_response_mask = policy_response_properties["response_mask"][mb_indices].to(
                    self.device
                )
                mb_old_logprobs = policy_response_properties["logprobs"][mb_indices].to(self.device)
                mb_advantages = advantages[mb_indices]  # Whitened advantages for policy loss
                mb_raw_advantages = raw_advantages[mb_indices]  # Raw advantages for value loss
                mb_value_preds = value_preds[mb_indices]

                # Forward pass with mixed precision
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    # Forward pass through policy model
                    logits = self.policy_model(
                        input_ids=mb_padded_generated_ids,
                    )
                    logits = logits[:, :-1, :]

                    logprobs = logprobs_from_logits(logits, labels=mb_padded_generated_ids[:, 1:])

                    # Calculate ratios with numerical stability
                    # Clamp log ratio to prevent extreme values
                    log_ratio = logprobs - mb_old_logprobs
                    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)  # Prevent exp overflow
                    ratios = torch.exp(log_ratio)

                    # Policy loss
                    surr1 = ratios * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratios,
                            1.0 - self.ppo_config.clip_epsilon,
                            1.0 + self.ppo_config.clip_epsilon,
                        )
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss = (policy_loss * mb_response_mask).sum() / mb_response_mask.sum()

                    # Value loss
                    values = self.value_model(input_ids=mb_padded_generated_ids).squeeze(-1)
                    values = values[:, :-1]
                    value_pred_clipped = mb_value_preds + torch.clamp(
                        values - mb_value_preds,
                        -self.ppo_config.vf_clip_epsilon,
                        self.ppo_config.vf_clip_epsilon,
                    )
                    # Use RAW advantages (not whitened) for computing returns
                    # returns = advantages + values, so returns = raw_adv + old_values
                    returns = mb_raw_advantages.detach() + mb_value_preds.detach()
                    value_losses1 = (values - returns) ** 2
                    value_losses2 = (value_pred_clipped - returns) ** 2
                    value_loss = torch.max(value_losses1, value_losses2)
                    value_loss = (value_loss * mb_response_mask).sum() / mb_response_mask.sum()

                    # Entropy (always compute for logging)
                    entropy = calculate_entropy(logits, mb_response_mask)
                    entropy_loss = (
                        -self.ppo_config.entropy_coef * entropy
                        if self.ppo_config.use_entropy_loss
                        else 0.0
                    )

                    total_loss = policy_loss + self.ppo_config.vf_coef * value_loss + entropy_loss

                # Check for NaN/Inf in loss before backward pass
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"[Warning] NaN/Inf detected in loss at step {self.train_steps_taken}.")
                    print(f"  policy_loss={policy_loss.item()}, value_loss={value_loss.item()}")
                    print(
                        f"  advantages: min={mb_advantages.min().item():.4f}, max={mb_advantages.max().item():.4f}"
                    )
                    print(
                        f"  logprobs: min={logprobs.min().item():.4f}, max={logprobs.max().item():.4f}"
                    )
                    print(
                        f"  old_logprobs: min={mb_old_logprobs.min().item():.4f}, max={mb_old_logprobs.max().item():.4f}"
                    )
                    self.policy_model.optimizer.zero_grad()
                    self.value_model.optimizer.zero_grad()
                    self.train_steps_taken += 1
                    continue

                self.policy_model.optimizer.zero_grad()
                self.value_model.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.policy_model.optimizer)
                self.scaler.unscale_(self.value_model.optimizer)

                # Check for NaN in gradients
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.training_config.max_grad_norm
                )
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_model.parameters(), self.training_config.max_grad_norm
                )

                if torch.isnan(policy_grad_norm) or torch.isnan(value_grad_norm):
                    print(
                        f"[Warning] NaN gradient detected at step {self.train_steps_taken}. Skipping update."
                    )
                    print(
                        f"  policy_grad_norm={policy_grad_norm.item()}, value_grad_norm={value_grad_norm.item()}"
                    )
                    self.policy_model.optimizer.zero_grad()
                    self.value_model.optimizer.zero_grad()
                    # Must call update() to reset scaler state after unscale_()
                    self.scaler.update()
                    self.train_steps_taken += 1
                    continue

                self.scaler.step(self.policy_model.optimizer)
                self.scaler.step(self.value_model.optimizer)
                self.scaler.update()
                self.train_steps_taken += 1

                # Collect metrics (detached)
                with torch.no_grad():
                    approx_kl = (
                        (mb_old_logprobs - logprobs) * mb_response_mask
                    ).sum() / mb_response_mask.sum()
                    clip_frac = calculate_clip_fraction(ratios, self.ppo_config.clip_epsilon)
                    value_clip_frac = calculate_value_clip_fraction(
                        values, mb_value_preds, self.ppo_config.vf_clip_epsilon
                    )
                    masked_ratio_mean = (ratios * mb_response_mask).sum() / mb_response_mask.sum()

                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropies.append(entropy.item())
                    all_approx_kl.append(approx_kl.item())
                    all_clip_fractions.append(clip_frac)
                    all_value_clip_fractions.append(value_clip_frac)
                    all_ratios.append(masked_ratio_mean.item())
                    epoch_kl_divs.append(approx_kl.item())
            # Check for early stopping based on epoch KL
            epoch_kl = sum(epoch_kl_divs) / len(epoch_kl_divs)
            if epoch_kl > self.ppo_config.max_kl:
                break

        # Compute aggregated metrics
        metrics = PPOMetrics(
            policy_loss=sum(all_policy_losses) / len(all_policy_losses),
            value_loss=sum(all_value_losses) / len(all_value_losses),
            total_loss=(sum(all_policy_losses) + self.ppo_config.vf_coef * sum(all_value_losses))
            / len(all_policy_losses),
            entropy=sum(all_entropies) / len(all_entropies),
            approx_kl=sum(all_approx_kl) / len(all_approx_kl),
            clip_fraction=sum(all_clip_fractions) / len(all_clip_fractions),
            value_clip_fraction=sum(all_value_clip_fractions) / len(all_value_clip_fractions),
            mean_reward=raw_rewards.mean().item(),
            std_reward=raw_rewards.std().item() if len(raw_rewards) > 1 else 0.0,
            min_reward=raw_rewards.min().item(),
            max_reward=raw_rewards.max().item(),
            mean_kl_penalty=kl_penalty.mean().item(),
            mean_value=value_preds.mean().item(),
            mean_advantage=raw_advantages.mean().item(),
            std_advantage=raw_advantages.std().item(),
            mean_ratio=sum(all_ratios) / len(all_ratios),
            std_ratio=torch.tensor(all_ratios).std().item() if len(all_ratios) > 1 else 0.0,
            policy_lr=self.policy_scheduler.get_last_lr()[0],
            value_lr=self.value_scheduler.get_last_lr()[0],
            mean_response_length=sum(response_lengths) / len(response_lengths),
            eos_rate=eos_count / len(policy_model_outputs),
        )

        return metrics

    def log_metrics(self, step: int, metrics: PPOMetrics):
        """Log metrics to console and save to file."""
        print(f"[Step {step:5d}] {metrics}")

        # Save to JSONL file
        metrics_dict = metrics.to_dict()
        metrics_dict["step"] = step
        self.metrics_history.append(metrics_dict)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_dict) + "\n")

    def train(self):
        """Run the PPO training for the specified number of steps."""
        print(f"Starting PPO training for {self.training_config.total_steps} steps...")
        print(f"Metrics will be saved to: {self.metrics_file}")
        print("-" * 100)

        update_count = 0
        while self.train_steps_taken < self.training_config.total_steps:
            metrics = self.train_step()
            self.policy_scheduler.step()
            self.value_scheduler.step()
            update_count += 1

            # Log metrics periodically
            if update_count % self.logging_config.log_freq == 0:
                self.log_metrics(self.train_steps_taken, metrics)

            # Run evaluation
            if update_count % self.logging_config.eval_freq == 0:
                self.eval()

            # Save models periodically
            if update_count % self.logging_config.save_freq == 0 and self._should_save():
                self.save_models()
                print(
                    f"[Step {self.train_steps_taken}] Saved models to {self.logging_config.experiment_dir}"
                )

            # Sync reference model if enabled
            if (
                self.ref_model_config.update_ref_model
                and self.ref_model_config.sync_freq is not None
                and update_count % self.ref_model_config.sync_freq == 0
            ):
                self.reference_model.load_state_dict(self.policy_model.state_dict())
                print(
                    f"[Step {self.train_steps_taken}] Synchronized reference model with policy model."
                )

        print("-" * 100)
        print(f"Training complete! Total steps: {self.train_steps_taken}")
        print(f"Metrics saved to: {self.metrics_file}")

    def eval(self):
        """Evaluate the current policy model."""
        self.policy_model.eval()
        print(f"\n{'=' * 50} EVALUATION (Step {self.train_steps_taken}) {'=' * 50}")

        prompts = self.dataset[:5]  # Evaluate on a subset of prompts

        generated_outputs = [
            self.policy_model.generate(prompt=prompt, config=self.generation_config)
            for prompt in prompts
        ]
        rewards_batch = self.get_rewards(generated_outputs)
        avg_reward = rewards_batch.mean().item()
        output_strings = [g["input_text"] + g["response_text"] for g in generated_outputs]
        for i in range(len(generated_outputs)):
            response_text = output_strings[i]
            reward = rewards_batch[i].item()
            print(f"  [Reward: {reward:+.3f}] {response_text}")

        print(f"{'=' * 120}\n")
        print(f"[Evaluation] Average reward: {avg_reward:+.3f}\n")
        print(f"Positive rates: {(rewards_batch > 0).float().mean().item():.2%}\n")

    def save_models(self):
        policy_path = os.path.join(self.logging_config.experiment_dir, "policy_model.pt")
        torch.save(self.policy_model.state_dict(), policy_path)
        return

    def _should_save(self) -> bool:
        """Determine if models should be saved based on rewards since last save."""
        if not self.rewards_since_last_save:
            return False
        avg_reward = sum(self.rewards_since_last_save) / len(self.rewards_since_last_save)
        print(
            f"[Info] Average reward since last save: {avg_reward:+.3f}, last saved: {self.last_saved_mean_reward:+.3f}"
        )
        if avg_reward > self.last_saved_mean_reward:
            self.last_saved_mean_reward = avg_reward
            self.rewards_since_last_save = []
            return True
        return False


if __name__ == "__main__":
    base_model_path = "/workspace/base_models/Qwen2.5-1.5B"
    trainer = PPOTrainer(
        policy_model=PolicyQwenModel(training_enabled=True, model_path=base_model_path),
        value_model=QwenModelValueHead(
            training_enabled=True, model_path=base_model_path, freeze_backbone=False
        ),
        reference_model=ReferenceQwenModel(model_path=base_model_path),
        reward_model=SentimentRewarder(),
        ppo_config=PPOConfig(),
        training_config=TrainingConfig(),
        generation_config=GenerationConfig(),
        logging_config=LoggingConfig(),
        ref_model_config=ReferenceModelConfig(),
    )
    trainer.train()
