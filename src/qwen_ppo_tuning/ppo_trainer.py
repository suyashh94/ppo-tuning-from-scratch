import json
import math
import os
from dataclasses import dataclass

import torch
from torch.optim import lr_scheduler

from qwen_ppo_tuning.models import PolicyQwenModel, QwenModelValueHead, ReferenceQwenModel
from qwen_ppo_tuning.rewarder import SentimentRewarder
from qwen_ppo_tuning.utils import (
    RunningMoments,
    calculate_clip_fraction,
    calculate_entropy,
    calculate_value_clip_fraction,
    logprobs_from_logits,
    whiten,
)


@dataclass
class PPOMetrics:
    """Container for PPO training metrics."""

    # Losses
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    entropy: float = 0.0

    # PPO-specific
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    value_clip_fraction: float = 0.0

    # Rewards
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    # KL penalty
    mean_kl_penalty: float = 0.0

    # Value predictions
    mean_value: float = 0.0
    mean_advantage: float = 0.0
    std_advantage: float = 0.0

    # Ratios
    mean_ratio: float = 0.0
    std_ratio: float = 0.0

    # Learning rates
    policy_lr: float = 0.0
    value_lr: float = 0.0

    # Response stats
    mean_response_length: float = 0.0
    eos_rate: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        return (
            f"reward={self.mean_reward:+.3f}±{self.std_reward:.3f} | "
            f"policy_loss={self.policy_loss:.4f} | "
            f"value_loss={self.value_loss:.4f} | "
            f"kl={self.approx_kl:.4f} | "
            f"clip_frac={self.clip_fraction:.2%} | "
            f"entropy={self.entropy:.3f} | "
            f"eos_rate={self.eos_rate:.1%}"
        )


class PPOTrainer:
    def __init__(
        self,
        policy_model: PolicyQwenModel,
        value_model: QwenModelValueHead,
        reference_model: ReferenceQwenModel,
        reward_model: SentimentRewarder,
        normalize_advantages: bool = True,
        clip_epsilon: float = 0.2,
        vf_clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 1,
        gae_lambda: float = 0.95,
        num_ppo_epochs: int = 4,
        kl_coef: float = 0.02,
        target_kl: float | None = 0.015,
        adaptive_kl: bool = False,
        max_kl: float = 0.05,
        num_rollouts_per_update: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
        warmup_steps_frac: float = 0.1,
        lr_scheduler_type: str = "cosine",
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        minibatch_size: int = 4,
        total_ppo_steps: int = 10000,
        ref_model_update: bool = False,
        ref_model_sync_freq: int | None = None,
        log_freq: int = 2,
        save_freq: int = 1000,
        eval_freq: int = 500,
        output_dir: str = "./ppo_output",
        use_entropy_loss: bool | None = False,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.normalize_advantages = normalize_advantages
        self.clip_epsilon = clip_epsilon
        self.vf_clip_epsilon = vf_clip_epsilon
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.use_entropy_loss = use_entropy_loss
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_ppo_epochs = num_ppo_epochs
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.max_kl = max_kl
        self.adaptive_kl = adaptive_kl
        self.num_rollouts_per_update = num_rollouts_per_update
        self.temperature = temperature
        self.top_k = top_k
        self.warmup_steps_frac = warmup_steps_frac
        self.lr_scheduler_type = lr_scheduler_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size
        self.total_ppo_steps = total_ppo_steps
        self.ref_model_update = ref_model_update
        self.ref_model_sync_freq = ref_model_sync_freq
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.running_moments = RunningMoments()
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.train_steps_taken = 0
        self.experience_collection_counter = 0
        self.last_eval_step = 0
        self.sample_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)

        # Metrics tracking
        self.metrics_history: list[dict] = []
        self.metrics_file = os.path.join(self.output_dir, "metrics.jsonl")

        self._setup_schedulers()

    def _create_cosine_warmup_scheduler(self, optimizer, min_lr_ratio):
        """Create a cosine annealing scheduler with linear warmup."""
        warmup_steps = int(self.warmup_steps_frac * self.total_ppo_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (self.total_ppo_steps - warmup_steps)
                progress = min(progress, 1.0)  # Clamp to 1.0
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _create_linear_warmup_scheduler(self, optimizer):
        """Create a linear warmup scheduler followed by linear decay."""
        warmup_steps = int(self.warmup_steps_frac * self.total_ppo_steps)
        total_steps = self.total_ppo_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _setup_schedulers(self):
        min_lr_ratio = 0.1
        if self.lr_scheduler_type == "cosine":
            self.policy_scheduler = self._create_cosine_warmup_scheduler(
                self.policy_model.optimizer, min_lr_ratio
            )
            self.value_scheduler = self._create_cosine_warmup_scheduler(
                self.value_model.optimizer, min_lr_ratio
            )
        elif self.lr_scheduler_type == "linear":
            self.policy_scheduler = self._create_linear_warmup_scheduler(
                self.policy_model.optimizer
            )
            self.value_scheduler = self._create_linear_warmup_scheduler(self.value_model.optimizer)
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")

    def collect_experience(self):
        """Collect experience using the current policy model."""
        self.policy_model.eval()
        self.value_model.eval()
        self.experience_collection_counter += 1
        i = 0
        policy_model_outputs = []
        ref_model_outputs = []

        while i < self.num_rollouts_per_update:
            policy_model_output = self.policy_model.generate(prompt="The movie was")
            ref_model_output = self.reference_model.generate(prompt="The movie was")
            policy_model_outputs.append(policy_model_output)
            ref_model_outputs.append(ref_model_output)
            i += 1

        policy_response_properties = self.policy_model.get_batched_response_properties(
            policy_model_outputs, pad_to=128
        )
        ref_response_properties = self.reference_model.get_batched_response_properties(
            policy_model_outputs, pad_to=128
        )
        policy_response_properties["old_logprobs"] = ref_response_properties["logprobs"].detach()

        with torch.no_grad():
            value_preds = self.value_model(
                input_ids=policy_response_properties["padded_generated_ids"].to(self.device)
            )
        value_preds = value_preds[:, :-1, :].squeeze(-1).to(self.device)

        rewards = self.get_rewards(policy_model_outputs)
        kl_diff = policy_response_properties["logprobs"].to(
            self.device
        ) - policy_response_properties["old_logprobs"].to(self.device)
        kl_penalty = self.kl_coef * kl_diff
        total_rewards = rewards - kl_penalty
        total_rewards_aligned = self.collate_rewards_tensor(
            total_rewards,
            policy_response_properties["response_id_lengths"],
            policy_response_properties["response_mask"].size(1),
        )

        return {
            "policy_response_properties": policy_response_properties,
            "value_preds": value_preds,
            "rewards": total_rewards,
            "kl_penalty": kl_penalty,
            "total_rewards": total_rewards,
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

        Example:
            For a batch of 2 responses, with input ids and response ids as follows:
                input_ids = [[101, 200, 300, 102], [101, 400, 500, 600, 102]]
                generated_ids = [[201, 202, EOS_ID], [401, 402, 403, EOS_ID]]
            The response_id_lengths would be [4+3, 5+4] = [7, 9] (including input and response lengths).
            If the rewards tensor is [1.0, 0.5], and max_len is 10, the output rewards_tensor would be:
                [[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0.5,0,0]] as reward is aligned to the last token that led to generation of EOS token.
        """
        batch_size = rewards.size(0)
        rewards_tensor = torch.zeros((batch_size, max_len), device=rewards.device)

        for i in range(batch_size):
            response_length = response_id_lengths[i]
            rewards_tensor[i, response_length - 1] = rewards[i]

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

            delta = rewards[:, t] + self.gamma * next_values * next_non_terminal - values[:, t]
            advantages[:, t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
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

        if self.normalize_advantages:
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

        for epoch in range(self.num_ppo_epochs):
            epoch_kl_divs = []
            for start_idx in range(0, self.num_rollouts_per_update, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                mb_indices = torch.arange(start_idx, end_idx)

                mb_padded_generated_ids = policy_response_properties["padded_generated_ids"][
                    mb_indices
                ].to(self.device)
                mb_response_mask = policy_response_properties["response_mask"][mb_indices].to(
                    self.device
                )
                mb_old_logprobs = policy_response_properties["logprobs"][mb_indices].to(self.device)
                mb_advantages = advantages[mb_indices]
                mb_value_preds = value_preds[mb_indices]

                # Forward pass with mixed precision
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    # Forward pass through policy model
                    logits = self.policy_model(
                        input_ids=mb_padded_generated_ids,
                    )
                    logits = logits[:, :-1, :]

                    logprobs = logprobs_from_logits(logits, labels=mb_padded_generated_ids[:, 1:])

                    # Calculate ratios
                    ratios = torch.exp(logprobs - mb_old_logprobs)

                    # Policy loss
                    surr1 = ratios * mb_advantages
                    surr2 = (
                        torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2)
                    policy_loss = (policy_loss * mb_response_mask).sum() / mb_response_mask.sum()

                    # Value loss
                    values = self.value_model(input_ids=mb_padded_generated_ids).squeeze(-1)
                    values = values[:, :-1]
                    value_pred_clipped = mb_value_preds + torch.clamp(
                        values - mb_value_preds, -self.vf_clip_epsilon, self.vf_clip_epsilon
                    )
                    returns = mb_advantages.detach() + mb_value_preds.detach()
                    value_losses1 = (values - returns) ** 2
                    value_losses2 = (value_pred_clipped - returns) ** 2
                    value_loss = torch.max(value_losses1, value_losses2)
                    value_loss = (value_loss * mb_response_mask).sum() / mb_response_mask.sum()

                    # Entropy (always compute for logging)
                    entropy = calculate_entropy(logits, mb_response_mask)
                    entropy_loss = -self.entropy_coef * entropy if self.use_entropy_loss else 0.0

                    total_loss = policy_loss + self.vf_coef * value_loss + entropy_loss

                self.policy_model.optimizer.zero_grad()
                self.value_model.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.policy_model.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
                self.scaler.step(self.policy_model.optimizer)
                self.scaler.step(self.value_model.optimizer)
                self.scaler.update()
                self.train_steps_taken += 1

                # Collect metrics (detached)
                with torch.no_grad():
                    approx_kl = (
                        (mb_old_logprobs - logprobs) * mb_response_mask
                    ).sum() / mb_response_mask.sum()
                    clip_frac = calculate_clip_fraction(ratios, self.clip_epsilon)
                    value_clip_frac = calculate_value_clip_fraction(
                        values, mb_value_preds, self.vf_clip_epsilon
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

            self.policy_scheduler.step()
            self.value_scheduler.step()

            # Check for early stopping based on epoch KL
            epoch_kl = sum(epoch_kl_divs) / len(epoch_kl_divs)
            if epoch_kl > self.max_kl:
                break

        # Compute aggregated metrics
        metrics = PPOMetrics(
            policy_loss=sum(all_policy_losses) / len(all_policy_losses),
            value_loss=sum(all_value_losses) / len(all_value_losses),
            total_loss=(sum(all_policy_losses) + self.vf_coef * sum(all_value_losses))
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
        print(f"Starting PPO training for {self.total_ppo_steps} steps...")
        print(f"Metrics will be saved to: {self.metrics_file}")
        print("-" * 100)

        update_count = 0
        while self.train_steps_taken < self.total_ppo_steps:
            metrics = self.train_step()
            update_count += 1

            # Log metrics periodically
            if update_count % self.log_freq == 0:
                self.log_metrics(self.train_steps_taken, metrics)

            # Sync reference model if enabled
            if (
                self.ref_model_update
                and self.ref_model_sync_freq is not None
                and update_count % self.ref_model_sync_freq == 0
            ):
                self.reference_model.load_state_dict(self.policy_model.state_dict())
                print(
                    f"[Step {self.train_steps_taken}] Synchronized reference model with policy model."
                )

            # Run evaluation
            if self.train_steps_taken - self.last_eval_step >= self.eval_freq:
                self.eval()
                self.last_eval_step = self.train_steps_taken

        print("-" * 100)
        print(f"Training complete! Total steps: {self.train_steps_taken}")
        print(f"Metrics saved to: {self.metrics_file}")

    def eval(self):
        """Evaluate the current policy model."""
        self.policy_model.eval()
        print(f"\n{'=' * 50} EVALUATION (Step {self.train_steps_taken}) {'=' * 50}")

        prompts = ["The movie was"] * 5
        rewards = []
        sentiments = []

        for i, prompt in enumerate(prompts):
            generated_output = self.policy_model.generate(prompt=prompt)
            full_text = generated_output["input_text"] + generated_output["response_text"]
            reward_info = self.reward_model.get_reward(full_text)

            rewards.append(reward_info["reward"])
            sentiments.append(reward_info["label"])

            print(f"  [{i + 1}] {prompt}{generated_output['response_text'][:80]}...")
            print(f"      → sentiment={reward_info['label']}, reward={reward_info['reward']:+.3f}")

        avg_reward = sum(rewards) / len(rewards)
        positive_rate = sum(1 for s in sentiments if s == "positive") / len(sentiments)

        print(f"\n  Summary: avg_reward={avg_reward:+.3f}, positive_rate={positive_rate:.0%}")
        print("=" * 110 + "\n")


if __name__ == "__main__":
    base_model_path = "/workspace/base_models/Qwen2.5-1.5B"
    trainer = PPOTrainer(
        policy_model=PolicyQwenModel(training_enabled=True, model_path=base_model_path),
        value_model=QwenModelValueHead(training_enabled=True, model_path=base_model_path),
        reference_model=ReferenceQwenModel(model_path=base_model_path),
        reward_model=SentimentRewarder(),
    )
    trainer.train()
