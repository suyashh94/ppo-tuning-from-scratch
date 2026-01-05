from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Core PPO algorithm hyperparameters."""

    clip_epsilon: float = 0.2
    vf_clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    use_entropy_loss: bool = False
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    num_ppo_epochs: int = 4
    kl_coef: float = 0.02
    target_kl: float | None = 0.015
    adaptive_kl: bool = False
    max_kl: float = 0.05


@dataclass
class TrainingConfig:
    """Training loop parameters."""

    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps_frac: float = 0.1
    lr_scheduler_type: str = "cosine"
    total_steps: int = 10000
    num_rollouts_per_update: int = 8
    minibatch_size: int = 4


@dataclass
class GenerationConfig:
    """Sampling parameters for text generation."""

    temperature: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 50
    repetition_penalty: float = 1.0
    use_kv_cache: bool = False


@dataclass
class LoggingConfig:
    """Logging, saving, and evaluation frequencies."""

    log_freq: int = 2
    save_freq: int = 50
    eval_freq: int = 10
    output_dir: str = "./ppo_outputs"
    experiment_name: str = "kv-cache"

    @property
    def experiment_dir(self) -> str:
        return f"{self.output_dir}/{self.experiment_name}"


@dataclass
class ReferenceModelConfig:
    """Reference model synchronization settings."""

    update_ref_model: bool = False
    sync_freq: int | None = None


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
        return {k: v for k, v in self.__dict__.items()}  # noqa: C416

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
