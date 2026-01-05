import inspect
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchtune.models.qwen2_5 import qwen2_5_1_5b_base
from torchtune.training import FullModelHFCheckpointer
from transformers import AutoTokenizer

from qwen_ppo_tuning.config import GenerationConfig
from qwen_ppo_tuning.utils import logprobs_from_logits, pad_sequences


class SetupQwenModel(nn.Module):
    def __init__(self, model_path: str, model_type: str = "QWEN2", **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_path = model_path
        self.block_size = 128000
        self.model_type = model_type
        self.training_enabled = kwargs.get("training_enabled", False)
        self.setup_model(**kwargs if kwargs else {})

    def setup_model(self, **kwargs):
        lr = kwargs.get("lr", 1e-06)
        weight_decay = kwargs.get("weight_decay", 0.0)
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        self.model = qwen2_5_1_5b_base()
        ckpt_files = sorted(
            f for f in os.listdir(self.base_model_path) if f.endswith(".safetensors")
        )
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=self.base_model_path,
            checkpoint_files=ckpt_files,
            model_type=self.model_type,
            output_dir=str(Path(self.base_model_path).parent / "models_output"),
        )
        state = checkpointer.load_checkpoint()["model"]
        self.model.load_state_dict(state)
        del state
        self.model.to(self.device).eval()
        self.optimizer = (
            self.configure_optimizer(
                lr,
                weight_decay,
                (beta1, beta2),
                device_type="cuda" if torch.cuda.is_available() else "cpu",
            )
            if self.training_enabled
            else None
        )

    def configure_optimizer(self, lr, weight_decay, betas, device_type="cuda"):
        params_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in params_dict.items() if p.dim() > 1]
        no_decay_params = [p for _, p in params_dict.items() if p.dim() <= 1]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"Optimizer groups: {num_decay_params} decay params with {len(decay_params)} tensors,\n {num_no_decay_params} no_decay params with {len(no_decay_params)} tensors."
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        return optimizer

    def enable_training(self, **kwargs):
        self.training_enabled = True
        self.model.train()
        if self.optimizer is None:
            self.optimizer = self.configure_optimizer(
                lr=kwargs.get("lr", 1e-06),
                weight_decay=kwargs.get("weight_decay", 0.0),
                betas=(kwargs.get("beta1", 0.9), kwargs.get("beta2", 0.999)),
                device_type="cuda" if torch.cuda.is_available() else "cpu",
            )


class QwenModel(SetupQwenModel):
    def __init__(self, model_path: str, model_type: str = "QWEN2", **kwargs):
        super().__init__(model_path=model_path, model_type=model_type, **kwargs)

    def forward(self, input_ids: torch.LongTensor):
        logits = self.model(input_ids)
        return logits

    def _delete_kv_caches(self):
        """Delete KV caches from all attention layers to allow normal forward passes."""
        for layer in self.model.layers:
            if hasattr(layer, "attn"):
                if hasattr(layer.attn, "kv_cache"):
                    layer.attn.kv_cache = None
                if hasattr(layer.attn, "cache_enabled"):
                    layer.attn.cache_enabled = False
        # Reset the cache seq len trackers
        self.model.decoder_max_cache_seq_len = None
        self.model.encoder_max_cache_seq_len = None

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        temperature: float,
        top_k: int | None,
        repetition_penalty: float,
        sample_max: bool,
    ) -> torch.Tensor:
        """Sample the next token from logits with various sampling strategies.

        Args:
            logits: Raw logits for the next token (1, vocab_size).
            generated_tokens: All tokens generated so far for repetition penalty.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            repetition_penalty: Repetition penalty factor.
            sample_max: Whether to use argmax instead of sampling.

        Returns:
            The sampled next token (1,).
        """
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

        # Handle numerical stability issues
        next_token_logits = torch.nan_to_num(
            next_token_logits,
            nan=0.0,
            posinf=1e20,
            neginf=-1e20,
        )

        # Sample or argmax
        if sample_max:
            next_token = torch.argmax(next_token_logits, dim=-1)
        else:
            probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
            probabilities = torch.clamp(probabilities, min=1e-20)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        return next_token

    def _generate_from_tensor(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        sample_max: bool = False,
        repetition_penalty: float = 1.0,
        return_logits: bool = False,
        use_kv_cache: bool = True,
    ):
        """Generate tokens from input tensor.

        Args:
            input_ids (torch.LongTensor): Input tensor of shape (1, sequence_length).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int | None): Top-k sampling parameter.
            sample_max (bool): Whether to sample the maximum probability token.
            repetition_penalty (float): Repetition penalty factor.
            return_logits (bool): Whether to return logits along with generated tokens.
            use_kv_cache (bool): Whether to use KV caching for faster generation.

        """
        if use_kv_cache:
            return self._generate_with_kv_cache(
                input_ids, max_new_tokens, temperature, top_k,
                sample_max, repetition_penalty, return_logits
            )
        else:
            return self._generate_without_cache(
                input_ids, max_new_tokens, temperature, top_k,
                sample_max, repetition_penalty, return_logits
            )

    def _generate_without_cache(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        sample_max: bool,
        repetition_penalty: float,
        return_logits: bool,
    ):
        """Generate tokens without KV caching (original implementation)."""
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_len = input_ids.size(1)
        total_len = min(input_len + max_new_tokens, self.block_size)
        max_new_tokens = total_len - input_len

        generated_tokens = torch.zeros((1, total_len), dtype=torch.long, device=self.device)
        generated_tokens[0, :input_len] = input_ids[0]

        generation_logits = []
        end_token_found = False
        n_tokens_generated = 0
        current_len = input_len

        with torch.no_grad():
            while not end_token_found and n_tokens_generated < max_new_tokens:
                # Use only the valid portion of the buffer
                valid_tokens = generated_tokens[:, :current_len]
                context_input = (
                    valid_tokens[:, -self.block_size :]
                    if current_len > self.block_size
                    else valid_tokens
                )

                logits = self.model(context_input)
                next_logits = logits[:, -1, :]

                if return_logits:
                    generation_logits.append(
                        (next_logits / temperature).clone().detach().cpu()
                    )

                next_token = self._sample_next_token(
                    next_logits,
                    valid_tokens,
                    temperature,
                    top_k,
                    repetition_penalty,
                    sample_max,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    end_token_found = True

                generated_tokens[0, current_len] = next_token
                current_len += 1
                n_tokens_generated += 1

        final_tokens = generated_tokens[:, :current_len]
        max_tokens_generated = n_tokens_generated >= max_new_tokens

        if return_logits:
            return {
                "response_ids": final_tokens,
                "logits": torch.stack(generation_logits, dim=1),
                "max_tokens_generated": max_tokens_generated,
                "end_token_found": end_token_found,
            }

        return {
            "response_ids": final_tokens,
            "max_tokens_generated": max_tokens_generated,
            "end_token_found": end_token_found,
        }

    def _generate_with_kv_cache(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        sample_max: bool,
        repetition_penalty: float,
        return_logits: bool,
    ):
        """Generate tokens using KV caching for efficiency."""
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        input_len = input_ids.size(1)
        total_len = min(input_len + max_new_tokens, self.block_size)
        max_new_tokens = total_len - input_len

        generated_tokens = torch.zeros((1, total_len), dtype=torch.long, device=self.device)
        generated_tokens[0, :input_len] = input_ids[0]

        generation_logits = []
        end_token_found = False
        n_tokens_generated = 0
        current_len = input_len

        with torch.no_grad():
            model_dtype = next(self.model.parameters()).dtype

            self.model.setup_caches(
                batch_size=1,
                dtype=model_dtype,
                decoder_max_seq_len=total_len,
            )
            self.model.to(self.device)

            try:
                causal_mask = torch.tril(
                    torch.ones(total_len, total_len, dtype=torch.bool, device=self.device)
                )

                # === Prefill phase ===
                prompt_positions = torch.arange(0, input_len, device=self.device).unsqueeze(0)
                prefill_mask = causal_mask[:input_len, :].unsqueeze(0)

                logits = self.model(input_ids, input_pos=prompt_positions, mask=prefill_mask)
                next_logits = logits[:, -1, :]

                if return_logits:
                    generation_logits.append((next_logits / temperature).clone().detach().cpu())

                next_token = self._sample_next_token(
                    next_logits,
                    generated_tokens[:, :current_len],
                    temperature,
                    top_k,
                    repetition_penalty,
                    sample_max,
                )

                if next_token.item() == self.tokenizer.eos_token_id:
                    end_token_found = True

                generated_tokens[0, current_len] = next_token
                current_len += 1
                n_tokens_generated += 1

                # === Decode phase ===
                while not end_token_found and n_tokens_generated < max_new_tokens:
                    new_token = generated_tokens[:, current_len - 1 : current_len]
                    new_pos = torch.tensor([[current_len - 1]], device=self.device)
                    decode_mask = causal_mask[current_len - 1, :].unsqueeze(0).unsqueeze(0)

                    logits = self.model(new_token, input_pos=new_pos, mask=decode_mask)
                    next_logits = logits[:, -1, :]

                    if return_logits:
                        generation_logits.append(
                            (next_logits / temperature).clone().detach().cpu()
                        )

                    next_token = self._sample_next_token(
                        next_logits,
                        generated_tokens[:, :current_len],
                        temperature,
                        top_k,
                        repetition_penalty,
                        sample_max,
                    )

                    if next_token.item() == self.tokenizer.eos_token_id:
                        end_token_found = True

                    generated_tokens[0, current_len] = next_token
                    current_len += 1
                    n_tokens_generated += 1

            finally:
                self._delete_kv_caches()

        final_tokens = generated_tokens[:, :current_len]
        max_tokens_generated = n_tokens_generated >= max_new_tokens

        if return_logits:
            return {
                "response_ids": final_tokens,
                "logits": torch.stack(generation_logits, dim=1),
                "max_tokens_generated": max_tokens_generated,
                "end_token_found": end_token_found,
            }

        return {
            "response_ids": final_tokens,
            "max_tokens_generated": max_tokens_generated,
            "end_token_found": end_token_found,
        }

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        sample_max: bool = False,
        return_logits: bool = False,
    ):
        """Generate tokens from a text prompt.

        Args:
            prompt (str): Input text prompt.
            config (GenerationConfig | None): Generation configuration. Uses defaults if None.
            sample_max (bool): Whether to sample the maximum probability token.
            return_logits (bool): Whether to return logits along with generated tokens.

        """
        config = config or GenerationConfig()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        res = self._generate_from_tensor(
            input_ids=input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            sample_max=sample_max,
            repetition_penalty=config.repetition_penalty,
            return_logits=return_logits,
            use_kv_cache=config.use_kv_cache,
        )
        generated_ids = res["response_ids"][0][len(input_ids[0]) :]
        response_text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        input_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        res["input_text"] = input_text  # string
        res["response_text"] = response_text  # string
        res["input_length"] = len(input_ids[0])  # int
        res["generated_response_length"] = len(generated_ids)  # int
        return res

    def get_batched_response_properties(self, generated_outputs_list: list[dict], **kwargs):
        """Get response properties from a batch of generated outputs.

        Args:
            generated_outputs_list (list[dict]): List of generated output dictionaries.
        Returns:
            dict: Dictionary containing lists of response properties.
            Response properties include:
                - log_probs : Log probabilities of the generated tokens.
                - logits: Logits of the generated tokens.
                - prompt_mask: Mask indicating prompt tokens.
                - response_mask: Mask indicating response tokens.
                - padding_mask: Mask indicating padding tokens.

        Example:
            - For one sample, we have the entire sequence, consisting of prompt tokens and response tokens.
            - generated ids have shape (1, prompt_length + response_length)
            - input_length is the length of the prompt tokens
            - We pad the generated ids to the same length across the batch.
            - On the basis of "tokens" that will play a role in PPO, we create masks for prompt, response, and padding.

            example:
                - generated_ids: [101, 2003, 2023, 102, 2054, 2003, 1996, 2562, 102]
                - input_length: 4 (length of prompt tokens)
                - response_length: 5 (length of response tokens)
                - padded to length of 12 ; padding token is 0
                - padded_generated_ids: [101, 2003, 2023, 102, 2054, 2003, 1996, 2562, 102, 0, 0, 0]
                - Our response properties will be of shape 1 X 11 and not 1 X 12 as first token is not considered for PPO (no action taken before first token)
                - prompt_mask will be [1,1,1,0,0,0,0,0,0,0,0,] and not [1,1,1,1,0,0,0,0,0,0,0,] because response starts when last prompt token is input; Hence, that action is accounted for in PPO.
                - reponse_mask will be 1 - prompt_mask
                - padding_mask will be [0,0,0,0,0,0,0,0,1,1,1]
        """
        pad_token_id = self.tokenizer.pad_token_id
        input_lengths = [g["input_length"] for g in generated_outputs_list]
        generated_ids = [g["response_ids"][0].tolist() for g in generated_outputs_list]
        response_id_lengths = [len(g["response_ids"][0]) for g in generated_outputs_list]

        padded_generated_ids = pad_sequences(
            generated_ids, pad_value=pad_token_id, padding="right", **kwargs
        )
        padded_generated_ids_tensor = torch.LongTensor(padded_generated_ids).to(self.device)
        with torch.no_grad():
            logits = self.model(padded_generated_ids_tensor)
            logits = logits[:, :-1, :]  # align logits with labels

        log_probs = logprobs_from_logits(
            logits, labels=padded_generated_ids_tensor[:, 1:]
        ).cpu()  # ignored first token as that is not sampled in generation
        batch_size, seq_len = padded_generated_ids_tensor.size()
        prompt_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
        response_mask = torch.zeros((batch_size, seq_len - 1), dtype=torch.bool)
        for i in range(batch_size):
            prompt_length = input_lengths[i]
            if prompt_length > 0:
                prompt_mask[i, : prompt_length - 1] = 1
            response_mask[i, prompt_length - 1 : response_id_lengths[i] - 1] = 1

        padding_mask = padded_generated_ids_tensor.eq(pad_token_id).cpu()
        padding_mask = padding_mask[:, : seq_len - 1]

        return {
            "logprobs": log_probs,  # shape (batch_size, seq_len - 1)
            "logits": logits.cpu(),  # shape (batch_size, seq_len - 1, vocab_size)
            "prompt_mask": prompt_mask,  # shape (batch_size, seq_len - 1)
            "response_mask": response_mask,  # shape (batch_size, seq_len - 1)
            "padding_mask": padding_mask,  # shape (batch_size, seq_len - 1)
            "padded_generated_ids": padded_generated_ids_tensor,  # shape (batch_size, seq_len)
            "response_id_lengths": response_id_lengths,  # list of ints
        }


class QwenModelValueHead(SetupQwenModel):
    def __init__(self, model_path: str, model_type: str = "QWEN2", **kwargs):
        # Extract and remove training_enabled to prevent parent from creating optimizer
        training_enabled = kwargs.pop("training_enabled", False)
        freeze_backbone = kwargs.pop("freeze_backbone", False)

        # Pass training_enabled=False to parent to prevent premature optimizer creation
        super().__init__(model_path=model_path, model_type=model_type, training_enabled=False, **kwargs)

        self.hidden_size = self._get_hidden_size()
        self.value_head = nn.Linear(self.hidden_size, 1)
        self.value_head.to(self.device)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Now configure optimizer to include value_head parameters
        if training_enabled:
            lr = kwargs.get("lr", 1e-06)
            weight_decay = kwargs.get("weight_decay", 0.0)
            beta1 = kwargs.get("beta1", 0.9)
            beta2 = kwargs.get("beta2", 0.999)
            self.training_enabled = True
            self.model.train()
            self.optimizer = self._configure_optimizer(lr, weight_decay, (beta1, beta2))

    def _configure_optimizer(self, lr, weight_decay, betas):
        """Configure optimizer including both model and value_head parameters."""
        import inspect

        # Get model parameters
        model_params = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in model_params.items() if p.dim() > 1]
        no_decay_params = [p for _, p in model_params.items() if p.dim() <= 1]

        head_params = {pn: p for pn, p in self.value_head.named_parameters() if p.requires_grad}
        head_decay_params = [p for _, p in head_params.items() if p.dim() > 1]
        head_no_decay_params = [p for _, p in head_params.items() if p.dim() <= 1]

        decay_params.extend(head_decay_params)
        no_decay_params.extend(head_no_decay_params)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(
            f"Value model optimizer: {num_decay} decay params, {num_no_decay} no_decay params (includes value_head)"
        )

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)

    def forward(self, input_ids: torch.LongTensor):
        hidden_states = self.get_last_hidden_state(input_ids.to(self.device))
        values = self.value_head(hidden_states)
        return values

    def _get_hidden_size(self):
        if hasattr(self.model, "tok_embeddings"):
            hidden_size = self.model.tok_embeddings.weight.size(1)
        elif hasattr(self.model, "embed_tokens"):
            hidden_size = self.model.embed_tokens.weight.size(1)
        else:
            raise ValueError("Cannot determine hidden size from the model architecture.")
        return hidden_size

    def get_last_hidden_state(self, input_ids: torch.LongTensor):
        if input_ids.size(1) > self.model.max_seq_len:
            raise ValueError(
                f"Input sequence length {input_ids.size(1)} exceeds model's maximum sequence length {self.model.max_seq_len}."
            )
        hidden_state = self.model.tok_embeddings(input_ids)
        for layer in self.model.layers:
            hidden_state = layer(hidden_state)
        hidden_state = self.model.norm(hidden_state)
        return hidden_state


class ReferenceQwenModel(QwenModel):
    def __init__(self, model_path: str, model_type: str = "QWEN2", **kwargs):
        super().__init__(model_path=model_path, model_type=model_type, **kwargs)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()


class PolicyQwenModel(QwenModel):
    def __init__(self, model_path: str, model_type: str = "QWEN2", **kwargs):
        super().__init__(model_path=model_path, model_type=model_type, **kwargs)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()


if __name__ == "__main__":
    model = QwenModel(model_path="/workspace/base_models/Qwen2.5-1.5B")
    print("Model and optimizer initialized successfully.")

    # Test with KV cache enabled (default)
    gen_config_cached = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        use_kv_cache=True,
    )

    # Test without KV cache
    gen_config_no_cache = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        use_kv_cache=False,
    )

    prompt = "Explain the theory of relativity in simple terms."

    print("\n=== Generation with KV cache ===")
    output_cached = model.generate(prompt, config=gen_config_cached)
    print("Prompt:", output_cached["input_text"])
    print("Response:", output_cached["response_text"])

    print("\n=== Generation without KV cache ===")
    output_no_cache = model.generate(prompt, config=gen_config_no_cache)
    print("Prompt:", output_no_cache["input_text"])
    print("Response:", output_no_cache["response_text"])

    # Test batched response properties
    generation_outputs = [output_cached, output_no_cache]
    response_properties = model.get_batched_response_properties(generation_outputs, pad_to=256)
    print("\nBatched response properties computed successfully.")

    value_model = QwenModelValueHead(model_path="/workspace/base_models/Qwen2.5-1.5B")
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    values = value_model(dummy_input_ids)
    print("Value head output shape:", values.shape)  # Expected shape: (1, 5, 1)
