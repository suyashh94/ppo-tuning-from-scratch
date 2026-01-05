import inspect
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchtune.models.qwen2_5 import qwen2_5_1_5b_base
from torchtune.training import FullModelHFCheckpointer
from transformers import AutoTokenizer

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
        decay_params = [p for n, p in params_dict.items() if p.dim() > 1]
        no_decay_params = [p for n, p in params_dict.items() if p.dim() <= 1]
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

    def _generate_from_tensor(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        sample_max: bool = False,
        repetition_penalty: float = 1.0,
        return_logits: bool = False,
    ):
        """Generate tokens from input tensor belonging to 1 tokenized prompt.

        Args:
            input_ids (torch.LongTensor): Input tensor of shape (1, sequence_length).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int | None): Top-k sampling parameter.
            sample_max (bool): Whether to sample the maximum probability token.
            repetition_penalty (float): Repetition penalty factor.
            return_logits (bool): Whether to return logits along with generated tokens.

        """

        # Ensure input is on the correct device
        input_ids = input_ids.to(self.device)
        # Ensure shape is (1, sequence_length)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        end_token_found = False
        max_tokens_generated = False
        n_tokens_generated = 0
        generation_logits = []
        input_len = input_ids.size(1)
        total_len = input_len + max_new_tokens

        # Pre-allocate buffer for generated tokens
        generated_tokens = torch.zeros((1, total_len), dtype=torch.long, device=self.device)
        generated_tokens[0, :input_len] = input_ids[0]
        current_len = input_len

        with torch.no_grad():
            while not end_token_found and not max_tokens_generated:
                # Use only the valid portion of the buffer
                valid_tokens = generated_tokens[:, :current_len]
                context_input = (
                    valid_tokens[:, -self.block_size :]
                    if current_len > self.block_size
                    else valid_tokens
                )

                logits = self.model(context_input)
                next_token_logits = logits[:, -1, :] / temperature
                if return_logits:
                    generation_logits.append(next_token_logits.clone().detach().cpu())

                if repetition_penalty != 1.0:
                    for token_id in set(valid_tokens.view(-1).tolist()):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty

                if top_k is not None:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                # Handle numerical stability issues
                # Replace NaN/Inf with very small/large finite values
                next_token_logits = torch.nan_to_num(
                    next_token_logits,
                    nan=0.0,
                    posinf=1e20,
                    neginf=-1e20,
                )

                if sample_max:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                else:
                    probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    # Clamp probabilities to ensure they're valid for multinomial
                    probabilities = torch.clamp(probabilities, min=1e-20)
                    # Renormalize after clamping
                    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    end_token_found = True

                # Write to pre-allocated buffer instead of concatenating
                generated_tokens[0, current_len] = next_token
                current_len += 1
                n_tokens_generated += 1

                if n_tokens_generated >= max_new_tokens:
                    max_tokens_generated = True

        # Return only the valid portion of the buffer
        final_tokens = generated_tokens[:, :current_len]

        if return_logits:
            return {
                "response_ids": final_tokens,
                "logits": torch.stack(generation_logits, dim=1),
                "max_tokens_generated": max_tokens_generated,
                "end_token_found": end_token_found,
            }

        return {
            "response_ids": final_tokens,  # 1 x (input_len + n_generated)
            "max_tokens_generated": max_tokens_generated,  # bool
            "end_token_found": end_token_found,  # bool
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        sample_max: bool = False,
        repetition_penalty: float = 1.0,
        return_logits: bool = False,
    ):
        """Generate tokens from a text prompt.

        Args:
            prompt (str): Input text prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int | None): Top-k sampling parameter.
            sample_max (bool): Whether to sample the maximum probability token.
            repetition_penalty (float): Repetition penalty factor.
            return_logits (bool): Whether to return logits along with generated tokens.

        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        res = self._generate_from_tensor(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            sample_max=sample_max,
            repetition_penalty=repetition_penalty,
            return_logits=return_logits,
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
        # Don't create optimizer in parent init - we'll do it after adding value_head
        training_enabled = kwargs.get("training_enabled", False)
        kwargs["training_enabled"] = False
        super().__init__(model_path=model_path, model_type=model_type, **kwargs)

        self.hidden_size = self._get_hidden_size()
        self.value_head = nn.Linear(self.hidden_size, 1)
        self.value_head.to(self.device)

        # Now configure optimizer to include value_head parameters
        if training_enabled:
            self.training_enabled = True
            self.model.train()
            self.optimizer = self._configure_optimizer_with_value_head(
                lr=kwargs.get("lr", 1e-06),
                weight_decay=kwargs.get("weight_decay", 0.0),
                betas=(kwargs.get("beta1", 0.9), kwargs.get("beta2", 0.999)),
            )

    def _configure_optimizer_with_value_head(self, lr, weight_decay, betas):
        """Configure optimizer including both model and value_head parameters."""
        import inspect

        # Get model parameters
        model_params = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in model_params.items() if p.dim() > 1]
        no_decay_params = [p for n, p in model_params.items() if p.dim() <= 1]

        # Add value_head parameters (value_head.weight has dim > 1, bias has dim 1)
        for name, param in self.value_head.named_parameters():
            if param.requires_grad:
                if param.dim() > 1:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"Value model optimizer: {num_decay} decay params, {num_no_decay} no_decay params (includes value_head)")

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

    prompts = ["Explain the theory of relativity in simple terms.", "How do airplanes fly?"]
    generation_outputs = [
        model.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            sample_max=False,
            repetition_penalty=1.2,
            return_logits=False,
        )
        for prompt in prompts
    ]

    for generation_output in generation_outputs:
        print("Prompt:", generation_output["input_text"])
        print("Response:", generation_output["response_text"])

    response_properties = model.get_batched_response_properties(generation_outputs, pad_to=256)
    print("Batched response properties computed successfully.")

    value_model = QwenModelValueHead(model_path="/workspace/base_models/Qwen2.5-1.5B")
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    values = value_model(dummy_input_ids)
    print("Value head output shape:", values.shape)  # Expected shape: (1, 5, 1)
