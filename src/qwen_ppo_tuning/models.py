import inspect
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchtune.models.qwen2_5 import qwen2_5_1_5b_base
from torchtune.training import FullModelHFCheckpointer
from transformers import AutoTokenizer


class SetupQwenModel(nn.Module):
    def __init__(self, model_path: str, model_type: str = "QWEN2", **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_path = model_path
        self.block_size = 128000
        self.model_type = model_type
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
        self.model.to(self.device).eval()
        self.optimizer = self.configure_optimizer(
            lr,
            weight_decay,
            (beta1, beta2),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
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
        generated_ids = input_ids.clone()

        while not end_token_found and not max_tokens_generated:
            context_input = (
                generated_ids[:, -self.block_size :]
                if generated_ids.size(1) > self.block_size
                else generated_ids
            )
            logits = self.model(context_input)
            next_token_logits = logits[:, -1, :] / temperature
            if return_logits:
                generation_logits.append(next_token_logits.clone().detach().cpu())

            if repetition_penalty != 1.0:
                for token_id in set(generated_ids.view(-1).tolist()):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty

            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            if sample_max:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                end_token_found = True

            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            n_tokens_generated += 1
            if n_tokens_generated > max_new_tokens:
                max_tokens_generated = True

        if return_logits:
            return {
                "generated_ids": generated_ids,
                "logits": torch.stack(generation_logits, dim=1),
                "max_tokens_generated": max_tokens_generated,
                "end_token_found": end_token_found,
            }

        return {
            "generated_ids": generated_ids,
            "max_tokens_generated": max_tokens_generated,
            "end_token_found": end_token_found,
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
        response_ids = res["generated_ids"][0][len(input_ids[0]) :]
        response_text = self.tokenizer.decode(response_ids.tolist(), skip_special_tokens=True)
        input_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        res["input_text"] = input_text
        res["response_text"] = response_text
        res["input_length"] = len(input_ids[0])
        res["response_length"] = len(response_ids)
        return res


if __name__ == "__main__":
    model = QwenModel(model_path="/workspace/base_models/Qwen2.5-1.5B")
    print("Model and optimizer initialized successfully.")

    prompt = "Explain the theory of relativity in simple terms."
    generation_output = model.generate(
        prompt,
        max_new_tokens=500,
        temperature=0.7,
        top_k=50,
        sample_max=False,
        repetition_penalty=1.2,
        return_logits=False,
    )
    print("Prompt:", generation_output["input_text"])
    print("Response:", generation_output["response_text"])
