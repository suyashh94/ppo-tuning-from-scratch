"""
Gradio app for comparing Base vs PPO-Aligned Qwen2.5-1.5B models.

This demo showcases the effect of PPO training with sentiment-based rewards.
The aligned model was trained to generate SHORT, POSITIVE responses.

NOTE: Models are loaded sequentially (one at a time) to work on CPU with limited memory.
"""

import gc
import os

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from models import GenerationConfig, QwenModel
from rewarder import SentimentRewarder


# ============================================================================
# Configuration
# ============================================================================

# Model paths
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"

# For HuggingFace Spaces
ALIGNED_WEIGHTS_REPO = os.environ.get("ALIGNED_WEIGHTS_REPO", "suyash94/qwen-ppo-sentiment-aligned")
ALIGNED_WEIGHTS_FILE = os.environ.get("ALIGNED_WEIGHTS_FILE", "policy_model.pt")

# Local paths for testing
LOCAL_ALIGNED_WEIGHTS = "/workspace/ppo_outputs/kv-cache/policy_model.pt"
LOCAL_BASE_MODEL = "/workspace/base_models/Qwen2.5-1.5B"

# Force CPU for memory efficiency
DEVICE = torch.device("cpu")

# Generation config (same as training)
GEN_CONFIG = GenerationConfig(
    temperature=1.0,
    top_k=50,
    max_new_tokens=50,
    repetition_penalty=1.0,
)

# Example prompts
EXAMPLE_PROMPTS = [
    "I think the book was",
    "The movie was",
    "Overall, the product is",
    "The restaurant experience was",
    "The service at the hotel was",
]


# ============================================================================
# Sequential Model Loading Utilities
# ============================================================================

def get_base_model_path():
    """Get the path for base model (local or HuggingFace)."""
    if os.path.exists(LOCAL_BASE_MODEL):
        return LOCAL_BASE_MODEL
    return BASE_MODEL_ID


def get_aligned_weights_path():
    """Get the path for aligned weights (local or download from Hub)."""
    if os.path.exists(LOCAL_ALIGNED_WEIGHTS):
        return LOCAL_ALIGNED_WEIGHTS

    # Download from HuggingFace Hub
    try:
        return hf_hub_download(
            repo_id=ALIGNED_WEIGHTS_REPO,
            filename=ALIGNED_WEIGHTS_FILE,
        )
    except Exception as e:
        print(f"Could not download aligned weights: {e}")
        return None


def clear_memory():
    """Clear GPU/CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_reward_display(reward_info: dict, eos_found: bool, final_reward: float) -> str:
    """Format reward information for display."""
    sentiment = reward_info["label"].upper()
    confidence = reward_info["confidence"]
    sentiment_reward = reward_info["reward"]

    if eos_found:
        return f"""Sentiment: {sentiment}
Confidence: {confidence:.1%}
Sentiment Reward: {sentiment_reward:+.2f}
EOS Token: Found
----------------
FINAL REWARD: {final_reward:+.2f}"""
    else:
        return f"""Sentiment: {sentiment}
Confidence: {confidence:.1%}
Sentiment Reward: {sentiment_reward:+.2f} (ignored)
EOS Token: Missing
----------------
FINAL REWARD: -10.00"""


# ============================================================================
# Main Generation Function (Generator for Progressive Updates)
# ============================================================================

def generate_comparison(prompt: str):
    """
    Generate responses from both models and compute rewards.
    Uses a generator to yield progressive updates as each step completes.
    """
    if not prompt.strip():
        yield (
            "",
            "Please enter a prompt.",
            "",
            "",
            "",
            "",
        )
        return

    try:
        # =====================================================================
        # Step 1: Load and generate from BASE model
        # =====================================================================
        yield (
            "Loading Base Model (Qwen2.5-1.5B)...",
            "",
            "",
            "",
            "",
            "",
        )

        print("Loading base model...")
        base_model = QwenModel(model_path=get_base_model_path(), device=DEVICE)

        yield (
            "Generating from Base Model...",
            "",
            "",
            "",
            "",
            "",
        )

        print("Generating from base model...")
        base_output = base_model.generate(prompt, config=GEN_CONFIG)

        # Cleanup base model
        del base_model
        clear_memory()
        print("Base model unloaded.")

        # Show base response
        base_response = base_output["response"]
        yield (
            "Base Model Complete",
            base_response,
            "",
            "Loading Aligned Model...",
            "",
            "",
        )

        # =====================================================================
        # Step 2: Load and generate from ALIGNED model
        # =====================================================================
        print("Loading aligned model...")
        aligned_model = QwenModel(model_path=get_base_model_path(), device=DEVICE)

        # Load aligned weights
        weights_path = get_aligned_weights_path()
        if weights_path:
            print(f"Loading aligned weights from: {weights_path}")
            aligned_model.load_aligned_weights(weights_path)
        else:
            print("WARNING: Using base weights (aligned weights not found)")

        yield (
            "Base Model Complete",
            base_response,
            "",
            "Generating from Aligned Model...",
            "",
            "",
        )

        print("Generating from aligned model...")
        aligned_output = aligned_model.generate(prompt, config=GEN_CONFIG)

        # Cleanup aligned model
        del aligned_model
        clear_memory()
        print("Aligned model unloaded.")

        # Show aligned response
        aligned_response = aligned_output["response"]
        yield (
            "Base Model Complete",
            base_response,
            "",
            "Aligned Model Complete",
            aligned_response,
            "",
        )

        # =====================================================================
        # Step 3: Load reward model and compute rewards
        # =====================================================================
        yield (
            "Computing Rewards...",
            base_response,
            "",
            "Computing Rewards...",
            aligned_response,
            "",
        )

        print("Loading reward model...")
        rewarder = SentimentRewarder(device=DEVICE)

        # Compute rewards
        base_reward_info = rewarder.get_reward(base_output["full_text"])
        aligned_reward_info = rewarder.get_reward(aligned_output["full_text"])

        # Cleanup reward model
        del rewarder
        clear_memory()
        print("Reward model unloaded.")

        # Apply EOS penalty (same as training)
        # If EOS not found, reward is -10. If found, reward is sentiment reward.
        if base_output["eos_found"]:
            base_reward = base_reward_info["reward"]
        else:
            base_reward = -10.0

        if aligned_output["eos_found"]:
            aligned_reward = aligned_reward_info["reward"]
        else:
            aligned_reward = -10.0

        # Format reward displays
        base_reward_text = format_reward_display(
            base_reward_info, base_output["eos_found"], base_reward
        )
        aligned_reward_text = format_reward_display(
            aligned_reward_info, aligned_output["eos_found"], aligned_reward
        )

        # Final output
        yield (
            "Done",
            base_response,
            base_reward_text,
            "Done",
            aligned_response,
            aligned_reward_text,
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield (
            "Error",
            error_msg,
            "",
            "Error",
            error_msg,
            "",
        )


# ============================================================================
# Gradio Interface
# ============================================================================

CUSTOM_CSS = ""

with gr.Blocks(css=CUSTOM_CSS, title="Base vs Aligned LLM Comparison") as demo:
    # Header
    gr.Markdown(
        """
        # Base vs PPO-Aligned Model Comparison

        Compare responses from a **base Qwen2.5-1.5B** model versus a **PPO-aligned** version.

        ### Alignment Objective
        The aligned model was trained using **Proximal Policy Optimization (PPO)** with a reward function that encourages:
        - **Positive sentiment** responses (reward: +10 x confidence)
        - **Short, complete** responses that end with an EOS token
        - **Penalizes** responses that don't terminate (-10 penalty) or are negative (-1 x confidence)

        ### Generation Settings
        Same settings used during training: `temperature=1.0`, `top_k=50`, `max_new_tokens=50`

        > **Note:** Models are loaded sequentially to work on CPU. Total generation time is ~60-90 seconds.
        """
    )

    gr.Markdown("---")

    # Input section
    gr.Markdown("### Enter a Prompt or Select an Example")

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Your Prompt",
            placeholder="Type a prompt here or click an example below...",
            lines=2,
            scale=4,
        )
        generate_btn = gr.Button("Generate", variant="primary", scale=1)

    # Example prompts
    gr.Markdown("**Quick Examples** (click to use):")
    with gr.Row():
        example_btns = []
        for example in EXAMPLE_PROMPTS:
            btn = gr.Button(f'"{example}"', size="sm")
            example_btns.append(btn)

    gr.Markdown("---")

    # Output section
    gr.Markdown("### Comparison Results")

    with gr.Row():
        # Base model column
        with gr.Column():
            gr.Markdown("#### Base Model (Qwen2.5-1.5B)")
            base_status = gr.Textbox(
                label="Status",
                interactive=False,
            )
            base_response = gr.Textbox(
                label="Response",
                lines=4,
                interactive=False,
            )
            base_reward = gr.Textbox(
                label="Reward Analysis",
                lines=6,
                interactive=False,
            )

        # Aligned model column
        with gr.Column():
            gr.Markdown("#### Aligned Model (PPO-trained)")
            aligned_status = gr.Textbox(
                label="Status",
                interactive=False,
            )
            aligned_response = gr.Textbox(
                label="Response",
                lines=4,
                interactive=False,
            )
            aligned_reward = gr.Textbox(
                label="Reward Analysis",
                lines=6,
                interactive=False,
            )

    gr.Markdown("---")

    # Footer
    gr.Markdown(
        """
        ### About This Demo

        This demo showcases **Reinforcement Learning from Human Feedback (RLHF)** style training,
        using **PPO** to align a language model toward generating positive sentiment responses.

        **Key Observations:**
        - The aligned model generates shorter, more positive completions
        - The aligned model reliably produces EOS tokens (proper termination)
        - The base model may generate longer, more varied (but potentially negative) text

        *Built with [Gradio](https://gradio.app) | Model: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)*
        """
    )

    # Event handlers - outputs now include status fields
    outputs = [base_status, base_response, base_reward, aligned_status, aligned_response, aligned_reward]

    generate_btn.click(
        fn=generate_comparison,
        inputs=[prompt_input],
        outputs=outputs,
    )

    prompt_input.submit(
        fn=generate_comparison,
        inputs=[prompt_input],
        outputs=outputs,
    )

    # Wire up example buttons
    for btn, example in zip(example_btns, EXAMPLE_PROMPTS):
        btn.click(fn=lambda e=example: e, outputs=[prompt_input]).then(
            fn=generate_comparison,
            inputs=[prompt_input],
            outputs=outputs,
        )


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    demo.launch()
