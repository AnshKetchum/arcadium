import argparse
import time
import torch
import json
import os
from dotenv import load_dotenv
from safetensors.torch import load_file
from arcadium.tasks.language.loader import load_language_model
from arcadium.models.language import LanguageModel
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

load_dotenv()


def sample_highest_prob(logits):
    return torch.argmax(logits, dim=-1)[:, -1].item()


def generate(
    input_data: str,
    tokenizer,
    net: LanguageModel,
    device: torch.device,
    max_output_length: int = 100,
    name: str = "",
    generation_folder: str = "",
    checkpoint_path: str = "",
    tokenizer_path: str = "",
    profile_steps: int = 0,
    use_kv_cache: bool = False,
    collect_hidden_states: bool = False,
):
    """
    Greedily generate up to max_output_length tokens from input_data.

    Saves a JSON result to generation_folder if that directory exists.
    Returns (output_text, hidden_states_per_step) where hidden_states_per_step is a list of
    per-token hidden state lists when collect_hidden_states=True, else None.
    """
    net.eval()

    tokenized_input_data = tokenizer.encode(input_data)
    n_prompt_tokens = len(tokenized_input_data)
    n_new = 0
    all_hidden_states = [] if collect_hidden_states else None

    profiler_ctx = (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else [ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(generation_folder, "profile")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        if profile_steps > 0
        else None
    )

    per_token_times = []
    time_to_first_token = None
    prev_time = time.time()

    with torch.no_grad():
        if profiler_ctx:
            profiler_ctx.__enter__()

        for i in tqdm(range(max_output_length), desc="Generating"):
            if profiler_ctx and i >= profile_steps:
                break

            input_tensor = torch.tensor([tokenized_input_data], device=device, dtype=torch.int)

            if profiler_ctx:
                with record_function("logits_generation"):
                    output = net(input_tensor, use_kv_cache=use_kv_cache, collect_hidden_states=collect_hidden_states)
            else:
                output = net(input_tensor, use_kv_cache=use_kv_cache, collect_hidden_states=collect_hidden_states)

            logits = output.logits if hasattr(output, "logits") else output
            next_token_id = sample_highest_prob(logits)

            if collect_hidden_states:
                hs = getattr(output, "metadata", None)
                hs = hs.get("hidden_states") if isinstance(hs, dict) else None
                all_hidden_states.append(hs)

            now = time.time()
            if n_new == 0:
                time_to_first_token = now - prev_time
            else:
                per_token_times.append(now - prev_time)
            prev_time = now

            if next_token_id == tokenizer.eos_token_id:
                print("EOS reached")
                break

            tokenized_input_data.append(next_token_id)
            n_new += 1

            if profiler_ctx:
                profiler_ctx.step()

        if profiler_ctx:
            profiler_ctx.__exit__(None, None, None)

    output_text = tokenizer.decode(tokenized_input_data)

    if os.path.exists(generation_folder) and os.path.isdir(generation_folder):
        data = {
            "output": output_text,
            "max_output_length": max_output_length,
            "model_name": name,
            "checkpoint_path": checkpoint_path,
            "tokenizer_path": tokenizer_path,
            "total_generated_tokens": n_new,
            "total_length_tokens": len(tokenized_input_data),
            "profiled_steps": profile_steps,
            "time_to_first_token": time_to_first_token,
            "per_token_times": per_token_times,
            "mean_per_token_time": sum(per_token_times) / len(per_token_times) if per_token_times else None,
        }
        save_path = os.path.join(generation_folder, f"generation-{time.strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    return output_text, all_hidden_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from an arcadium language model.")
    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to a checkpoint directory (saved with save_pretrained) or a "
                             ".pt file (legacy format, loads model_state_dict key)")
    parser.add_argument("--input_data", type=str, required=True, help="Prompt text")
    parser.add_argument("--max_output_tokens", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generation_folder", type=str, default="generations")
    parser.add_argument("--use_kv_cache", action="store_true")
    parser.add_argument("--profile", type=int, default=0,
                        help="Steps to profile with torch.profiler (0 = disabled)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.generation_folder, exist_ok=True)

    name, _, net, tokenizer = load_language_model(args.model_config, device)

    if os.path.isdir(args.checkpoint_path):
        from arcadium.tasks.language.loader import load_language_model_from_pretrained
        net, tokenizer = load_language_model_from_pretrained(args.checkpoint_path, device)
    else:
        state = torch.load(args.checkpoint_path, map_location=device)
        net.load_state_dict(state["model_state_dict"])

    result, _ = generate(
        args.input_data, tokenizer, net, device,
        args.max_output_tokens, name, args.generation_folder,
        args.checkpoint_path, "", args.profile, args.use_kv_cache,
    )
    print("Output:", result)
