import argparse
import time
import torch 
import torch.nn as nn
import json
import os 
from dotenv import load_dotenv 
from models.loader import load_prexisting_tokenizer, load_language_model
from models.tasks.language.language_tokenizer import BasicTokenizer
from models.tasks.language.architecture import LanguageModel
from torch.profiler import profile, record_function, ProfilerActivity

load_dotenv()

def sample_highest_prob(logits):
    optimal_index = torch.argmax(logits, dim = -1)
    next_token_id = optimal_index[:, -1].item()
    return next_token_id

def generate(
    input_data: str,
    tokenizer: BasicTokenizer,
    net: LanguageModel,
    device: torch.device,
    max_output_length: int = 100,
    name: str = "",
    generation_folder: str = "",
    checkpoint_path: str = "",
    tokenizer_path: str = "",
    profile_steps: int = 0,
):
    net.eval()

    tokenized_input_data = tokenizer.encode(input_data)
    tokenized_input_data.insert(0, tokenizer.get_beginning_of_sequence_token())

    new_tokens = []
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

        for i in range(max_output_length):
            if profiler_ctx and i >= profile_steps:
                break

            input_tensor = torch.tensor(
                [tokenized_input_data], device=device, dtype=torch.int
            )

            if profiler_ctx:
                with record_function("logits_generation"):
                    logits = net(input_tensor)
            else:
                logits = net(input_tensor)

            next_token_id = sample_highest_prob(logits)

            now = time.time()
            if len(new_tokens) == 0:
                # Time to first token
                time_to_first_token = now - prev_time
            else:
                # Time per token
                per_token_times.append(now - prev_time)
            prev_time = now

            if next_token_id == tokenizer.get_end_of_sequence_token():
                print("EOS Token reached")
                break

            tokenized_input_data.append(next_token_id)
            new_tokens.append(tokenizer.decode_single(next_token_id))

            if profiler_ctx:
                profiler_ctx.step()

        if profiler_ctx:
            profiler_ctx.__exit__(None, None, None)

    output = " ".join(BasicTokenizer.get_tokens(input_data) + new_tokens)
    if os.path.exists(generation_folder) and os.path.isdir(generation_folder):
        data = {
            "output": output,
            "max_output_length": max_output_length,
            "model_name": name,
            "checkpoint_path": checkpoint_path,
            "tokenizer_path": tokenizer_path,
            "total_generated_tokens": len(new_tokens),
            "total_length_tokens": len(BasicTokenizer.get_tokens(input_data))
            + len(new_tokens),
            "profiled_steps": profile_steps,
            "time_to_first_token": time_to_first_token,
            "per_token_times": per_token_times,
            "mean_per_token_time": sum(per_token_times) / len(per_token_times)
            if per_token_times
            else None,
        }

        save_path = os.path.join(
            generation_folder, f"generation-{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data using a language model.")

    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe.yaml",
                    help="Path to model config YAML")

    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer pkl file", default="tokenizer.pkl")

    parser.add_argument("--input_data", type=str, help="Input data to generate")
    parser.add_argument("--max_output_tokens", type=int, help="Maximum tokens to generate", default=100)
    parser.add_argument("--device", type=str, help="Hardware device to generate on", default="cuda")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default=1)
    parser.add_argument("--generation_folder", type=str, help="Folder to save generations", default="generations")

    # New profiling args
    parser.add_argument("--profile", type=int, default=0,
                        help="Number of steps to profile with torch.profiler (0 = disable)")

    args = parser.parse_args()

    # --- Load configs ---
    model_config = args.model_config
    tokenizer_path = args.tokenizer_path
    checkpoint_path = args.checkpoint_path
    input_data = args.input_data
    hardware_device = args.device
    seed = args.seed
    generation_folder = args.generation_folder
    profile_steps = args.profile

    os.makedirs(generation_folder, exist_ok=True)

    # Plant our seed for reproducibility
    torch.manual_seed(seed)
    print("Using a seed of ", seed)

    # Hardware
    device = torch.device(hardware_device if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = load_prexisting_tokenizer(tokenizer_path)

    # Load net 
    name, model_type, net = load_language_model(model_config, device)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["model_state_dict"])

    result = generate(
        input_data,
        tokenizer, 
        net, 
        device, 
        args.max_output_tokens,
        name,
        generation_folder,
        checkpoint_path,
        tokenizer_path,
        profile_steps
    )

    print("Output text", result)
