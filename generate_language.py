import argparse
import time
import torch 
import torch.nn as nn
import json
import os 
from dotenv import load_dotenv 
from models.loader import load_prexisting_tokenizer, load_language_model
from models.tasks.language.tokenizer import BasicTokenizer
from models.tasks.language.model import LanguageModel

load_dotenv()

def sample_highest_prob(logits):
    optimal_index = torch.argmax(logits, dim = -1)
    next_token_id = optimal_index[:, -1].item()
    return next_token_id

def generate(input_data: str, tokenizer: BasicTokenizer, net: LanguageModel, device: torch.device, max_output_length: int = 100, name: str = "", generation_folder: str = "", checkpoint_path: str = "", tokenizer_path: str = ""):
    net.eval()

    tokenized_input_data = tokenizer.encode(input_data)
    tokenized_input_data.insert(0, tokenizer.get_beginning_of_sequence_token())

    new_tokens = []
    with torch.no_grad():
        for i in range(max_output_length):

            # Come up with an input tensor
            input_tensor = torch.tensor([tokenized_input_data], device=device, dtype=torch.int)

            # Obtain the logits
            logits = net(input_tensor)

            # Extract the id
            next_token_id = sample_highest_prob(logits)

            if next_token_id == tokenizer.get_end_of_sequence_token():
                print("EOS Token reached")
                break

            tokenized_input_data.append(next_token_id)
            new_tokens.append(tokenizer.decode_single(next_token_id))

    output = " ".join(BasicTokenizer.get_tokens(input_data) + new_tokens)
    if os.path.exists(generation_folder) and os.path.isdir(generation_folder):
        data = {
            "output" : output, 
            "max_output_length": max_output_length,
            "model_name" : name,
            "checkpoint_path" : checkpoint_path,
            "tokenizer_path" : tokenizer_path,
            "total_generated_tokens" : len(new_tokens),
            "total_length_tokens" : len(BasicTokenizer.get_tokens(input_data)) + len(new_tokens)
        }

        save_path = os.path.join(generation_folder, f"generation-{time.strftime('%Y-%m-%d %H:%M:%S')}.json")
        with open(save_path, "w") as f:
            json.dump(data, f)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data using a language model.")

    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe.yaml",
                    help="Path to model config YAML")

    parser.add_argument("--checkpoint_path", type=str, help="Path to model config YAML")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer pkl file", default="tokenizer.pkl")

    parser.add_argument("--input_data", type=str, help="Input data to generate")
    parser.add_argument("--max_output_tokens", type=int, help="Maximum tokens to generate", default = 100)
    parser.add_argument("--device", type=str, help="Hardware device to generate on", default = "cuda")
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default = 1)
    parser.add_argument("--generation_folder", type=str, help="Default generation", default = "generations")


    args = parser.parse_args()


    # --- Load configs ---
    model_config = args.model_config
    tokenizer_path = args.tokenizer_path
    checkpoint_path = args.checkpoint_path
    input_data = args.input_data
    hardware_device = args.device
    seed = args.seed
    generation_folder = args.generation_folder

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

    result = generate(input_data,
        tokenizer, 
        net, 
        device, 
        args.max_output_tokens,
        name,
        generation_folder,
        checkpoint_path,
        tokenizer_path

    )

    print("Output text", result)