import argparse
import torch 
import torch.nn as nn
from dotenv import load_dotenv 
from models.loader import load_prexisting_tokenizer, load_language_model
from models.tasks.language.tokenizer import BasicTokenizer
from models.tasks.language.model import LanguageModel

load_dotenv()

def sample_highest_prob(logits):
    optimal_index = torch.argmax(logits, dim = -1)
    next_token_id = optimal_index[:, -1].item()
    return next_token_id

def generate(input_data: str, tokenizer: BasicTokenizer, net: LanguageModel, device: torch.device, max_output_length: int = 100):
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

    print("Generated tokens", new_tokens)
    return input_data + " ".join(new_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data using a language model.")

    parser.add_argument("--model_config", type=str, default="configs/models/tiny-moe.yaml",
                    help="Path to model config YAML")

    parser.add_argument("--checkpoint_path", type=str, help="Path to model config YAML")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer pkl file", default="tokenizer.pkl")

    parser.add_argument("--input_data", type=str, help="Input data to generate")
    parser.add_argument("--max_output_tokens", type=int, help="Maximum tokens to generate", default = 100)
    parser.add_argument("--device", type=str, help="Hardware device to generate on", default = "cuda")

    args = parser.parse_args()


    # --- Load configs ---
    model_config = args.model_config
    tokenizer_path = args.tokenizer_path
    checkpoint_path = args.checkpoint_path
    input_data = args.input_data
    hardware_device = args.device

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
        args.max_output_tokens
    )

    print("Output text", result)