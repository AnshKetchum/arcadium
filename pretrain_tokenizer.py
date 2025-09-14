import argparse
import os 

from models.loader import load_tokenizer, ingest_file
from models.tasks.language.tokenizers.base import BasicTokenizer
from utils import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_config", type=str, required=True, help="Path to raw data to set tokenizer onto")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="Path to configuration describing tokenizer")
    parser.add_argument("--save_directory", type=str, required=False, default=None, help="Path to data to set tokenizer onto. Otherwise, we default to the tokenizer path listed")

    args = parser.parse_args()

    # Load configs
    data_config_path = args.raw_data_config
    tokenizer_config_path = args.tokenizer_config

    assert os.path.exists(data_config_path), f"{data_config_path} doesn't exist"

    # Load the actual data config
    data_config = load_config(data_config_path, "parameters")
    tokenizer_config = load_config(tokenizer_config_path, "parameters")

    # Instantiate a tokenizer 
    tok = load_tokenizer(**tokenizer_config)

    # Ingest all relevant files
    for fl in data_config.get("files", []):
        ingest_file(fl, tok)
    
    for fldr in data_config.get("folders", []):
        for fl in os.listdir(fldr):
            print("Reading", fl)
            ingest_file(os.path.join(fldr, fl), tok)


    # Save the tokenizer
    save_path = tokenizer_config["tokenizer_path"] if not args.save_directory else args.save_directory
    print("Saving to ", save_path)
    tok.save(save_path)