import argparse
import os 

from models.loader import ingest_file
from models.tasks.language.tokenizer import BasicTokenizer
from utils import load_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, required=True, help="Path to data to set tokenizer onto")
    parser.add_argument("--save_path", type=str, required=False, default="tokenizer.pkl", help="Path to data to set tokenizer onto")

    args = parser.parse_args()

    # Load configs
    data_config_path = args.data_config
    save_path = args.save_path

    assert os.path.exists(data_config_path), f"{data_config_path} doesn't exist"

    # Load the actual data config
    data_config = load_config(data_config_path, "parameters")

    # Instantiate a tokenizer 
    tok = BasicTokenizer()

    # Ingest all relevant files
    for fl in data_config.get("files", []):
        ingest_file(fl, tok)
    
    for fldr in data_config.get("folders", []):
        for fl in os.listdir(fldr):
            ingest_file(os.path.join(fldr, fl), tok)


    # Save the tokenizer
    tok.save(save_path)