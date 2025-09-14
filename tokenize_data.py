import argparse
import os
import pickle

from models.loader import load_tokenizer, load_config
from models.tasks.language.tokenizers.base import BasicTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="List of individual files to tokenize"
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=[],
        help="List of folders containing files to tokenize"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=False,
        default=None,
        help="Path to serialized tokenizer"
    )
    parser.add_argument(
        "--tokenizer_config",
        type=str,
        required=True,
        help="Path to serialized tokenizer"
    )
    parser.add_argument(
        "--shard_save_path",
        type=str,
        required=True,
        help="Path to save tokenized shard"
    )

    args = parser.parse_args()

    # Load the tokenizer
    config = load_config(args.tokenizer_config, "parameters")
    tok = load_tokenizer(**config)

    if args.tokenizer_path:
        print("Custom load path specified, loading from", args.tokenizer_path)
        tok.load(args.tokenizer_path)

    # Ingest all relevant files
    all_documents = []

    # Handle direct file paths
    for fl in args.files:
        if not os.path.isfile(fl):
            print(f"Warning: {fl} is not a valid file, skipping.")
            continue
        print("Adding", fl, "to shard")
        with open(fl, 'r') as f:
            data = f.read()

        tokens = tok.encode(data)
        all_documents.append({
            "file_name": fl,
            "tokens": tokens
        })

    # Handle all files in given folders
    for fldr in args.folders:
        if not os.path.isdir(fldr):
            print(f"Warning: {fldr} is not a valid folder, skipping.")
            continue
        for fl in os.listdir(fldr):
            full_path = os.path.join(fldr, fl)
            if not os.path.isfile(full_path):
                continue
            print("Adding", full_path, "to shard")
            with open(full_path, 'r') as f:
                data = f.read()

            tokens = tok.encode(data)
            all_documents.append({
                "file_name": full_path,
                "tokens": tokens
            })

    # Save the shard
    data = {"documents": all_documents}

    with open(args.shard_save_path, 'wb') as f:
        pickle.dump(data, f)

    print("Successfully saved to", args.shard_save_path)
