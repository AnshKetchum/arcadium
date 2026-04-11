import argparse
import os
import pickle

from models.loader import load_tokenizer, load_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize files into shards for pretraining.")
    parser.add_argument("--files", nargs="*", default=[], help="Individual files to tokenize")
    parser.add_argument("--folders", nargs="*", default=[], help="Folders of files to tokenize")
    parser.add_argument("--tokenizer_config", type=str, required=True,
                        help="Path to tokenizer config YAML (expects model_name_or_path key)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Override: load tokenizer from this path instead of the config")
    parser.add_argument("--shard_save_path", type=str, required=True,
                        help="Path to write the tokenized shard (.pkl)")
    args = parser.parse_args()

    if args.tokenizer_path:
        tok = load_tokenizer(args.tokenizer_path)
    else:
        config = load_config(args.tokenizer_config, "parameters")
        tok = load_tokenizer(**config)

    all_documents = []

    for fl in args.files:
        if not os.path.isfile(fl):
            print(f"Warning: not a file, skipping: {fl}")
            continue
        print("Adding", fl)
        with open(fl, "r") as f:
            tokens = tok.encode(f.read())
        all_documents.append({"file_name": fl, "tokens": tokens})

    for fldr in args.folders:
        if not os.path.isdir(fldr):
            print(f"Warning: not a directory, skipping: {fldr}")
            continue
        for fname in os.listdir(fldr):
            full = os.path.join(fldr, fname)
            if not os.path.isfile(full):
                continue
            print("Adding", full)
            with open(full, "r") as f:
                tokens = tok.encode(f.read())
            all_documents.append({"file_name": full, "tokens": tokens})

    with open(args.shard_save_path, "wb") as f:
        pickle.dump({"documents": all_documents}, f)
    print(f"Saved {len(all_documents)} documents to {args.shard_save_path}")
