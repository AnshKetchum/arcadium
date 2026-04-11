"""
Train a word-level tokenizer from a corpus and save it as a HuggingFace tokenizer.

The resulting directory can be loaded with AutoTokenizer.from_pretrained() and is
directly usable in model configs via tokenizer_type: model_name_or_path: <save_dir>.

Usage:
  python pretrain_tokenizer.py \
      --raw_data_config configs/raw_data/simple-corpus.yaml \
      --save_dir tokenizers/basic \
      [--vocab_size 8192]
"""
import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from utils import load_config


def collect_files(data_config: dict) -> list[str]:
    """Collect all corpus file paths from a raw_data config dict."""
    files = list(data_config.get("files", []))
    for folder in data_config.get("folders", []):
        if not os.path.isdir(folder):
            print(f"Warning: folder not found, skipping: {folder}")
            continue
        for fname in os.listdir(folder):
            full = os.path.join(folder, fname)
            if os.path.isfile(full):
                files.append(full)
    return files


def train_tokenizer(files: list[str], vocab_size: int, save_dir: str):
    """
    Train a whitespace-split word-level tokenizer on the given corpus files and
    save it to save_dir with save_pretrained() so it can be loaded with
    AutoTokenizer.from_pretrained(save_dir).

    Special tokens: [PAD], [UNK], [BOS], [EOS].
    """
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=files, trainer=trainer)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    os.makedirs(save_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Saved tokenizer with vocab size {len(hf_tokenizer)} to {save_dir}")
    return hf_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a word-level HuggingFace tokenizer from corpus.")
    parser.add_argument("--raw_data_config", type=str, required=True,
                        help="Path to raw data config YAML (expects 'files' and/or 'folders' keys)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the tokenizer (loadable with AutoTokenizer.from_pretrained)")
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="Maximum vocabulary size (default: 8192)")
    args = parser.parse_args()

    assert os.path.exists(args.raw_data_config), f"Config not found: {args.raw_data_config}"
    data_config = load_config(args.raw_data_config, "parameters")

    files = collect_files(data_config)
    assert files, "No corpus files found — check your raw_data_config."
    print(f"Training on {len(files)} file(s)...")

    train_tokenizer(files, args.vocab_size, args.save_dir)
