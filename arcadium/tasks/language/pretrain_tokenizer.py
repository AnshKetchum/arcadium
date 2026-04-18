"""
Train a tokenizer from a corpus and save it as a HuggingFace tokenizer.

Three tokenizer types are supported:
  word          Whitespace-split word-level (original behaviour)
  bpe           Byte-level BPE — same algorithm as tiktoken / GPT-2
  sentencepiece Metaspace BPE  — same algorithm as SentencePiece BPE

All three save via PreTrainedTokenizerFast.save_pretrained(), producing a
tokenizers.json that is loadable with AutoTokenizer.from_pretrained().

Usage:
  python -m arcadium.tasks.language.pretrain_tokenizer \
      --raw_data_config configs/raw_data/simple-corpus.yaml \
      --save_dir tokenizers/basic \
      [--vocab_size 8192] \
      [--type word|bpe|sentencepiece]
"""
import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import ByteLevel, Metaspace, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, Metaspace as MetaspaceDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from transformers import PreTrainedTokenizerFast

from arcadium.utils import load_config


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


def train_tokenizer(files: list[str], vocab_size: int, save_dir: str) -> PreTrainedTokenizerFast:
    """Whitespace-split word-level tokenizer. Special tokens: [PAD], [UNK], [BOS], [EOS]."""
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


def train_bpe_tokenizer(files: list[str], vocab_size: int, save_dir: str) -> PreTrainedTokenizerFast:
    """
    Byte-level BPE tokenizer — the same algorithm used by tiktoken and GPT-2.

    Uses a ByteLevel pre-tokenizer so every Unicode codepoint is representable
    without an explicit UNK token. Special token: <|endoftext|>.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.train(files=files, trainer=trainer)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )

    os.makedirs(save_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Saved BPE tokenizer with vocab size {len(hf_tokenizer)} to {save_dir}")
    return hf_tokenizer


def train_sentencepiece_bpe_tokenizer(files: list[str], vocab_size: int, save_dir: str) -> PreTrainedTokenizerFast:
    """
    Metaspace BPE tokenizer — the same algorithm used by SentencePiece BPE models.

    Prepends a '▁' marker to each word so spaces are part of the vocabulary,
    matching the behaviour of models like ALBERT and XLNet.
    Special tokens: <unk>, <s>, </s>.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = MetaspaceDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"],
    )
    tokenizer.train(files=files, trainer=trainer)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="</s>",
    )

    os.makedirs(save_dir, exist_ok=True)
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Saved SentencePiece BPE tokenizer with vocab size {len(hf_tokenizer)} to {save_dir}")
    return hf_tokenizer


_TRAINERS = {
    "word": train_tokenizer,
    "bpe": train_bpe_tokenizer,
    "sentencepiece": train_sentencepiece_bpe_tokenizer,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HuggingFace tokenizer from corpus.")
    parser.add_argument("--raw_data_config", type=str, required=True,
                        help="Path to raw data config YAML (expects 'files' and/or 'folders' keys)")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the tokenizer (loadable with AutoTokenizer.from_pretrained)")
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="Maximum vocabulary size (default: 8192)")
    parser.add_argument("--type", dest="tokenizer_type", default="word",
                        choices=list(_TRAINERS),
                        help="Tokenizer algorithm: word, bpe (tiktoken-style), or sentencepiece (default: word)")
    args = parser.parse_args()

    assert os.path.exists(args.raw_data_config), f"Config not found: {args.raw_data_config}"
    data_config = load_config(args.raw_data_config, "parameters")

    files = collect_files(data_config)
    assert files, "No corpus files found — check your raw_data_config."
    print(f"Training {args.tokenizer_type!r} tokenizer on {len(files)} file(s)...")

    _TRAINERS[args.tokenizer_type](files, args.vocab_size, args.save_dir)
