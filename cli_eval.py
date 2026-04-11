import argparse
import torch
from safetensors.torch import load_file
from models.loader import load_language_model, load_language_model_from_pretrained
from pretrain_language import run_lm_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an arcadium model via lm-eval")

    # Two ways to load a model:
    #   1. From a checkpoint directory saved with save_pretrained (preferred)
    #   2. From a YAML config + safetensors or legacy .pt checkpoint
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf_dir", type=str,
                       help="Path to a checkpoint directory saved with save_pretrained(). "
                            "Tokenizer must be saved in the same directory.")
    group.add_argument("--model_config", type=str,
                       help="Path to model config YAML (use with --checkpoint)")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint or model.safetensors file "
                             "(required when using --model_config)")
    parser.add_argument("--tasks", nargs="+", default=["wikitext"],
                        help="lm-eval tasks to run (e.g. wikitext hellaswag arc_challenge)")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of few-shot examples")
    parser.add_argument("--batch_size", default="auto",
                        help="Batch size passed to HFLM ('auto' or an integer)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (useful for quick tests)")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of times to repeat each eval; reports min/max across runs")
    parser.add_argument("--gpu_memory_fraction", type=float, default=None,
                        help="Fraction of GPU VRAM to allow lm-eval to use (0.0–1.0). "
                             "Omit to use all available VRAM.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.hf_dir:
        model, tokenizer = load_language_model_from_pretrained(args.hf_dir, device)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when using --model_config")
        _, _, model, tokenizer = load_language_model(args.model_config, device)
        if args.checkpoint.endswith(".pt"):
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(load_file(args.checkpoint))

    model.eval()

    eval_conf = {
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_runs": args.num_runs,
        "gpu_memory_fraction": args.gpu_memory_fraction,
    }
    results = run_lm_eval(model, tokenizer, eval_conf, device)

    for task, res in results.items():
        print(f"== {task}, shots: {args.num_fewshot} ==")
        for metric, agg in res.items():
            if isinstance(agg, dict) and "runs" in agg:
                runs_str = ", ".join(f"{v:.4f}" for v in agg["runs"] if isinstance(v, float))
                if "min" in agg:
                    print(f"  {metric}: min={agg['min']:.4f}  max={agg['max']:.4f}  runs=[{runs_str}]")
                else:
                    print(f"  {metric}: [{runs_str}]")
            else:
                print(f"  {metric}: {agg}")


if __name__ == "__main__":
    main()
