# cli_eval.py
import models.eval_entry  # ensures @register_model runs
from lm_eval import evaluator

if __name__ == "__main__":
    results = evaluator.simple_evaluate(
        model="my_lm",
        model_args="config_path=configs/models/tiny-moe-64-emb-8-decoder-8192.yaml,tokenizer_path=tokenizer.pkl,ckpt=checkpoints/moe-language-modeling-tiny-run-2025-09-10-21-40-32/weights/moe-basic-1m-64-emb-1L-iter999.pt,device=cuda",
        tasks=["wikitext"],
        num_fewshot=0,
        limit=100  # <-- limit the number of examples evaluated
    )
    print(results)
