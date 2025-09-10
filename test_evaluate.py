# run_eval.py
import torch
from lm_eval import evaluator
from models.tasks.language.model import MyLMConfig, MyLMHF
from models.tasks.language.evaluators import MyLMEval

# 1. Load HuggingFace-style model
hf_config = MyLMConfig(
    config_path="configs/models/tiny-moe-64-emb-8-decoder-8192.yaml",
    tokenizer_path="tokenizer.pkl"
)
model = MyLMHF(hf_config, device="cuda")

# 2. Load checkpoint
ckpt = torch.load("checkpoints/moe-language-modeling-tiny-moe-basic-1m-64-emb-1L-2025-09-10-12:50:49/moe-basic-1m-64-emb-1L-iter999.pt", map_location="cuda")
model.lm.load_state_dict(ckpt["model_state_dict"])

# 3. Wrap with lm-eval interface
mylm = MyLMEval(model, device="cuda")

# 4. Run evaluation
results = evaluator.simple_evaluate(
    model=mylm,
    tasks=["wikitext"],  # replace with ["lambada_openai", "hellaswag", ...]
    num_fewshot=0
)

print(results)
