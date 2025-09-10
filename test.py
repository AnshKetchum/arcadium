import torch
from models.tasks.language.model import MyLMConfig, MyLMHF


# HuggingFace-style config pointing to your YAML + tokenizer
hf_config = MyLMConfig(
    config_path="configs/models/tiny-moe-64-emb-8-decoder-8192.yaml",
    tokenizer_path="tokenizer.pkl"
)

model = MyLMHF(hf_config, device="cuda")

# Load your saved checkpoint
ckpt = torch.load("checkpoints/moe-language-modeling-tiny-moe-basic-1m-64-emb-1L-2025-09-10-12:50:49/moe-basic-1m-64-emb-1L-iter999.pt", map_location="cuda")
model.lm.load_state_dict(ckpt["model_state_dict"])

# run generation
input_ids = model.encode("Shakespeare")
out = model.generate(input_ids, max_length=20, top_k=5, temperature=0.8)
print("Generated:", model.decode(out[0].tolist()))
