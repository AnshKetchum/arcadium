#!/usr/bin/env python3
import os
import yaml
import subprocess

base_config = {
    "parameters": {
        "name": "moe-hyperparameter-search",
        "type": "moe",
        "configuration": {
            "model": {
                "decoder_layers": 1  # to overwrite
            },
            "decoder": {
                "input_dimension": 64,
                "output_dimension": 64,
                "hidden_dimension": 64,
                "heads": 8,
                "experts": 8,
                "norm_eps": 1e-5,
            },
        },
    },
    "language_model": {
        "vocab_size": 4096,
    },
}

output_dir = "configs/models/generated_configs"
os.makedirs(output_dir, exist_ok=True)

for layers in range(1, 16):  # 1 through 15
    config = base_config.copy()
    config["parameters"]["configuration"]["model"]["decoder_layers"] = layers
    config["parameters"]["name"] = f"moe-basic-1m-64-emb-{layers}L"

    filename = os.path.join(output_dir, f"tiny-moe-64-emb-{layers}-decoder.yaml")
    # Write config
    with open(filename, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"\n=== Running grid search for {layers} decoder layer(s) ===")
    subprocess.run(
        ["python", "pretrain_language.py", "--model_config", filename],
        check=True
    )
