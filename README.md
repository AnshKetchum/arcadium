# A basic experiment to understand distributed MoE trainings, and perf 

also, to train an MoE model :D

# Setup 

```bash
uv pip install -r requirements.txt
```

# Pretraining from scratch

- Model params are in `configs/models/tiny-moe.yaml`
- Data params are in `configs/data/simple-corpus.yaml`
- Training params are in `configs/training/basic.yaml`

```bash
python pretrain_language.py
```

# Generating output using a pretrained LLM

It works, but output isn't that sensible.

```bash 
python generate_language.py --checkpoint_path "checkpoints/<checkpoint path>/<path to .pt file>" --input_data "Shakespeare produced most"
```


Underlying targets:

1. To understand VRAM capacity usage of MoEs! 

Current Interesting Findings:
    - Optimizers take more VRAM than model (specifically, the Adam optimizer takes 2 x more MB of VRAM than model weights)
    - Leading source of VRAM comes from *forward pass intermediates*

2. What is a fabric, and how do fabric parameters properly affect pre-training performance?
    - What is a fabric spine
    - What

3. How distributed perf actually pans out with comms latencies 

4. Make a sandbox to test kernels, perf optims, etc.