import torch
from lm_eval.api.registry import register_model
from models.tasks.language.model import MyLMConfig, MyLMHF
from models.tasks.language.evaluators import MyLMEval

@register_model("my_lm")
class MyLMEvalWrapper(MyLMEval):
    """
    Wrapper for lm_eval-harness with memory-optimized initialization.
    """

    def __init__(self, **kwargs):
        config_path = kwargs.get("config_path")
        tokenizer_path = kwargs.get("tokenizer_path")
        ckpt_path = kwargs.get("ckpt")
        device = kwargs.get("device", "cuda")
        chunk_size = kwargs.get("chunk_size", 512)  # default chunking for loglikelihood

        # 1. Build HF-style model
        hf_config = MyLMConfig(config_path=config_path, tokenizer_path=tokenizer_path)
        model = MyLMHF(hf_config, device="cpu")  # initially load on CPU

        # 2. Load checkpoint on CPU
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.lm.load_state_dict(ckpt["model_state_dict"])

        # 3. Move model to device in eval mode
        model.lm.eval()
        model.lm.to(device)

        # 4. Call parent constructor with chunking
        super().__init__(model, device=device, max_length=kwargs.get("max_length", 64), chunk_size=chunk_size)
