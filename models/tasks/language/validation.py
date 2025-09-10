import torch
from lm_eval.api.model import LM

# import your model + config class
from .architecture import LanguageModel
from my_package.config import load_config  # you need a way to load config
from my_package.tokenizer import MyTokenizer  # replace with your tokenizer

class MyLM(LM):
    def __init__(self, checkpoint_path, config_path, vocab_size, device="cuda"):
        super().__init__()
        self.device = device

        # 1. Load config
        config = load_config(config_path)

        # 2. Build model
        self.model = LanguageModel(config.transformer_cls, config, vocab_size)
        self.model.to(device)

        # 3. Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 4. Load tokenizer
        self.tokenizer = MyTokenizer(config)

    def loglikelihood(self, requests):
        results = []
        for context, continuation in requests:
            context_ids = self.tokenizer.encode(context)
            cont_ids = self.tokenizer.encode(continuation)

            input_ids = torch.tensor([context_ids + cont_ids], device=self.device)

            with torch.no_grad():
                logits = self.model(input_ids[:, :-1])
                log_probs = torch.log_softmax(logits, dim=-1)

            cont_start = len(context_ids)
            target_ids = input_ids[:, cont_start:]
            pred_log_probs = log_probs[:, cont_start-1:-1, :]

            token_log_probs = pred_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            total_loglikelihood = token_log_probs.sum().item()

            results.append((total_loglikelihood, True))
        return results

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
