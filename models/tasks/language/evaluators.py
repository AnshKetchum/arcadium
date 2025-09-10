# my_eval.py
from lm_eval.api.model import LM
import torch
import torch.nn.functional as F

class MyLMEval(LM):
    def __init__(self, model, max_length=2048, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer  # from wrapped model
        self._max_length = max_length
        self.device = device

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        res = []
        for context, continuation in requests:
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)

            input_ids = torch.tensor([ctx_ids + cont_ids[:-1]], device=self.device)
            target_ids = torch.tensor([cont_ids], device=self.device)

            with torch.no_grad():
                logits = self.model.lm(input_ids)
                logits = logits[:, -len(cont_ids):, :]
                log_probs = F.log_softmax(logits, dim=-1)

                selected = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                loglikelihood = selected.sum().item()

            res.append((loglikelihood, True))
        return res

    def loglikelihood_rolling(self, requests):
        """
        Approximate implementation: compute loglikelihood over the entire string,
        without splitting context/continuation.
        """
        res = []
        for string in requests:
            ids = self.tok_encode(string)
            if len(ids) <= 1:
                res.append((0.0, True))
                continue

            input_ids = torch.tensor([ids[:-1]], device=self.device)
            target_ids = torch.tensor([ids[1:]], device=self.device)

            with torch.no_grad():
                logits = self.model.lm(input_ids)
                log_probs = F.log_softmax(logits, dim=-1)

                selected = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                loglikelihood = selected.sum().item()

            res.append((loglikelihood, True))
        return res

    def generate_until(self, requests):
        res = []
        for context, until in requests:
            ctx_ids = self.tok_encode(context)
            input_ids = torch.tensor([ctx_ids], device=self.device)

            out_ids = self.model.generate(input_ids, max_length=self._max_length)[0]
            text = self.tok_decode(out_ids.tolist())
            res.append(text)
        return res
