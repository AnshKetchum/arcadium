from lm_eval.api.model import LM
import torch
import torch.nn.functional as F
import math


class MyLMEval(LM):
    def __init__(self, model, max_length=2048, device="cuda", chunk_size=512):
        """
        Args:
            model: Wrapped PyTorch language model with .tokenizer
            max_length: maximum sequence length for generation
            device: 'cuda' or 'cpu'
            chunk_size: max length per chunk for loglikelihood evaluation
        """
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer
        self._max_length = max_length
        self.device = device
        self.chunk_size = chunk_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 64

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, string):
        return self.tokenizer.encode(string.doc["page"])

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _compute_loglikelihood(self, input_ids, target_ids):
        """Compute loglikelihood for a chunk of tokens"""
        with torch.no_grad():
            logits = self.model.lm(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            return selected.sum().item()

    def loglikelihood(self, requests):
        res = []
        for context, continuation in requests:
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)

            # Split into manageable chunk to avoid OOM
            total_ids = ctx_ids + cont_ids
            total_len = len(cont_ids)
            loglikelihood = 0.0

            start_idx = 0
            while start_idx < total_len:
                end_idx = min(start_idx + self.chunk_size, total_len)
                input_ids = torch.tensor([total_ids[:end_idx]], device=self.device)
                target_ids = torch.tensor([total_ids[start_idx:end_idx]], device=self.device)
                loglikelihood += self._compute_loglikelihood(input_ids, target_ids)
                start_idx = end_idx

            res.append((loglikelihood, True))
        return res

    def loglikelihood_rolling(self, requests):
        res = []
        for string in requests:
            ids = self.tok_encode(string)
            if len(ids) <= 1:
                res.append((0.0, True))
                continue

            # Process in chunks
            loglikelihood = 0.0
            start_idx = 0
            while start_idx < len(ids) - 1:
                end_idx = min(start_idx + self.chunk_size, len(ids) - 1)
                input_ids = torch.tensor([ids[start_idx:end_idx]], device=self.device)
                target_ids = torch.tensor([ids[start_idx + 1:end_idx + 1]], device=self.device)
                loglikelihood += self._compute_loglikelihood(input_ids, target_ids)
                start_idx = end_idx

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
