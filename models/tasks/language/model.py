import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from models.tasks.language.architecture import LanguageModel
from models.tasks.language.tokenizers.base import BasicTokenizer
from models.loader import load_language_model  # your existing loader


class MyLMConfig(PretrainedConfig):
    model_type = "mylm"

    def __init__(self, config_path=None, tokenizer_path=None, **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path


class MyLMHF(PreTrainedModel):
    config_class = MyLMConfig

    def __init__(self, config: MyLMConfig, device="cuda"):
        super().__init__(config)

        if config.config_path is None:
            raise ValueError("Must provide a YAML config_path to build model")

        # Load your model using existing loader
        _, _, net, tokenizer = load_language_model(config.config_path, device)
        self.lm = net

        # Load tokenizer
        if config.tokenizer_path is None:
            raise ValueError("Must provide tokenizer_path")
        self.tokenizer = tokenizer 
        
        # Store our own device reference (avoids conflict with HF property)
        self._device = torch.device(device)
        self.lm.to(self._device)

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.lm(input_ids.to(self._device))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return {"loss": loss, "logits": logits}

    def encode(self, text: str):
        return torch.tensor([self.tokenizer.encode(text)], device=self._device)

    def decode(self, token_ids):
        return " ".join([self.tokenizer.decode_single(t) for t in token_ids])

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_length=50,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
    ):
        """
        HuggingFace-style generate for lm-eval compatibility.
        - Greedy by default
        - Supports temperature and top-k sampling
        """
        generated = input_ids.clone().to(self._device)
        eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.tokenizer.get_end_of_sequence_token()
        )

        for _ in range(max_length):
            logits = self.lm(generated)  # [batch, seq_len, vocab]
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # keep only top_k logits
                values, indices = torch.topk(next_token_logits, top_k)
                mask = torch.full_like(next_token_logits, float("-inf"))
                mask.scatter_(1, indices, values)
                next_token_logits = mask

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated
