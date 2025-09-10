import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .architecture import LanguageModel

# HuggingFace Config
class MyLMConfig(PretrainedConfig):
    model_type = "mylm"

    def __init__(self, vocab_size=32000, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # add any fields from your existing config.decoder
        self.decoder = kwargs.get("decoder", {})

# HuggingFace Model wrapper
class MyLMHF(PreTrainedModel):
    config_class = MyLMConfig

    def __init__(self, config):
        super().__init__(config)

        # rebuild your model
        self.lm = LanguageModel(
            transformer_cls=Transformer,
            config=config,
            vocab_size=config.vocab_size
        )

    def forward(self, input_ids, labels=None):
        logits = self.lm(input_ids)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # shift so that tokens predict the next token
            loss = loss_fct(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1)
            )
        return {"loss": loss, "logits": logits}
