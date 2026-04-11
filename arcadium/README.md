# arcadium

the core library. everything lives here.

## layout

```
arcadium/
├── data/           dataset classes (fineweb, local shards, round-robin aggregation, sequence-length scheduling)
├── tokenizers/     tokenizer wrappers (currently thin stubs — use load_tokenizer from tasks/language/loader)
├── embeddings/     positional encodings (rope, sinusoidal)
├── optimizers/     optimizers (muon, adam, muon+adam)
├── components/     reusable architectural pieces
│   ├── activations.py      swiglu
│   ├── base.py             FFN block
│   ├── moe_layer.py        mixture-of-experts routing + experts
│   ├── attentions/         multi-head, multi-query, grouped-query attention + kv cache
│   └── associators/        rwkv and similar recurrent mixers
├── models/         full predefined architectures
│   ├── dense.py            dense transformer (decoder only)
│   ├── moe.py              mixture-of-experts transformer
│   └── language.py         LanguageModel (PreTrainedModel) + LanguageModelConfig
└── tasks/          training/eval code, organised by modality
    └── language/
        ├── loader.py       load_language_model, load_dataset, load_tokenizer
        ├── evaluators.py   MyLMEval (lm-eval harness base class)
        └── eval_entry.py   legacy lm-eval registry wrapper
```

## quick start

run from the project root (`arcadium/`):

```bash
bash examples/simple.sh
```

that calls `pretrain_language.py` which uses:
- `arcadium.tasks.language.loader` — model + dataset loading
- `arcadium.data.sequence_length` — sequence length curriculum sampler
- `arcadium.optimizers.loader` — optimizer factory

## adding stuff

- new dataset → `data/`
- new positional encoding → `embeddings/`
- new optimizer → `optimizers/`
- new attention or mixer → `components/attentions/` or `components/associators/`
- new backbone (e.g. mamba) → `models/`
- new training script (e.g. vision) → `tasks/vision/`
