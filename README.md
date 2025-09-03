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

# Progress

*Before I began logging*
- Added VRAM estimates to see how VRAM is being utilized
- Interesting things from a HW pers

*8/31/2025*: 
- Pretraining seems to work, noting this as the lm_loss (cross entropy) seems to be going down. 
- Looks okay at a high level ... but then my tokens are all garbled in generation? Getting only "the"s as output.
- Could it be because we don't have enough useful data? Or because our param count is low? Let's try plotting attention maps, this seems like the quickest way to add in more interpretability

*9/1/2025*
- Updated the loss function, I'm no longer just pretraining based on the next token. I.e previously, I had it set up so that we were predicting a distribution of P(t_k | t_k_1, t_k-2, ...), i.e predicting the next token from a preexisting sequence. That may have been why my model was so bad! I made it more "next-token-like" by having the labels be the entire sequence shifted over by 1, i.e, each sequence's label is literally the next token now. Huh ... so that means that previously, my model was using the single next token for the sequence as labels for all tokens???  **TODO: expand more on this in a blog?**
- Output's has a LOT less the's. Yay! Now, it's sensible within small parts of the sentence, but doesn't seem to know how to properly start and end. Hmmm ... what to do? Some ideas:
    - Configure the loss function to specifically consider beginning-of-sequence tokens and end-tokens?
    - Just scale up data/params? Maybe this is emergent behaviour?
    - How do you know whether you are being limited by your sequence length during training?

*9/3/2025*
- Tried varying the input/output/embedding dimension. I'm thinking of just fixing this as one param.

- Seems like increasing dim from 64 -> 256 does have an effect on sensibility. New output - 

```python
"""Output text Shakespeare produced most of his known works between 1589 and 1613. His early plays were primarily comedies and histories and are regarded as some of the best works produced in these genres. He then wrote mainly tragedies until 1608, among them Hamlet, Othello, King James VI Losing team: Protégé project manager on the best works produced in these genres. He then wrote mainly tragedies until 1608, among them Hamlet, Othello, King Lear and Macbeth, all considered to be among them Hamlet, Othello, King Lear and Macbeth, all considered to be among the sources of the sources of the sources of the sources of"""
```

So, we've went from incoherant 'the's --> somewhat coherent partial sentences --> coherent but repetitive sentences.
