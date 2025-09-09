---
layout: default
title:  "NanoGPT ao nanoMoE: Part‑1"
date:   2025-08-23 22:03:16 -0300
categories: moe
---
# *NanoGPT ao nanoMoE*: Part‑1


## Introduction

*NanoGPT* is a minimalist implementation of GPT-2 written with a focus on **simplicity** and **clarity**. The code makes it easy to read and modify every step of a Transformer, avoiding complex external dependencies (e.g., *Hugging Face*).
Thanks to this lightness, we can use it as a **starting point** to test ideas, tweak hyper-parameters, and—most importantly—create extensions, such as turning the model into a *Mixture of Experts* (MoE).

> In the next posts we’ll dissect the original architecture, show the training process, and then detail the changes required to insert an MoE layer: *gate*, *top-k*, balancing metrics, and the impact on perplexity.

---

## Why start with a small model?

* **Fast iteration**: training runs that last minutes let you tweak code and hyper-parameters without waiting hours.
* **Low cost**: consumes little GPU/CPU and avoids premature infrastructure optimizations.
* **Clear diagnostics**: *overfitting*, *loss* instability, or bugs appear more visibly.
* **Incremental intuition**: change only *one* hyper-parameter (`n_layer`, `n_head`, `n_embd`, `dropout`) and observe the direct effect on *loss*/*perplexity*.
* **Pipeline sanity**: ensure tokenization, target *shift*, and generation work (controlled overfit on a single paragraph).
* **Minimal baseline**: establishes a reference to compare future improvements (e.g., adding MoE) and measure real gains.
* **Ease of debugging**: fewer parameters reduce noise when investigating gradients or numerical explosions.
* **Preparation for scaling**: understand limits (when it saturates) before investing in larger models.

---

## Understanding nanoGPT through its parameters

The `GPTConfig` class defines the model’s *personality*. Each parameter directly influences the capacity, performance, and behavior of nanoGPT during training and text generation.

```python
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size (50257) rounded to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linear + LayerNorm (like GPT-2). False: slightly better and faster
```

* **block\_size**: context length. In this implementation, attention is *quadratic*.
* **vocab\_size**: number of distinct tokens the model recognizes.
* **n\_layer**: number of Transformer *decoder* layers. More layers ≈ more complex patterns, but cost `O(block_size² * n_embd)`.
* **n\_head**: attention heads; split `n_embd` into `n_head` slices. More heads learn additional parallel relations.
* **n\_embd**: token *embedding* dimension. Larger values increase **representation capacity**.
* **dropout**: probability of dropping neurons during training. On small data it helps generalization, but on large corpora it can slow convergence.
* **bias**: controls the use of *bias* in linear and normalization layers. Newer Transformers tend to omit it.

---

## How do you evaluate a GPT?

*Metrics used*

### Loss (Cross-Entropy)

$\mathcal{L}_{CE} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$

### Perplexity

Perplexity is `exp(loss)`. Before training its value tends to be close to the vocabulary size; values near **1** indicate the model memorized the data.

$\mathrm{PPL} = \exp(\mathcal{L}_{CE})$

---

## Training script structure

1. **Tokenize** a single text — sanity test.
2. **Create targets** by shifting 1 token (*next-token prediction*); the last token is ignored by the loss.
3. **Train for a few epochs**, tracking *loss* and *perplexity*.
4. **Evaluate** every 25 % of the epochs:

   * **Reconstruction**: token-by-token prediction versus the training text.
   * **Continuation**: the model receives a snippet and completes up to the desired number of tokens.

This cycle validates the entire *pipeline* (*tokenization → shift → forward → loss → generation*) before increasing complexity.

---

## Preparing the data

```python
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

targets = input_ids.clone()
targets[:, :-1] = input_ids[:, 1:]
targets[:, -1] = -1  # ignore the last token in the loss
```

The tokenizer chosen is a Brazilian Portuguese version of GPT-2: [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese).

---

## Main training parameters

```python
# --- Training params ---
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.95)
CONT_PREFIX_TOKENS = 10  # prefix tokens for generation
device = "cuda" if torch.cuda.is_available() else "cpu"
```

* **EPOCHS**: more iterations ⇒ higher chance of memorization (perplexity falls then plateaus).
* **LR** (*learning rate*): high values speed up but may oscillate; low values are stable but slow.
* **WEIGHT\_DECAY**: regularization to delay *overfitting*.
* **BETAS**: inertia of *AdamW*.
* **CONT\_PREFIX\_TOKENS**: how many initial tokens we keep for continuation (e.g., 10 → 10 tokens → up to 25 new tokens generated).

---

## Minimal model config

```python
config = GPTConfig(
    block_size=100,
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.2,
    bias=True,
)
model = GPT(config).to(device)
```

Change `n_layer`, `n_head`, and `n_embd` to feel the capacity-versus-cost trade-off.

---

## Initial test

### Measuring *Loss* and Perplexity

```python
with torch.no_grad():
    logits, loss = model(input_ids, targets=targets)
init_ppl = torch.exp(loss).item() if loss is not None else float("nan")
print("Initial loss:", loss.item())  # variant used in the second version
```

### Generation with a short prefix

```python
generated_ids = model.generate(
    input_ids[:, :CONT_PREFIX_TOKENS],
    max_new_tokens=10,
    temperature=0.1,
    top_k=1,
)
# low temperature + top_k=1 ≈ almost greedy.
```

---

## Training loop

```python
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()

    if is_quarter(epoch + 1):
        # print metrics + generation
```

Checkpoint every 25 %:

```python
def is_quarter(e):
    return e == EPOCHS or e % (EPOCHS // 4) == 0
```

---

## Direct execution

```bash
pip install uv
uv sync
uv run -m scripts.01_train_nanogpt  # or scripts/01_train_nanogpt.py
```

---

## Next post

We’ll move from the basic architecture to **MoE**: gate implementation, *top-k* choice, balancing metrics, and analysis of the impact on perplexity.

Full code: see [`scripts/01_train_nanogpt.py`](https://github.com/sagui-nlp/nanoGPT-moe/blob/feat/blog-writing/scripts/01_train_nanogpt.py) in the repository.

