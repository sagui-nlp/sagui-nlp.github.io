---
layout: default
title:  "NanoGPT ao nanoMoE: Part‑1"
date:   2025-08-23 22:03:16 -0300
categories: moe
---
# *NanoGPT ao nanoMoE*: Part‑1

## Introdução

*NanoGPT* é uma implementação minimalista do GPT‑2 escrita com foco em **simplicidade** e **didática**. O código facilita a leitura e a modificação de cada etapa de um Transformer, dispensando dependências externas complexas (por exemplo, *Hugging Face* ou *PyTorch*).
Graças a essa leveza, a comunidade passou a usá‑lo como **ponto de partida** para testar ideias, ajustar hiperparâmetros e, principalmente, criar extensões—como esta série que transforma o modelo em um *Mixture of Experts* (MoE).

> Nos próximos posts vamos dissecar a arquitetura original, mostrar o processo de treinamento e, em seguida, detalhar as alterações necessárias para inserir uma camada MoE: *gate*, *top‑k*, métricas de balanceamento e o impacto na perplexidade.

---

## Por que começar por um modelo pequeno?

* **Iteração rápida**: treinos que duram minutos permitem ajustar código e hiperparâmetros sem esperar horas.
* **Custo baixo**: consome pouca GPU/CPU e evita otimizações prematuras de infraestrutura.
* **Diagnóstico claro**: *overfitting*, instabilidade de *loss* ou bugs aparecem de forma mais visível.
* **Intuição incremental**: mudar apenas *um* hiperparâmetro (`n_layer`, `n_head`, `n_embd`, `dropout`) e observar o efeito direto em *loss*/*perplexity*.
* **Sanidade da *pipeline***: garantir que tokenização, *shift* de *targets* e geração funcionam (overfit controlado em um único parágrafo).
* **Baseline mínima**: estabelece referência para comparar futuras melhorias (por exemplo, adicionar MoE) e medir ganho real.
* **Facilidade de depuração**: menos parâmetros reduzem ruído ao investigar gradientes ou explosões numéricas.
* **Preparação para escalar**: entender limites (quando saturar) antes de investir em modelos maiores.

---

## Entendendo o nanoGPT através dos parâmetros

A classe `GPTConfig` define a *personalidade* do modelo. Cada parâmetro influencia diretamente a capacidade, o desempenho e o comportamento do nanoGPT durante o treinamento e a geração de texto.

```python
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT‑2 vocab_size (50257) arredondado para múltiplo de 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias em Linear + LayerNorm (como GPT‑2). False: ligeiramente melhor e mais rápido
```

* **block\_size**: tamanho do contexto. Nesta implementação a atenção é *quadrática*—diferentemente de modelos recentes, como o *Mixtral*, que usam abordagens como *SMoE*.
* **vocab\_size**: número de tokens distintos que o modelo reconhece.
* **n\_layer**: quantidade de camadas *decoder* do Transformer. Mais camadas ≈ padrões mais complexos, porém custo de `O(block_size² * n_embd)`.
* **n\_head**: cabeças de atenção; partem `n_embd` em `n_head` fatias. Mais cabeças aprendem relações paralelas adicionais.
* **n\_embd**: dimensão do *embedding* dos tokens. Valores maiores aumentam a **capacidade de representação**.
* **dropout**: probabilidade de desativar neurônios durante treino. Com dados pequenos ajuda na generalização, mas pode atrasar a convergência em corpora grandes.
* **bias**: controla o uso de *bias* em camadas lineares e de normalização. Transformers mais recentes tendem a omitir.

---

## Como avaliar um GPT?

*Métricas utilizadas*

### Loss (Cross‑Entropy)

$$ \mathcal{L}_{CE} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) $$

### Perplexity

Perplexidade é `exp(loss)`. Antes do treino seu valor tende a ser próximo ao tamanho do vocabulário; valores perto de **1** indicam que o modelo memorizou os dados.

$$ \mathrm{PPL} = \exp(\mathcal{L}_{CE}) $$

---

## Estrutura do *script* de treino

1. **Tokenização** de um único texto — teste de sanidade.
2. **Criação dos alvos** deslocando 1 token (*next‑token prediction*); o último token é ignorado pela loss.
3. **Treino por algumas épocas**, acompanhando *loss* e *perplexity*.
4. **Avaliação** a cada 25 % das épocas:

   * **Reconstrução**: previsão token‑a‑token em relação ao texto de treino.
   * **Continuação**: o modelo recebe um trecho e completa até o número de tokens desejados.

Esse ciclo valida toda a *pipeline* (*tokenização → shift → forward → loss → geração*) antes de aumentar a complexidade.

---

## Preparando os dados

```python
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

targets = input_ids.clone()
targets[:, :-1] = input_ids[:, 1:]
targets[:, -1] = -1  # ignora o último token na loss
```

O tokenizer escolhido é uma versão pt‑BR do GPT‑2: [`pierreguillou/gpt2-small-portuguese`](https://huggingface.co/pierreguillou/gpt2-small-portuguese).

---

## Parâmetros principais de treinamento

```python
# --- Training params ---
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 1e-2
BETAS = (0.9, 0.95)
CONT_PREFIX_TOKENS = 10  # tokens de prefixo para geração
device = "cuda" if torch.cuda.is_available() else "cpu"
```

* **EPOCHS**: mais iterações ⇒ mais chance de memorizar (perplexity cai e estabiliza).
* **LR** (*learning rate*): valores altos aceleram mas podem oscilar; baixos são estáveis porém lentos.
* **WEIGHT\_DECAY**: regularização para retardar *overfitting*.
* **BETAS**: inércia do *AdamW*.
* **CONT\_PREFIX\_TOKENS**: quantos tokens iniciais mantemos para a continuação (ex.: 10 → 10 tokens → até 25 novos tokens gerados).

---

## Config mínima do modelo

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

Altere `n_layer`, `n_head` e `n_embd` para sentir a variação de capacidade vs. custo.

---

## Teste inicial

### Medindo *Loss* e Perplexity

```python
with torch.no_grad():
    logits, loss = model(input_ids, targets=targets)
init_ppl = torch.exp(loss).item() if loss is not None else float("nan")
print("Loss inicial:", loss.item())  # variante usada na segunda versão
```

### Geração com prefixo curto

```python
generated_ids = model.generate(
    input_ids[:, :CONT_PREFIX_TOKENS],
    max_new_tokens=10,
    temperature=0.1,
    top_k=1,
)
# temperature baixa + top_k=1 ≈ quase greedy.
```

---

## Loop de treino

```python
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()

    if is_quarter(epoch + 1):
        # imprime métricas + geração
```

Checkpoint a cada 25 %:

```python
def is_quarter(e):
    return e == EPOCHS or e % (EPOCHS // 4) == 0
```

---

## Execução direta

```bash
pip install uv
uv sync
uv run -m scripts.01_train_nanogpt  # ou scripts/01_train_nanogpt.py
```

---

## Próximo post

Passaremos da arquitetura básica para o **MoE**: implementação do *gate*, escolha de *top‑k*, métricas de balanceamento e análise do impacto na perplexidade.

Código completo: consulte [`scripts/01_train_nanogpt.py`](https://github.com/sagui-nlp/nanoGPT-moe/blob/feat/blog-writing/scripts/01_train_nanogpt.py) no repositório.

