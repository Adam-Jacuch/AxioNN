# AxioNN

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**AxioNN** is a high-level, topologically-aware functional neural network library built on top of **[Axiom](https://github.com/Adam-Jacuch/Axiom2)**. It provides a complete suite of elegant, JIT-compilable, topologically-safe deep learning modules, optimization/alignment steps, data-streaming utilities, and trainer primitives.

---

## Why AxioNN?

In traditional deep learning libraries, developers must constantly track tensor dimensions mentally or write error-prone code using manual reshapes, squeezes, and permutations (e.g., `x.view(B, S, H, D)` or `einops.rearrange`). 

**AxioNN leverages Axiom's named-axis system**, where every `Tensor` holds a formal topology of named, sized `Axis` objects. AxioNN modules use this topology to automatically orchestrate projections, head-splitting, channel merging, and reductions. 
- **Topologically-Safe**: Shape mismatches are caught during topology resolution rather than runtime.
- **Zero Permutations**: Dynamic projection and targeted axis operations completely eliminate the need for manual dimension swapping.
- **Unified & Pure**: Fully functional, side-effect-free steps that integrate natively with JAX and Optax.

---

## Core Features

### 🚀 Topologically-Safe Attention Modules (`axionn.nn`)
- **Grouped Query Attention (GQA)**: Automatically orchestrates query projections, head-splitting, key/value replication across groups, and output projection.
- **Multi-Head Attention (MHA)**: Realized as a clean, symmetric variation of GQA.
- **Sliding Window Attention (SWA)**: Restricts token attention to local causal or non-causal windows using dynamic relative position masks.
- **Linear Attention**: Replaces softmax with a kernel feature map ($\text{elu}(x) + 1$) to achieve $O(N)$ space and time complexity for long context lengths.

### 🌀 Unified $N$-Dimensional Convolutions
- **`conv`**: A single convolution primitive capable of convolving over arbitrary spatial dimensions, merging multi-channel topologies, and projecting back to targeted configurations. Supports same/valid padding and causal masking.

### 📈 Stable State-Space Models (SSMs)
- **`ssm_scan`**: Expressive state-space scan primitive implementing $h_t = A_t \odot h_{t-1} + X'_t$ via Axiom's parallel associative or sequential scan engine. Supports different gating modes (`convex`, `norm_preserving`, `stable`, `linear`) and activations (`sigmoid`, `softplus`, `none`).

### 🤝 State-of-the-Art Alignment Primitives (`axionn.alignment`)
- **Direct Preference Optimization (DPO)**: Standard preference-alignment step using a reference and policy model completely outside/inside the gradient tape.
- **Kahneman-Tversky Optimization (KTO)**: Stateless utility-maximization step that operates on unpaired datasets with binary desirability flags.

### ⚡ Zero-Waste Streaming DataLoader (`axionn.data`)
- **`StatefulTopologicalStream`**: Infinite sequence-packing dataloader streaming directly from Hugging Face Datasets. Supports on-the-fly tokenization, near-zero RAM footprint, and complete state serialization (`state_dict` / `load_state_dict`) for seamless preemption/resumption.

### 🛠️ Unified Training & Telemetry (`axionn.training`)
- **`Trainer`**: Stateful class that compiles models, auto-initializes optimizer states, executes JIT-compiled training steps (`.step()`), and returns step-by-step losses.
- **`LocalRunLogger`**: Fast, structured local logger that saves configuration hyperparameters, writes metrics to JSON Lines (`metrics.jsonl`), and logs real-time throughput metrics (tokens/sec).

---

## Installation & Hardware Binding

AxioNN is designed to be highly portable and **hardware-agnostic**. By default, it does not enforce any specific JAX or Axiom compilation hardware targets in its package dependencies. It will **seamlessly bind to whichever JAX variant** (CPU, CUDA, ROCm, TPU) is currently installed in your Python environment.

### 1. Add to your project with `uv` (Recommended)

To add AxioNN directly to your project using [**`uv`**](https://github.com/astral-sh/uv), simply run:

```bash
uv add git+https://github.com/Adam-Jacuch/AxioNN.git
```

`uv` will automatically resolve all package dependencies, fetch the required functional **Axiom** backend, and lock the environment.

### 2. Install using `pip` (or equivalent fallback)

You can install AxioNN directly into your active Python environment:

```bash
pip install git+https://github.com/Adam-Jacuch/AxioNN.git
```

## Quick Start: Training a Transformer on TinyStories

Below is a complete, runnable example demonstrating how to stream a dataset, build a topologically-safe transformer with Multi-Head Attention, and train it using JIT-compiled steps and telemetry tracking.

```python
from dataclasses import dataclass
import optax
from transformers import AutoTokenizer
from axiom import ax, nn
from axionn.nn import MHA
from axionn.training import Trainer, LocalRunLogger
from axionn.data.stream import build_topological_stream

# 1. Define structured input/output axes
@dataclass
class Config:
    v = ax.v(32000)  # Vocabulary size
    b = ax.b(32)     # Batch size
    s = ax.s(512)    # Sequence length
    d = ax.d(128)    # Model dimension

cfg = Config()

# 2. Build model architecture using functional layers
@ax.remat
def layer(x, h=4, d=4):
    # Apply Root Mean Square Norm followed by MHA
    x = x + MHA(x.d.rms_norm(), num_heads=h, head_dim=d)
    
    # MLP block: rms_norm -> project -> gated silu activation
    g, u = (x & x).d.rms_norm().d.proj()
    return x + (g * u.silu()).d.proj()

@ax.model
def transformer(tokens, depth=6):
    # Embed tokens and pass through layers
    x = nn.embed(tokens, cfg.v, cfg.d)
    for _ in range(depth):
        x = layer(x)
    return x.d.proj(cfg.v)

# Initialize model parameters dynamically based on input topology
transformer.init(cfg.b, cfg.s)

# 3. Set up the dataset streaming pipeline
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
stream = build_topological_stream(
    "roneneldan/TinyStories", 
    seq_ax=cfg.s, 
    batch_ax=cfg.b, 
    tokenizer=tokenizer
)

# 4. Build Trainer and Local Logger
optimizer = optax.adam(1e-3)
trainer = Trainer(transformer, optimizer, autoregressive=cfg.s)

logger = LocalRunLogger(
    run_name="transformer_stories",
    config={
        "vocab_size": cfg.v.size,
        "batch_size": cfg.b.size,
        "seq_len": cfg.s.size,
        "emb_dim": cfg.d.size,
        "learning_rate": 1e-3,
        "depth": 6
    }
)

# 5. Execute JIT-compiled training loops
print("Training started...")
for i in range(100):
    x_batch = next(stream)
    loss = trainer.step(x_batch)  # Compiled step forward/backward
    
    # Log metrics & throughput (tokens/sec)
    tokens_processed = cfg.b.size * cfg.s.size
    if i % 10 == 0:
        logger.log(step=i, metrics={"loss": loss}, tokens_processed=tokens_processed)
```

---

## API Showcase

### 1. Grouped Query Attention (GQA)
Enables memory-efficient decoding by sharing key/value heads across query groups.

```python
from axionn.nn import GQA

# GQA automatically projects, splits heads, duplicates KV groups, 
# computes attention, merges heads, and projects back to model dimension.
out = GQA(
    x, 
    num_queries_heads=8, 
    num_kv_heads=2, 
    head_dim=64, 
    mask=causal_mask
)
```

### 2. State-Space Model Scan (`ssm_scan`)
Computes deep linear recurrences over sequence dimensions with parallel associative prefix-scan efficiency.

```python
from axionn.nn import ssm_scan

# Execute an associative parallel scan
h = ssm_scan(
    x,                 # Input sequence tensor
    a,                 # Transition/decay parameters
    seq_ax=seq,        # Axis representing the time dimension
    mode="convex",     # Stable convex interpolation: (1 - A) * x + A * h_{t-1}
    decay_activation="sigmoid"
)
```

### 3. Direct Preference Optimization (DPO)
Performs preference alignment directly on log probabilities without training separate reward models.

```python
from axionn.alignment.stateless import dpo_step

# Runs a single functional step updating the policy model
policy_model, optimizer_state, loss = dpo_step(
    policy_model=policy_model,
    ref_model=ref_model,
    optim=(optimizer, opt_state),
    chosen_x=chosen_tokens_tensor,
    rejected_x=rejected_tokens_tensor,
    seq_ax=seq_axis,
    beta=0.1
)
```

---

## Running Tests

All components are rigorously tested. You can run unit tests directly using pytest:

```bash
uv run pytest
```

---

## License

This project is licensed under the [MIT License](LICENSE).
