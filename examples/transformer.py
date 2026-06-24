# examples/transformer.py
from dataclasses import dataclass
import optax
from transformers import AutoTokenizer
from axiom import ax, nn
from axionn.nn import MHA
from axionn.training import build_trainer, LocalRunLogger
from axionn.data.stream import build_topological_stream

@dataclass
class Config:
    v = ax.v(32000)  # Mistral 32k vocabulary size
    b = ax.b(32)
    s = ax.s(512)
    d = ax.d(128)


cfg = Config()


@ax.remat
def layer(x, h=4, d=4):
    x = x + MHA(x.d.rms_norm(), num_heads=h, head_dim=d)
    g, u = (x & x).d.rms_norm().d.proj()
    return x + (g * u.silu()).d.proj()


@ax.model
def transformer(tokens, depth=6):
    x = nn.embed(tokens, cfg.v, cfg.d)
    for _ in range(depth):
        x = layer(x)
    return x.d.proj(cfg.v)


# 1. Initialize Model and Load public Mistral 32k Tokenizer
transformer.init(cfg.b, cfg.s)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# 2. Build Topological Stream with our tokenizer convolved over raw text
dataset_path = "roneneldan/TinyStories"
stream = build_topological_stream(dataset_path, seq_ax=cfg.s, batch_ax=cfg.b, tokenizer=tokenizer)

# 3. Build Trainer Generator (using our Adam optimizer and build_trainer)
optimizer = optax.adam(1e-3)
trainer = build_trainer(transformer, optimizer, autoregressive=cfg.s)

# Initialize LocalRunLogger with configuration hyperparameters
logger = LocalRunLogger(
    run_name="transformer_tinystories",
    config={
        "vocab_size": cfg.v.size,
        "batch_size": cfg.b.size,
        "seq_len": cfg.s.size,
        "emb_dim": cfg.d.size,
        "learning_rate": 1e-3,
        "depth": 6
    }
)

# 4. Train using the generator .send() interface and log progress!
print("Training on dynamically tokenized TinyStories with Mistral 32k...")
for i in range(1000):
    x_batch = next(stream)
    loss = trainer.send(x_batch)
    
    # Calculate tokens processed in this batch to log throughput
    tokens_processed = cfg.b.size * cfg.s.size

    if i % 10 == 0 or i == 0:
        logger.log(step=i, metrics={"loss": loss}, tokens_processed=tokens_processed)