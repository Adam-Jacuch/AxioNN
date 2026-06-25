# tests/test_axionn.py

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from unittest.mock import patch
import jax.numpy as jnp
import optax
import axiom
from axiom.core import Axis, wrap, Tensor
from axionn.nn.attention import GQA, MHA
from axionn.training.steps import autoregressive_ce_step, mse_step, Trainer
from axionn.data.stream import build_topological_stream
from axionn.alignment.stateless import dpo_step, kto_step
from axionn.nn import ssm_scan, conv


def test_gqa_forward():
    # 1. Define input axes
    batch = Axis("batch", 2)
    seq = Axis("seq", 16)
    d_model = Axis("d_model", 32)
    
    # 2. Wrap inputs
    x = wrap(jnp.ones((2, 16, 32)), batch, seq, d_model)
    
    # 3. Build model
    @axiom.ax.model
    def model(inp):
        return GQA(inp, num_queries_heads=4, num_kv_heads=2, head_dim=8)
    
    model.init(batch, seq, d_model)
    out = model(x)
    
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, d_model)
    assert out.unwrap().shape == (2, 16, 32)


def test_mha_forward():
    # 1. Define input axes
    batch = Axis("batch", 2)
    seq = Axis("seq", 16)
    d_model = Axis("d_model", 32)
    
    # 2. Wrap inputs
    x = wrap(jnp.ones((2, 16, 32)), batch, seq, d_model)
    
    # 3. Build model
    @axiom.ax.model
    def model(inp):
        return MHA(inp, num_heads=4, head_dim=8)
    
    model.init(batch, seq, d_model)
    out = model(x)
    
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, d_model)
    assert out.unwrap().shape == (2, 16, 32)


def test_mse_step_sgd():
    batch = Axis("batch", 4)
    d_model = Axis("d_model", 16)
    d_out = Axis("d_out", 8)
    
    x = wrap(jnp.ones((4, 16)), batch, d_model)
    y = wrap(jnp.ones((4, 8)), batch, d_out)
    
    @axiom.ax.model
    def model(inp):
        return getattr(inp, "d_model").proj(d_out)
    
    model.init(batch, d_model)
    orig_params = {k: v.copy() for k, v in model.params.items()}
    
    # Run step using SGD (float learning rate)
    updated_model, updated_optim, loss = mse_step(model, 0.01, x, y)
    
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert updated_optim == 0.01
    
    # Ensure params are updated
    assert any(jnp.any(orig_params[k] != updated_model.params[k]) for k in orig_params)


def test_mse_step_optax():
    batch = Axis("batch", 4)
    d_model = Axis("d_model", 16)
    d_out = Axis("d_out", 8)
    
    x = wrap(jnp.ones((4, 16)), batch, d_model)
    y = wrap(jnp.ones((4, 8)), batch, d_out)
    
    @axiom.ax.model
    def model(inp):
        return getattr(inp, "d_model").proj(d_out)
    
    model.init(batch, d_model)
    orig_params = {k: v.copy() for k, v in model.params.items()}
    
    optimizer = optax.adam(1e-3)
    opt_state = None  # Will be auto-initialized by apply_updates
    
    updated_model, updated_optim, loss = mse_step(model, (optimizer, opt_state), x, y)
    
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert isinstance(updated_optim, tuple)
    assert updated_optim[0] == optimizer
    assert updated_optim[1] is not None
    
    # Ensure params are updated
    assert any(jnp.any(orig_params[k] != updated_model.params[k]) for k in orig_params)


def test_autoregressive_ce_step_sgd():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    
    tokens = wrap(jnp.array([[0, 2, 4, 1, 9, 3, 5, 7], [1, 1, 3, 5, 2, 4, 6, 8]]), batch, seq)
    
    @axiom.ax.model
    def model(toks):
        vocab_ax = Axis("vocab", 10)
        embed_ax = Axis("embed", 16)
        x_emb = axiom.nn.embed(toks, vocab_ax, embed_ax)
        return getattr(x_emb, embed_ax.name).proj(vocab_ax)
    
    model.init(batch, seq)
    orig_params = {k: v.copy() for k, v in model.params.items()}
    
    updated_model, updated_optim, loss = autoregressive_ce_step(model, 0.01, tokens, seq)
    
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert updated_optim == 0.01
    
    assert any(jnp.any(orig_params[k] != updated_model.params[k]) for k in orig_params)


def test_autoregressive_ce_step_optax():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    
    tokens = wrap(jnp.array([[0, 2, 4, 1, 9, 3, 5, 7], [1, 1, 3, 5, 2, 4, 6, 8]]), batch, seq)
    
    @axiom.ax.model
    def model(toks):
        vocab_ax = Axis("vocab", 10)
        embed_ax = Axis("embed", 16)
        x_emb = axiom.nn.embed(toks, vocab_ax, embed_ax)
        return getattr(x_emb, embed_ax.name).proj(vocab_ax)
    
    model.init(batch, seq)
    orig_params = {k: v.copy() for k, v in model.params.items()}
    
    optimizer = optax.adamw(1e-3)
    opt_state = None
    
    updated_model, updated_optim, loss = autoregressive_ce_step(model, (optimizer, opt_state), tokens, seq)
    
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert isinstance(updated_optim, tuple)
    assert updated_optim[0] == optimizer
    assert updated_optim[1] is not None
    
    assert any(jnp.any(orig_params[k] != updated_model.params[k]) for k in orig_params)


def test_build_topological_stream():
    # Mock dataset examples
    mock_examples = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8, 9, 10]},
        {"input_ids": [11, 12, 13, 14, 15]},
    ]
    
    class MockDataset:
        def shuffle(self, **kwargs):
            return self
        
        def __iter__(self):
            return iter(mock_examples)
            
    with patch("axionn.data.stream.load_dataset", return_value=MockDataset()):
        batch = Axis("batch", 2)
        seq = Axis("seq", 4)
        
        stream = build_topological_stream("dummy_path", seq_ax=seq, batch_ax=batch)
        
        # Take the first batch
        x = next(stream)
        
        assert isinstance(x, Tensor)
        assert x.topology == (batch, seq)
        assert jnp.array_equal(x.unwrap(), jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]]))


def test_build_topological_stream_with_tokenizer():
    # Mock raw text examples
    mock_examples = [
        {"text": "hello"},
        {"text": "world"},
        {"text": "axiom"},
    ]
    
    class MockDataset:
        def shuffle(self, **kwargs):
            return self
        
        def __iter__(self):
            return iter(mock_examples)
            
    class MockTokenizer:
        def encode(self, text):
            return [ord(c) for c in text]

    with patch("axionn.data.stream.load_dataset", return_value=MockDataset()):
        batch = Axis("batch", 2)
        seq = Axis("seq", 4)
        
        stream = build_topological_stream("dummy_path", seq_ax=seq, batch_ax=batch, tokenizer=MockTokenizer())
        
        x = next(stream)
        assert isinstance(x, Tensor)
        assert x.topology == (batch, seq)
        # hello (104, 101, 108, 108, 111), world (119, 111, 114, 108, 100) -> pack [104, 101, 108, 108] and [111, 119, 111, 114]
        expected = jnp.array([[104, 101, 108, 108], [111, 119, 111, 114]])
        assert jnp.array_equal(x.unwrap(), expected)


def test_build_topological_stream_preemption():
    # Mock dataset examples
    mock_examples = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8, 9, 10]},
        {"input_ids": [11, 12, 13, 14, 15]},
    ]
    
    class StatefulMockIterator:
        def __init__(self, examples):
            self.examples = examples
            self.index = 0
            
        def __next__(self):
            if self.index >= len(self.examples):
                raise StopIteration
            res = self.examples[self.index]
            self.index += 1
            return res
            
        def state_dict(self):
            return {"index": self.index}
            
        def load_state_dict(self, state):
            self.index = state["index"]
            
    class MockDataset:
        def shuffle(self, **kwargs):
            return self
        
        def __iter__(self):
            return StatefulMockIterator(mock_examples)
            
    with patch("axionn.data.stream.load_dataset", return_value=MockDataset()):
        batch = Axis("batch", 1)
        seq = Axis("seq", 4)
        
        stream = build_topological_stream("dummy_path", seq_ax=seq, batch_ax=batch)
        
        # Take first batch: 1, 2, 3, 4
        x1 = next(stream)
        assert jnp.array_equal(x1.unwrap(), jnp.array([[1, 2, 3, 4]]))
        
        # Save state! (The token_buffer now has [5])
        state = stream.state_dict()
        
        # Take second batch: 5, 6, 7, 8
        x2 = next(stream)
        assert jnp.array_equal(x2.unwrap(), jnp.array([[5, 6, 7, 8]]))
        
        # Restore state!
        stream.load_state_dict(state)
        
        # Take second batch again (should be identical to x2!)
        x2_restored = next(stream)
        assert jnp.array_equal(x2_restored.unwrap(), jnp.array([[5, 6, 7, 8]]))


def test_dpo_step():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    
    @axiom.ax.model
    def dummy_model(tokens):
        vocab_ax = Axis("vocab", 10)
        embed_ax = Axis("embed", 16)
        x_emb = axiom.nn.embed(tokens, vocab_ax, embed_ax)
        return getattr(x_emb, embed_ax.name).proj(vocab_ax)

    dummy_model.init(batch, seq)
    orig_params = {k: v.copy() for k, v in dummy_model.params.items()}
    
    ref_model = axiom.compiler.AxiomModel(dummy_model.fn, {k: v.copy() for k, v in dummy_model.params.items()})
    ref_model.is_initialized = True
    
    chosen = wrap(jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]]), batch, seq)
    rejected = wrap(jnp.array([[8, 7, 6, 5, 4, 3, 2, 1], [9, 8, 7, 6, 5, 4, 3, 2]]), batch, seq)
    
    updated_model, updated_optim, loss = dpo_step(dummy_model, ref_model, 0.01, chosen, rejected, seq)
    
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert updated_optim == 0.01
    assert any(jnp.any(orig_params[k] != updated_model.params[k]) for k in orig_params)


def test_kto_step():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    
    @axiom.ax.model
    def dummy_model(tokens):
        vocab_ax = Axis("vocab", 10)
        embed_ax = Axis("embed", 16)
        x_emb = axiom.nn.embed(tokens, vocab_ax, embed_ax)
        return getattr(x_emb, embed_ax.name).proj(vocab_ax)

    dummy_model.init(batch, seq)
    orig_params = {k: v.copy() for k, v in dummy_model.params.items()}
    
    ref_model = axiom.compiler.AxiomModel(dummy_model.fn, {k: v.copy() for k, v in dummy_model.params.items()})
    ref_model.is_initialized = True
    
    tokens = wrap(jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]), batch, seq)
    is_desirable = wrap(jnp.array([1.0, 0.0]), batch)
    
    updated_model, updated_optim, loss = kto_step(dummy_model, ref_model, 0.01, tokens, is_desirable, seq)
    
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert updated_optim == 0.01
    assert any(jnp.any(orig_params[k] != updated_model.params[k]) for k in orig_params)


# --- SSM Scan Tests ---

def test_ssm_scan_modes():
    # 1. Define input axes
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    d_model = Axis("d_model", 16)

    # 2. Wrap inputs
    x = wrap(jnp.ones((2, 8, 16)) * 0.5, batch, seq, d_model)
    a = wrap(jnp.ones((2, 8, 16)) * -0.5, batch, seq, d_model)

    for mode in ["convex", "norm_preserving", "stable", "linear"]:
        out = ssm_scan(x, a, seq_ax=seq, mode=mode, associative=True)
        assert isinstance(out, Tensor)
        assert out.topology == (batch, seq, d_model)
        assert out.unwrap().shape == (2, 8, 16)


def test_ssm_scan_sequential_vs_associative():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    d_model = Axis("d_model", 16)

    x = wrap(jnp.ones((2, 8, 16)) * 0.5, batch, seq, d_model)
    a = wrap(jnp.ones((2, 8, 16)) * -0.2, batch, seq, d_model)

    for mode in ["convex", "norm_preserving", "stable"]:
        out_assoc = ssm_scan(x, a, seq_ax=seq, mode=mode, associative=True)
        out_seq = ssm_scan(x, a, seq_ax=seq, mode=mode, associative=False)

        # Check that associative and sequential scans yield the exact same numerical result
        assert jnp.allclose(out_assoc.unwrap(), out_seq.unwrap(), atol=1e-5)


def test_ssm_scan_decay_activations():
    batch = Axis("batch", 2)
    seq = Axis("seq", 4)
    d_model = Axis("d_model", 8)

    x = wrap(jnp.ones((2, 4, 8)), batch, seq, d_model)
    a = wrap(jnp.ones((2, 4, 8)), batch, seq, d_model)

    for activation in ["sigmoid", "softplus", "none"]:
        out = ssm_scan(x, a, seq_ax=seq, mode="convex", decay_activation=activation)
        assert isinstance(out, Tensor)
        assert out.topology == (batch, seq, d_model)


def test_ssm_scan_dynamic_seq_axis_resolution():
    batch = Axis("batch", 2)
    seq = Axis("seq", 4)
    d_model = Axis("d_model", 8)

    x = wrap(jnp.ones((2, 4, 8)), batch, seq, d_model)
    a = wrap(jnp.ones((2, 4, 8)), batch, seq, d_model)

    # seq_ax is None, it should resolve to 'seq'
    out = ssm_scan(x, a, seq_ax=None, mode="convex")
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, d_model)


def test_ssm_scan_stability_convex():
    # In convex mode, the outputs must be bounded by the input range [0.0, 1.0]
    # even with extremely large decay parameters 'a' (which would normally blow up in unstable systems)
    batch = Axis("batch", 1)
    seq = Axis("seq", 100)
    d_model = Axis("d_model", 1)

    x = wrap(jnp.ones((1, 100, 1)) * 0.8, batch, seq, d_model)
    a = wrap(jnp.ones((1, 100, 1)) * 5.0, batch, seq, d_model)  # Large positive a -> stable

    out = ssm_scan(x, a, seq_ax=seq, mode="convex")
    
    # State cannot exceed the max input value 0.8
    assert jnp.all(out.unwrap() <= 0.8)
    assert jnp.all(out.unwrap() >= 0.0)


# --- Unified nD Convolution Tests ---

def test_conv_1d():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    in_channels = Axis("in_channels", 16)
    out_channels = Axis("out_channels", 32)

    x = wrap(jnp.ones((2, 8, 16)) * 0.5, batch, seq, in_channels)

    # 1D Symmetric Padding
    out_sym = conv(x, in_channel_axes=in_channels, out_channel_axes=out_channels, spatial_axes=seq, kernel_sizes=3, padding="same")
    assert isinstance(out_sym, Tensor)
    assert out_sym.topology == (batch, seq, out_channels)
    assert out_sym.unwrap().shape == (2, 8, 32)

    # 1D Causal Padding
    out_causal = conv(x, in_channel_axes=in_channels, out_channel_axes=out_channels, spatial_axes=seq, kernel_sizes=3, padding="same", causal=True)
    assert isinstance(out_causal, Tensor)
    assert out_causal.topology == (batch, seq, out_channels)


def test_conv_2d():
    batch = Axis("batch", 2)
    h_ax = Axis("H", 8)
    w_ax = Axis("W", 8)
    in_channels = Axis("channels", 3)
    out_channels = Axis("out_channels", 16)

    x = wrap(jnp.ones((2, 8, 8, 3)), batch, h_ax, w_ax, in_channels)

    # 2D Same Padding
    out_2d = conv(x, in_channel_axes=in_channels, out_channel_axes=out_channels, spatial_axes=(h_ax, w_ax), kernel_sizes=(3, 3), padding="same")
    assert isinstance(out_2d, Tensor)
    assert out_2d.topology == (batch, h_ax, w_ax, out_channels)
    assert out_2d.unwrap().shape == (2, 8, 8, 16)


def test_conv_multi_channel():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    in_channels = Axis("in_channels", 16)
    out_heads = Axis("out_heads", 4)
    out_head_dim = Axis("out_head_dim", 8)

    x = wrap(jnp.ones((2, 8, 16)), batch, seq, in_channels)

    # Project directly to multi-channel/multi-head configuration without manual reshapes!
    out = conv(x, in_channel_axes=in_channels, out_channel_axes=(out_heads, out_head_dim), spatial_axes=seq, kernel_sizes=3, padding="same")
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, out_heads, out_head_dim)
    assert out.unwrap().shape == (2, 8, 4, 8)


# --- Tests for stateful Trainer class ---

def test_trainer_autoregressive():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    
    tokens = wrap(jnp.array([[0, 2, 4, 1, 9, 3, 5, 7], [1, 1, 3, 5, 2, 4, 6, 8]]), batch, seq)
    
    @axiom.ax.model
    def model(toks):
        vocab_ax = Axis("vocab", 10)
        embed_ax = Axis("embed", 16)
        x_emb = axiom.nn.embed(toks, vocab_ax, embed_ax)
        return getattr(x_emb, embed_ax.name).proj(vocab_ax)
    
    model.init(batch, seq)
    
    optimizer = optax.adam(1e-3)
    trainer = Trainer(model, optimizer, autoregressive=seq)
    
    loss = trainer.step(tokens)
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert trainer.model is not None
    assert trainer.opt_state is not None


def test_trainer_supervised():
    batch = Axis("batch", 4)
    d_model = Axis("d_model", 16)
    d_out = Axis("d_out", 8)
    
    x = wrap(jnp.ones((4, 16)), batch, d_model)
    y = wrap(jnp.ones((4, 8)), batch, d_out)
    
    @axiom.ax.model
    def model(inp):
        return getattr(inp, "d_model").proj(d_out)
    
    model.init(batch, d_model)
    
    optimizer = optax.adam(1e-3)
    trainer = Trainer(model, optimizer)
    
    loss = trainer.step(x, y)
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0
    assert trainer.model is not None
    assert trainer.opt_state is not None


def test_trainer_autoregressive_custom_loss():
    batch = Axis("batch", 2)
    seq = Axis("seq", 8)
    feature = Axis("feature", 10)
    data = wrap(jnp.ones((2, 8, 10)), batch, seq, feature)
    
    @axiom.ax.model
    def model(x):
        return getattr(x, "feature").proj(feature)
    
    model.init(batch, seq, feature)
    
    optimizer = optax.adam(1e-3)
    
    # We pass a custom loss function (e.g. mse_loss)
    from axiom.nn import mse_loss
    trainer = Trainer(model, optimizer, loss_fn=mse_loss, autoregressive=seq)
    
    loss = trainer.step(data)
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0


def test_trainer_supervised_custom_loss():
    batch = Axis("batch", 4)
    d_model = Axis("d_model", 16)
    d_out = Axis("d_out", 8)
    
    x = wrap(jnp.ones((4, 16)), batch, d_model)
    y = wrap(jnp.ones((4, 8)), batch, d_out)
    
    @axiom.ax.model
    def model(inp):
        return getattr(inp, "d_model").proj(d_out)
    
    model.init(batch, d_model)
    
    # A user-defined custom loss function
    def custom_l1_loss(preds, targets):
        return (preds - targets).abs().mean()
        
    optimizer = optax.adam(1e-3)
    trainer = Trainer(model, optimizer, loss_fn=custom_l1_loss)
    
    loss = trainer.step(x, y)
    assert isinstance(loss, Tensor)
    assert loss.topology == ()
    assert loss.unwrap() > 0


# --- Tests for StatefulTopologicalStream chat dataset support ---

def test_stateful_topological_stream_chat_template():
    # Define axes
    seq = Axis("seq", 8)
    batch = Axis("batch", 2)

    # Mock dataset with chat format
    mock_chat_data = [
        {"text": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]},
        {"text": [{"role": "user", "content": "how are you?"}, {"role": "assistant", "content": "doing well"}]},
        {"text": [{"role": "user", "content": "great"}, {"role": "assistant", "content": "thanks"}]}
    ]

    class MockIterator:
        def __init__(self, data):
            self.data = data
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx >= len(self.data):
                raise StopIteration
            res = self.data[self.idx]
            self.idx += 1
            return res

    class MockTokenizer:
        def apply_chat_template(self, chat, tokenize=True):
            # Return some fake token list based on chat content
            return [101, 102, 103, 104]

    # Patch load_dataset to return our MockIterator/iterable
    with patch("axionn.data.stream.load_dataset") as mock_load:
        class MockDataset:
            def shuffle(self, *args, **kwargs):
                return self
            def __iter__(self):
                return MockIterator(mock_chat_data)

        mock_load.return_value = MockDataset()

        tokenizer = MockTokenizer()
        stream = build_topological_stream(
            "dummy_path",
            seq_ax=seq,
            batch_ax=batch,
            tokenizer=tokenizer,
            text_key="text"
        )

        # Draw a batch from the stream
        batch_tensor = next(stream)
        assert isinstance(batch_tensor, Tensor)
        assert batch_tensor.topology == (batch, seq)
        assert batch_tensor.unwrap().shape == (2, 8)


def test_stateful_topological_stream_standard():
    # Define axes
    seq = Axis("seq", 8)
    batch = Axis("batch", 2)

    # Mock dataset with standard strings
    mock_data = [
        {"text": "hello"},
        {"text": "world"},
        {"text": "axiom"},
        {"text": "stream"},
    ]

    class MockIterator:
        def __init__(self, data):
            self.data = data
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx >= len(self.data):
                raise StopIteration
            res = self.data[self.idx]
            self.idx += 1
            return res

    class MockTokenizer:
        def encode(self, text):
            return [5, 6, 7, 8]

    # Patch load_dataset to return our MockIterator/iterable
    with patch("axionn.data.stream.load_dataset") as mock_load:
        class MockDataset:
            def shuffle(self, *args, **kwargs):
                return self
            def __iter__(self):
                return MockIterator(mock_data)

        mock_load.return_value = MockDataset()

        tokenizer = MockTokenizer()
        stream = build_topological_stream(
            "dummy_path",
            seq_ax=seq,
            batch_ax=batch,
            tokenizer=tokenizer,
            text_key="text"
        )

        # Draw a batch from the stream
        batch_tensor = next(stream)
        assert isinstance(batch_tensor, Tensor)
        assert batch_tensor.topology == (batch, seq)
        assert batch_tensor.unwrap().shape == (2, 8)


def test_swa_forward():
    from axionn.nn.attention import SWA
    batch = Axis("batch", 2)
    seq = Axis("seq", 16)
    d_model = Axis("d_model", 32)
    
    x = wrap(jnp.ones((2, 16, 32)), batch, seq, d_model)
    
    @axiom.ax.model
    def model(inp):
        return SWA(inp, num_heads=4, head_dim=8, window_size=4, causal=True)
    
    model.init(batch, seq, d_model)
    out = model(x)
    
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, d_model)
    assert out.unwrap().shape == (2, 16, 32)


def test_linear_attention_causal_forward():
    from axionn.nn.attention import LinearAttention
    batch = Axis("batch", 2)
    seq = Axis("seq", 16)
    d_model = Axis("d_model", 32)
    
    x = wrap(jnp.ones((2, 16, 32)), batch, seq, d_model)
    
    @axiom.ax.model
    def model(inp):
        return LinearAttention(inp, num_heads=4, head_dim=8, causal=True)
    
    model.init(batch, seq, d_model)
    out = model(x)
    
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, d_model)
    assert out.unwrap().shape == (2, 16, 32)


def test_linear_attention_non_causal_forward():
    from axionn.nn.attention import LinearAttention
    batch = Axis("batch", 2)
    seq = Axis("seq", 16)
    d_model = Axis("d_model", 32)
    
    x = wrap(jnp.ones((2, 16, 32)), batch, seq, d_model)
    
    @axiom.ax.model
    def model(inp):
        return LinearAttention(inp, num_heads=4, head_dim=8, causal=False)
    
    model.init(batch, seq, d_model)
    out = model(x)
    
    assert isinstance(out, Tensor)
    assert out.topology == (batch, seq, d_model)
    assert out.unwrap().shape == (2, 16, 32)


def test_local_run_logger():
    from axionn.training import LocalRunLogger
    import tempfile
    import os
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"learning_rate": 0.001, "batch_size": 32}
        logger = LocalRunLogger("test_run", log_dir=tmpdir, config=config, console_verbose=False)
        
        # Verify config was written
        assert os.path.exists(logger.config_file)
        with open(logger.config_file, "r") as f:
            saved_config = json.load(f)
        assert saved_config == config
        
        # Log some steps
        # Log step 1 with standard float
        logger.log(step=1, metrics={"loss": 2.5})
        
        # Log step 2 with a Tensor wrapped float
        val_tensor = wrap(jnp.array(1.8))
        logger.log(step=2, metrics={"loss": val_tensor}, tokens_processed=100)
        
        # Verify metrics log file was written
        assert os.path.exists(logger.log_file)
        with open(logger.log_file, "r") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        
        # Parse first line
        step_1_data = json.loads(lines[0])
        assert step_1_data["step"] == 1
        assert step_1_data["loss"] == 2.5
        
        # Parse second line
        step_2_data = json.loads(lines[1])
        assert step_2_data["step"] == 2
        assert abs(step_2_data["loss"] - 1.8) < 1e-5
        assert "tokens_per_sec" in step_2_data


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__]))
