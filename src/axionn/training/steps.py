# src/axionn/training/steps.py

from typing import Any, Tuple, Optional
from axiom.core import Tensor, Axis
from axiom.compiler import value_and_grad, apply_updates, AxiomModel
from axiom.nn import cross_entropy_loss, mse_loss


def autoregressive_ce_step(
        model: Any,
        optim: Any,
        x: Tensor,
        axis: Axis,
        loss_fn_arg: Optional[Any] = None,
) -> Tuple[Any, Any, Tensor]:
    """
    Executes a single autoregressive cross-entropy training step.
    """
    @value_and_grad(has_aux=True)
    def loss_fn(m: AxiomModel, inp: Tensor, seq_axis: Axis):
        logits: Tensor = m(inp)
        logits_sliced = getattr(logits, seq_axis.name)[:-1].chunk_tensor
        targets_sliced = getattr(inp, seq_axis.name)[1:].chunk_tensor
        vocab_ax = logits_sliced.topology[-1]
        if loss_fn_arg is None or loss_fn_arg is cross_entropy_loss:
            loss_val = cross_entropy_loss(getattr(logits_sliced, vocab_ax.name), targets_sliced)
        else:
            loss_val = loss_fn_arg(logits_sliced, targets_sliced)
        return loss_val.mean(), logits

    (loss, logits), grads = loss_fn(model, x, axis)

    if isinstance(optim, tuple) and len(optim) == 2:
        optimizer, opt_state = optim
        updated_model, new_opt_state = apply_updates(model, grads, optimizer, opt_state)
        updated_optim = (optimizer, new_opt_state)
    else:
        lr = optim if isinstance(optim, (int, float)) else 0.001
        updated_model = model - (grads * lr).params
        updated_optim = optim

    return updated_model, updated_optim, loss


def mse_step(
        model: Any,
        optim: Any,
        x: Tensor,
        y: Tensor
) -> Tuple[Any, Any, Tensor]:
    """
    Executes a standard Mean Squared Error training step.
    """
    @value_and_grad(has_aux=False)
    def loss_fn(m: AxiomModel, inp: Tensor, targets: Tensor):
        predictions: Tensor = m(inp)
        return mse_loss(predictions, targets)

    loss, grads = loss_fn(model, x, y)

    if isinstance(optim, tuple) and len(optim) == 2:
        optimizer, opt_state = optim
        updated_model, new_opt_state = apply_updates(model, grads, optimizer, opt_state)
        updated_optim = (optimizer, new_opt_state)
    else:
        lr = optim if isinstance(optim, (int, float)) else 0.001
        updated_model = model - (grads * lr).params
        updated_optim = optim

    return updated_model, updated_optim, loss


def general_step(
        model: Any,
        optim: Any,
        x: Tensor,
        y: Tensor,
        loss_fn_arg: Any
) -> Tuple[Any, Any, Tensor]:
    """
    Executes a training step using an arbitrary loss function.
    """
    @value_and_grad(has_aux=False)
    def loss_fn(m: AxiomModel, inp: Tensor, targets: Tensor):
        predictions: Tensor = m(inp)
        return loss_fn_arg(predictions, targets)

    loss, grads = loss_fn(model, x, y)

    if isinstance(optim, tuple) and len(optim) == 2:
        optimizer, opt_state = optim
        updated_model, new_opt_state = apply_updates(model, grads, optimizer, opt_state)
        updated_optim = (optimizer, new_opt_state)
    else:
        lr = optim if isinstance(optim, (int, float)) else 0.001
        updated_model = model - (grads * lr).params
        updated_optim = optim

    return updated_model, updated_optim, loss


def build_trainer(
        model: Any,
        optimizer: Any,
        loss_fn: Optional[Any] = None,
        autoregressive: Optional[Axis] = None,
):
    """
    Unified training factory returning a compiled generator interface (.send(x)).
    """
    from axiom import ax

    opt_state = optimizer.init(model)

    if autoregressive is not None:
        # Autoregressive sequence training
        step_loss_fn = loss_fn if loss_fn is not None else cross_entropy_loss

        @ax.jit
        def train_step(m, state, data):
            m_up, (_, s_up), loss_val = autoregressive_ce_step(m, (optimizer, state), data, autoregressive, step_loss_fn)
            return m_up, s_up, loss_val

        def trainer_generator():
            data = yield
            while True:
                nonlocal model, opt_state
                model, opt_state, loss = train_step(model, opt_state, data)
                data = yield loss
    else:
        # Supervised target-based training (e.g. MSE or general loss)
        step_loss_fn = loss_fn if loss_fn is not None else mse_loss

        @ax.jit
        def train_step(m, state, inp, targets):
            m_up, (_, s_up), loss_val = general_step(m, (optimizer, state), inp, targets, step_loss_fn)
            return m_up, s_up, loss_val

        def trainer_generator():
            pair = yield
            while True:
                nonlocal model, opt_state
                inp, targets = pair
                model, opt_state, loss = train_step(model, opt_state, inp, targets)
                pair = yield loss

    gen = trainer_generator()
    next(gen)  # Prime the generator
    return gen
