# src/axionn/alignment/stateless.py

import jax
import jax.numpy as jnp
from typing import Any, Tuple
from axiom.core import Tensor, Axis
from axiom.compiler import value_and_grad, apply_updates, AxiomModel
from axiom.nn import cross_entropy_loss


def _get_seq_logprobs(model: AxiomModel, x: Tensor, seq_ax: Axis) -> Tensor:
    """
    Helper function to calculate the sequence-level log probabilities.
    Slices for next-token prediction and sums over the sequence axis.
    """
    logits = model(x)

    # Sliced logits: 0 to S-2 on the sequence axis
    logits_sliced = getattr(logits, seq_ax.name)[:-1].chunk_tensor

    # Sliced targets: 1 to S-1 on the sequence axis
    targets_sliced = getattr(x, seq_ax.name)[1:].chunk_tensor
    vocab_ax = logits_sliced.topology[-1]

    # Axiom's cross_entropy_loss natively returns the negative log probabilities
    # We use reduction='none' to get the token-level values, then sum across the sequence.
    neg_log_probs = cross_entropy_loss(
        getattr(logits_sliced, vocab_ax.name),
        targets_sliced,
        reduction='none'
    )

    # Return the positive log probabilities summed over the sequence
    return -getattr(neg_log_probs, seq_ax.name).sum()


def dpo_step(
        policy_model: Any,
        ref_model: Any,
        optim: Any,
        chosen_x: Tensor,
        rejected_x: Tensor,
        seq_ax: Axis,
        beta: float = 0.1
) -> Tuple[Any, Any, Tensor]:
    """
    Executes a single Direct Preference Optimization (DPO) step.

    Formula: -log sigmoid( beta * ( (log_pi_y_w - log_ref_y_w) - (log_pi_y_l - log_ref_y_l) ) )
    """
    # 1. Compute reference log-probs completely outside the gradient tape
    # (Since ref_model is strictly an AxiomModel, it won't track gradients here)
    ref_chosen_logps = _get_seq_logprobs(ref_model, chosen_x, seq_ax)
    ref_rejected_logps = _get_seq_logprobs(ref_model, rejected_x, seq_ax)

    @value_and_grad(has_aux=False)
    def loss_fn(m: AxiomModel, c_x: Tensor, r_x: Tensor):
        # 2. Compute policy log-probs inside the gradient tape
        pi_chosen_logps = _get_seq_logprobs(m, c_x, seq_ax)
        pi_rejected_logps = _get_seq_logprobs(m, r_x, seq_ax)

        # 3. Calculate implicit rewards
        pi_logratios = pi_chosen_logps - pi_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        # 4. Standard DPO log-sigmoid loss
        # We unwrap() to use raw JAX primitives for the final scalar reduction
        loss_val = -jnp.mean(jax.nn.log_sigmoid(beta * logits.unwrap()))
        return Tensor(loss_val)

    loss, grads = loss_fn(policy_model, chosen_x, rejected_x)

    # 5. Apply Updates
    if isinstance(optim, tuple) and len(optim) == 2:
        optimizer, opt_state = optim
        updated_model, new_opt_state = apply_updates(policy_model, grads, optimizer, opt_state)
        updated_optim = (optimizer, new_opt_state)
    else:
        lr = optim if isinstance(optim, (int, float)) else 1e-6
        updated_model = policy_model - (grads * lr).params
        updated_optim = optim

    return updated_model, updated_optim, loss


def kto_step(
        policy_model: Any,
        ref_model: Any,
        optim: Any,
        x: Tensor,
        is_desirable: Tensor,
        seq_ax: Axis,
        beta: float = 0.1
) -> Tuple[Any, Any, Tensor]:
    """
    Executes a Kahneman-Tversky Optimization (KTO) step.
    Unlike DPO, this does not require paired preferences, just a binary label
    per sequence (is_desirable = 1.0 for chosen, 0.0 for rejected).
    """
    ref_logps = _get_seq_logprobs(ref_model, x, seq_ax)

    @value_and_grad(has_aux=False)
    def loss_fn(m: AxiomModel, inp: Tensor, desirable_flags: Tensor):
        pi_logps = _get_seq_logprobs(m, inp, seq_ax)

        # Calculate the implicit KL reward estimate
        # r(x, y) = beta * (log_pi - log_ref)
        reward_estimate = beta * (pi_logps.unwrap() - ref_logps.unwrap())

        # In a full batch KTO implementation, we estimate the baseline KL divergence
        # Here we approximate it by the batch mean of the implicit reward
        kl_baseline = jnp.mean(reward_estimate)

        # KTO conditional loss routing
        # If desirable (+1): 1 - sigmoid(reward - kl_baseline)
        # If undesirable (0): 1 - sigmoid(kl_baseline - reward)

        flags = desirable_flags.unwrap()

        chosen_loss = 1.0 - jax.nn.sigmoid(reward_estimate - kl_baseline)
        rejected_loss = 1.0 - jax.nn.sigmoid(kl_baseline - reward_estimate)

        # Route the loss based on the boolean flags
        routed_losses = jnp.where(flags > 0.5, chosen_loss, rejected_loss)

        return Tensor(jnp.mean(routed_losses))

    loss, grads = loss_fn(policy_model, x, is_desirable)

    # Apply Updates
    if isinstance(optim, tuple) and len(optim) == 2:
        optimizer, opt_state = optim
        updated_model, new_opt_state = apply_updates(policy_model, grads, optimizer, opt_state)
        updated_optim = (optimizer, new_opt_state)
    else:
        lr = optim if isinstance(optim, (int, float)) else 1e-6
        updated_model = policy_model - (grads * lr).params
        updated_optim = optim

    return updated_model, updated_optim, loss
