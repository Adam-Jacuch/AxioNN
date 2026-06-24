# src/axionn/data/stream.py

import logging
import numpy as np
from datasets import load_dataset
from typing import Optional, Any
from axiom.core import Tensor, Axis

logger = logging.getLogger(__name__)


class StatefulTopologicalStream:
    """
    A stateful, preemption-ready dataloader stream that supports on-the-fly tokenization,
    infinite sequence packing, and serialization of its cursor position and buffers.
    """
    def __init__(
        self,
        dataset_path: str,
        seq_ax: Axis,
        batch_ax: Axis,
        split: str = "train",
        config_name: Optional[str] = None,
        token_key: str = "input_ids",
        buffer_size: int = 10000,
        seed: int = 42,
        tokenizer: Optional[Any] = None,
        text_key: str = "text"
    ):
        self.seq_ax = seq_ax
        self.batch_ax = batch_ax
        self.seq_len = seq_ax.size
        self.batch_size = batch_ax.size
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.token_key = token_key

        # Load in streaming mode to keep RAM usage near zero
        self.dataset = load_dataset(dataset_path, name=config_name, split=split, streaming=True)
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)

        self.iterator = iter(self.dataset)
        self.token_buffer = []
        self.batch_buffer = []

    def __iter__(self):
        return self

    def __next__(self) -> Tensor:
        while True:
            try:
                example = next(self.iterator)
            except StopIteration:
                # Re-initialize the dataset stream (epoch loop) for infinite cycling
                self.iterator = iter(self.dataset)
                example = next(self.iterator)

            if self.tokenizer is not None:
                # Tokenize raw text on-the-fly
                text = example[self.text_key]
                if hasattr(self.tokenizer, "encode") and callable(self.tokenizer.encode):
                    tokens = self.tokenizer.encode(text)
                elif callable(self.tokenizer):
                    res = self.tokenizer(text)
                    tokens = res["input_ids"] if isinstance(res, dict) else res
                else:
                    raise ValueError("Tokenizer must be callable or have an '.encode()' method.")
            else:
                # Fallback to pre-tokenized array
                tokens = example[self.token_key]

            self.token_buffer.extend(tokens)

            # Pack sequences tightly
            while len(self.token_buffer) >= self.seq_len:
                seq = self.token_buffer[:self.seq_len]
                self.token_buffer = self.token_buffer[self.seq_len:]
                self.batch_buffer.append(seq)

                # Yield strictly shaped batches
                if len(self.batch_buffer) >= self.batch_size:
                    raw_batch = np.array(self.batch_buffer[:self.batch_size], dtype=np.int32)
                    self.batch_buffer = self.batch_buffer[self.batch_size:]
                    return Tensor(raw_batch, self.batch_ax, self.seq_ax)

    def state_dict(self) -> dict:
        """Saves current state of the dataloader for preemption checkpoints."""
        hf_state = {}
        if hasattr(self.iterator, "state_dict") and callable(self.iterator.state_dict):
            try:
                hf_state = self.iterator.state_dict()
            except Exception as e:
                logger.warning(f"Failed to serialize dataset iterator state_dict: {e}")
        else:
            logger.warning("Dataset iterator does not support state_dict. Re-loading may start from the beginning.")
        return {
            "dataset_state": hf_state,
            "token_buffer": list(self.token_buffer),
            "batch_buffer": list(self.batch_buffer),
        }

    def load_state_dict(self, state_dict: dict):
        """Restores the dataloader's exact cursor position and buffers."""
        if hasattr(self.iterator, "load_state_dict") and callable(self.iterator.load_state_dict):
            try:
                self.iterator.load_state_dict(state_dict["dataset_state"])
            except Exception as e:
                logger.warning(f"Failed to load dataset iterator state_dict: {e}")
        else:
            logger.warning("Dataset iterator does not support load_state_dict. Resumption may start from the beginning.")
        self.token_buffer = list(state_dict["token_buffer"])
        self.batch_buffer = list(state_dict["batch_buffer"])


def build_topological_stream(
        dataset_path: str,
        seq_ax: Axis,
        batch_ax: Axis,
        split: str = "train",
        config_name: Optional[str] = None,
        token_key: str = "input_ids",
        buffer_size: int = 10000,
        seed: int = 42,
        tokenizer: Optional[Any] = None,
        text_key: str = "text"
) -> StatefulTopologicalStream:
    """
    Streams a Hugging Face dataset directly into topologically safe Axiom Tensors.
    Supports both pre-tokenized inputs and on-the-fly tokenization of raw text.
    Infinite packing ensures zero padding waste.
    """
    return StatefulTopologicalStream(
        dataset_path=dataset_path,
        seq_ax=seq_ax,
        batch_ax=batch_ax,
        split=split,
        config_name=config_name,
        token_key=token_key,
        buffer_size=buffer_size,
        seed=seed,
        tokenizer=tokenizer,
        text_key=text_key
    )
