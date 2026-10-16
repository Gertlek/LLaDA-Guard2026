"""Batching and optimizer schedule for the recovered training procedure."""

import math
import random
from collections import deque

from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import Sampler


class RouteMixedBatchSampler(Sampler):
    def __init__(self, has_response, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.prompt = [i for i, flag in enumerate(has_response) if not flag]
        self.response = [i for i, flag in enumerate(has_response) if flag]

    def __len__(self):
        return math.ceil((len(self.prompt) + len(self.response)) / self.batch_size)

    def __iter__(self):
        prompt, response = deque(self.prompt), deque(self.response)
        random.shuffle(prompt)
        random.shuffle(response)
        if self.batch_size == 1:
            take_response = bool(random.getrandbits(1))
            while prompt or response:
                if prompt and response:
                    yield [(response if take_response else prompt).popleft()]
                    take_response = not take_response
                else:
                    yield [(prompt if prompt else response).popleft()]
            return
        while prompt or response:
            if prompt and response:
                prompt_target = self.batch_size // 2
            else:
                prompt_target = self.batch_size if prompt else 0
            response_target = self.batch_size - prompt_target
            batch = [prompt.popleft() for _ in range(min(prompt_target, len(prompt)))]
            batch.extend(
                response.popleft() for _ in range(min(response_target, len(response)))
            )
            while len(batch) < self.batch_size and (prompt or response):
                queue = prompt if len(prompt) >= len(response) and prompt else response
                batch.append(queue.popleft())
            random.shuffle(batch)
            yield batch


def shard_examples(examples, seed, rank, world_size):
    if not examples or not 0 <= rank < world_size:
        raise ValueError("Nonempty examples and a valid distributed rank are required")
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    shard_size = math.ceil(len(examples) / world_size)
    padded_size = shard_size * world_size
    original_size = len(examples)
    examples.extend(
        examples[i % original_size] for i in range(padded_size - original_size)
    )
    return examples[rank * shard_size : (rank + 1) * shard_size]


def build_scheduler(optimizer, warmup_steps, total_steps):
    if warmup_steps < 0 or total_steps < 1:
        raise ValueError("Invalid warmup or training duration")
    decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=max(total_steps - warmup_steps, 1),
    )
    if not warmup_steps:
        return decay
    warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )
