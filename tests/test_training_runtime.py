import random

import pytest
import torch

from llada_guard.training_runtime import (
    RouteMixedBatchSampler,
    build_scheduler,
    shard_examples,
)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "flags", [[False] * 10, [True] * 7, [False, True] * 9, [False] * 3 + [True] * 12]
)
def test_batch_coverage_and_repeatability(batch_size, flags):
    sampler = RouteMixedBatchSampler(flags, batch_size)
    random.seed(42)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(flags)))
    assert len(batches) == len(sampler)
    assert all(0 < len(batch) <= batch_size for batch in batches)
    random.seed(42)
    assert list(sampler) == batches


def test_balanced_route_batches():
    flags = [False, True] * 8
    for batch in RouteMixedBatchSampler(flags, 4):
        assert sum(flags[i] for i in batch) == 2


def test_shards_are_contiguous_after_seeded_shuffle():
    original = list(range(11))
    expected = original.copy()
    random.Random(42).shuffle(expected)
    shards = [shard_examples(original, 42, rank, 2) for rank in range(2)]
    assert shards[0] + shards[1] == expected + expected[:1]
    assert original == list(range(11))
    assert [shard_examples([9], 42, rank, 3) for rank in range(3)] == [[9]] * 3


def test_historical_warmup_and_decay():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(optimizer, 2, 4)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates == pytest.approx([0.01, 0.505, 1.0, 0.5, 0.0])
