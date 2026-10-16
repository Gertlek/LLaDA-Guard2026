from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llada_guard.objectives import training_loss


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(12, 12))
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append(input_ids.clone())
        context = F.one_hot(input_ids, 12).float().sum(1)
        logits = (context @ self.weight)[:, None, :].expand(-1, input_ids.shape[1], -1)
        return SimpleNamespace(logits=logits)


def make_batch():
    batch = {
        "has_response": torch.tensor([0.0, 1.0]),
        "prompt_is_unsafe": torch.tensor([0.0, 0.0]),
        "response_is_unsafe": torch.tensor([1.0, 1.0]),
        "prompt_label_mask": torch.ones(2),
        "response_label_mask": torch.ones(2),
    }
    for branch, verdict in (("safe", 7), ("unsafe", 8)):
        batch[f"gen_{branch}_input_ids"] = torch.tensor(
            [[1, 2, 3, 4, verdict], [1, 2, 3, 4, verdict]]
        )
        batch[f"gen_{branch}_block_ids"] = torch.tensor(
            [[0, 1, 1, 2, 0], [0, 1, 1, 2, 0]]
        )
        batch[f"gen_{branch}_real_lengths"] = torch.tensor([5, 5])
    return batch


@pytest.mark.parametrize("objective", ["ranking", "generative"])
def test_loss_formula_masks_and_gradients(objective):
    torch.manual_seed(42)
    model = ToyModel()
    batch = make_batch()
    loss, _ = training_loss(model, batch, objective=objective, pad_id=0, mask_id=11)
    noisy = model.calls[0]
    assert torch.equal(noisy[:2, :4], noisy[2:, :4])
    assert noisy[0, 3] == 11  # Hide the synthetic response.
    assert torch.equal(noisy[1, :3], batch["gen_safe_input_ids"][1, :3])
    scores, true_ce = [], []
    logits = model(noisy, torch.ones_like(noisy)).logits
    for row in range(2):
        positions = (noisy[row] == 11) & (
            batch["gen_safe_block_ids"][row] == (1 if row == 0 else 2)
        )
        ces = [
            F.cross_entropy(
                logits[row + offset, positions],
                batch[f"gen_{branch}_input_ids"][row, positions],
            )
            for branch, offset in (("safe", 0), ("unsafe", 2))
        ]
        scores.append(ces[0] - ces[1])
        true_ce.append(ces[row])
    expected = (
        2
        * F.binary_cross_entropy_with_logits(
            torch.stack(scores), torch.tensor([0.0, 1.0])
        )
        if objective == "ranking"
        else torch.stack(true_ce).mean()
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.abs().sum() > 0


def test_reject_missing_scoped_label():
    batch = make_batch()
    batch["response_label_mask"][1] = 0
    with pytest.raises(ValueError, match="label"):
        training_loss(ToyModel(), batch, objective="ranking", pad_id=0)


def test_reject_unknown_objective():
    with pytest.raises(ValueError, match="Unknown objective"):
        training_loss(ToyModel(), make_batch(), objective="typo", pad_id=0)
