import pytest
import torch
import torch.nn.functional as F

from ndae.models.time_embedding import SinusoidalTimeEmbedding, TimeMLP


def test_sinusoidal_time_embedding_scalar_shape_and_bounds() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    output = embedding(torch.tensor(0.5))

    assert output.shape == (64,)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_sinusoidal_time_embedding_batch_shape_and_bounds() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    output = embedding(torch.tensor([0.0, 0.5, 1.0]))

    assert output.shape == (3, 64)
    assert torch.all(output >= -1.0)
    assert torch.all(output <= 1.0)


def test_sinusoidal_time_embedding_distinguishes_different_times() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    first = embedding(torch.tensor(0.5))
    second = embedding(torch.tensor(1.0))
    similarity = F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item()

    assert not torch.allclose(first, second)
    assert similarity < 1.0


def test_sinusoidal_time_embedding_is_smooth_for_nearby_times() -> None:
    embedding = SinusoidalTimeEmbedding(64)

    first = embedding(torch.tensor(0.5))
    second = embedding(torch.tensor(0.501))
    similarity = F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item()

    assert similarity > 0.999


def test_time_mlp_batch_output_shape() -> None:
    mlp = TimeMLP(32)

    output = mlp(torch.tensor([0.0, 0.5, 1.0]))

    assert output.shape == (3, 64)


def test_time_mlp_gradient_flows_to_input_time() -> None:
    mlp = TimeMLP(32)
    t = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)

    mlp(t).sum().backward()

    assert t.grad is not None
    assert not torch.isnan(t.grad).any()


@pytest.mark.parametrize("dim", [3, 5])
def test_time_embedding_rejects_invalid_dim(dim: int) -> None:
    with pytest.raises(ValueError, match="even dim >= 4"):
        SinusoidalTimeEmbedding(dim)


def test_time_embedding_rejects_dim_smaller_than_four() -> None:
    with pytest.raises(ValueError, match="even dim >= 4"):
        SinusoidalTimeEmbedding(2)


@pytest.mark.parametrize("module_cls", [SinusoidalTimeEmbedding, TimeMLP])
def test_time_embedding_rejects_invalid_input_rank(
    module_cls: type[SinusoidalTimeEmbedding] | type[TimeMLP],
) -> None:
    module = module_cls(32)

    with pytest.raises(ValueError, match=r"expects time shaped \[\] or \[B\]"):
        module(torch.zeros(2, 1))
