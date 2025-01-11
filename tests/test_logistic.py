import pytest
import torch

from src.wrangle.logistic import ScaledLogistic


def synthetic_data(
    scale_true: float, coef_true: float, b_true: float, num_samples: int = 10000
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    X = torch.randn(num_samples, 1, dtype=torch.float64) * 5

    coef_true_tensor = torch.tensor([coef_true], dtype=torch.float64)
    scale_true_tensor = torch.tensor(scale_true, dtype=torch.float64)
    b_true_tensor = torch.tensor(b_true, dtype=torch.float64)

    logits = X @ coef_true_tensor + b_true_tensor
    p = torch.sigmoid(logits)
    p_scaled = scale_true_tensor * p
    # Bernoulli draws
    y = torch.bernoulli(p_scaled)  # shape (num_samples,)
    return X, y


def test_logistic_scaled() -> None:
    # Generate data
    X, y = synthetic_data(0.5, -3.0, 0.0)
    model = ScaledLogistic(1)
    params = model.train(X, y, sample_weight=torch.ones_like(y), num_epochs=2000)

    # Remove print statement or use pytest.logging
    # print(params)

    # Use pytest's built-in assert or approx
    assert pytest.approx(params.scale, abs=0.1) == 0.5
    assert pytest.approx(params.coef, abs=0.1) == -3.0
    assert pytest.approx(params.intercept, abs=0.1) == 0.0
