
import numpy as np
import pytest
from flagx.gating import SomClassifier, MLPClassifier


@pytest.fixture
def small_X():
    np.random.seed(42)
    return np.random.rand(100, 4)

@pytest.fixture
def small_y():
    np.random.seed(42)
    return np.random.choice([0, 1], size=100)

@pytest.fixture
def som_classifier():
    return SomClassifier(
        som_dimensions=(3, 3),
        n_epochs=6,
        verbosity=0,
    )

@pytest.fixture
def large_X():
    np.random.seed(42)
    return np.random.rand(110000, 4)

@pytest.fixture
def large_y():
    np.random.seed(42)
    return np.random.choice([0, 1], size=110000)

@pytest.fixture
def large_som_classifier():
    return SomClassifier(
        som_dimensions=(11, 11),
        n_epochs=6,
        verbosity=0,
    )

@pytest.fixture
def mlp_classifier():
    return MLPClassifier(
        layer_sizes=(16, 8, 4),
        n_epochs=6,
        verbosity=0,
    )
