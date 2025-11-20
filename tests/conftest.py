
import numpy as np
import pytest
from pathlib import Path
from flagx.gating import SomClassifier, MLPClassifier
from flagx.io import FlowDataManager

TEST_DATA_DIR = Path(__file__).parent / 'test_data'
N_FILES = 5

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
        som_dimensions=(2, 2),
        n_epochs=2,
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
        n_epochs=2,
        verbosity=0,
    )

@pytest.fixture
def mlp_classifier():
    return MLPClassifier(
        layer_sizes=(8, 4, 2),
        n_epochs=2,
        verbosity=0,
    )

@pytest.fixture(scope='session', params=['csv', 'fcs'])
def data_format(request):
    return request.param

@pytest.fixture(scope='function')
def test_files(data_format):
    """Collect only files of the correct format."""
    if data_format == 'csv':
        files = sorted(TEST_DATA_DIR.glob('*.csv'))
    else:
        files = sorted(TEST_DATA_DIR.glob('*.fcs'))

    assert len(files) == N_FILES, f'Expected {N_FILES} {data_format} files.'

    return [f.name for f in files]


@pytest.fixture(scope='function')
def fdm(test_files, tmp_path_factory):
    """Separate FlowDataManager instance for each format."""
    save_dir = tmp_path_factory.mktemp('fdm_output')

    manager = FlowDataManager(
        data_file_names=test_files,
        data_file_path=str(TEST_DATA_DIR),
        save_path=str(save_dir),
        verbosity=0,
    )
    return manager


