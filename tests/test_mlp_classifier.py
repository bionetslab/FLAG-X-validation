
import pytest
import torch
import numpy as np
from sklearn.exceptions import NotFittedError

from flagx.gating import MLPClassifier



def test_fit_sets_attributes(mlp_classifier, small_X, small_y):
    mlp_classifier.fit(small_X, small_y)

    assert mlp_classifier.is_fitted_ is True
    assert hasattr(mlp_classifier, 'model_')
    assert hasattr(mlp_classifier, 'classes_')
    assert mlp_classifier.classes_.shape[0] == 2
    assert len(mlp_classifier.losses_) == mlp_classifier.n_epochs
    assert len(mlp_classifier.n_corrects_) == mlp_classifier.n_epochs
    assert mlp_classifier.device.type in {'cpu', 'cuda'}


def test_predict(mlp_classifier, small_X, small_y):
    mlp_classifier.fit(small_X, small_y)

    preds = mlp_classifier.predict(small_X)

    assert preds.shape == (small_X.shape[0],)
    assert set(preds).issubset(set(small_y))


def test_predict_and_predict_proba(mlp_classifier, small_X, small_y):
    mlp_classifier.fit(small_X, small_y)

    probas = mlp_classifier.predict_proba(small_X)

    assert probas.shape == (small_X.shape[0], 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert np.all((probas >= 0) & (probas <= 1))


def test_predict_before_fit_raises(mlp_classifier, small_X):
    with pytest.raises(NotFittedError):
        mlp_classifier.predict(small_X)

    with pytest.raises(NotFittedError):
        mlp_classifier.predict_proba(small_X)


def test_score_returns_valid_f1(mlp_classifier, small_X, small_y):
    mlp_classifier.fit(small_X, small_y)
    score = mlp_classifier.score(small_X, small_y)
    assert 0.0 <= score <= 1.0


def test_score_with_sample_weights(mlp_classifier, small_X, small_y):
    mlp_classifier.fit(small_X, small_y)
    weights = np.ones(len(small_y))
    score = mlp_classifier.score(small_X, small_y, sample_weight=weights)
    assert 0.0 <= score <= 1.0


def test_fit_warns_when_no_data_loader_params(mlp_classifier, small_X, small_y):
    with pytest.warns(UserWarning, match="No data_loader_params provided"):
        mlp_classifier.fit(small_X, small_y)


def test_custom_data_loader_params_used(small_X, small_y):
    custom_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 1}
    clf = MLPClassifier(n_epochs=2, data_loader_params=custom_params)
    clf.fit(small_X, small_y)
    assert clf.data_loader_.batch_size == 16
    assert clf.data_loader_.num_workers == 1


def test_class_label_remapping_non_consecutive(mlp_classifier, small_X):
    y = np.random.choice([0, 5, 10], size=100)
    mlp_classifier.fit(small_X, y)

    assert len(mlp_classifier.classes_) == 3
    assert set(mlp_classifier.og_classes_) == {0, 5, 10}
    assert mlp_classifier.new_to_og_classes_dict_ == {0: 0, 1: 5, 2: 10, -1: -1}

    # Predict should return original labels
    preds = mlp_classifier.predict(small_X[:5])
    assert set(preds).issubset({0, 5, 10})


def test_training_loop_records_metrics(mlp_classifier, small_X, small_y):
    mlp_classifier.n_epochs = 5
    mlp_classifier.fit(small_X, small_y)

    assert len(mlp_classifier.losses_) == 5
    assert all(isinstance(l, float) for l in mlp_classifier.losses_)
    assert all(l >= 0.0 for l in mlp_classifier.losses_)
    assert all(0 <= c <= small_y.shape[0] for c in mlp_classifier.n_corrects_)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_device_uses_gpu_when_available(small_X, small_y):
    clf = MLPClassifier(n_epochs=2, device=None)  # auto-detect
    clf.fit(small_X, small_y)
    assert clf.device.type == 'cuda'


def test_device_forced_cpu(small_X, small_y):
    clf = MLPClassifier(n_epochs=2, device='cpu')
    clf.fit(small_X, small_y)
    assert clf.device.type == 'cpu'


def test_save_and_load_roundtrip(mlp_classifier, small_X, small_y, tmp_path):
    mlp_classifier.fit(small_X, small_y)

    path = tmp_path / 'mlp_test.pkl'
    mlp_classifier.save(filename=path.name, filepath=str(tmp_path))

    loaded = MLPClassifier.load(filename=path.name, filepath=str(tmp_path))

    assert isinstance(loaded, MLPClassifier)
    assert loaded.is_fitted_ is True
    assert loaded.classes_.tolist() == mlp_classifier.classes_.tolist()
    assert loaded.n_epochs == mlp_classifier.n_epochs

    # Predictions should be very similar (same weights)
    pred_original = mlp_classifier.predict(small_X[:10])
    pred_loaded = loaded.predict(small_X[:10])
    assert np.allclose(pred_original, pred_loaded)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_load_with_map_location_from_gpu_to_cpu(mlp_classifier, small_X, small_y, tmp_path):

    # Trains on GPU by default
    mlp_classifier.fit(small_X, small_y)

    # Save model
    path = tmp_path / 'mlp_gpu.pkl'
    mlp_classifier.save(filename=path.name, filepath=str(tmp_path))

    # Load to CPU explicitly
    loaded = MLPClassifier.load(
        filename=path.name,
        filepath=str(tmp_path),
        map_location='cpu'
    )

    for param in loaded.model_.parameters():
        assert param.device.type == 'cpu'


def test_get_num_correct():
    # true labels
    y_true = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # pretend these are model logits for 3 classes (shape: [10, 3])
    # predictions will be argmax along dim=1 → [1, 1, 1, 2, 0, 1, 1, 1, 2, 2]
    y_pred_logits = torch.tensor([
        [0.1, 0.9, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 3.0, 0.1],
        [0.0, 0.1, 5.0],
        [2.0, 0.1, 0.0],
        [0.1, 0.9, 0.2],
        [0.1, 0.9, 0.3],
        [0.0, 1.0, 0.8],
        [0.0, 0.1, 2.5],
        [0.0, 0.2, 3.1],
    ])
    num_correct = MLPClassifier._get_num_correct(y_pred_logits, y_true)
    assert num_correct == 3


def test_process_class_labels():
    y = np.array([10, 20, 10, 30, 20])
    y_new, new_classes, counts, priors, new_to_og, og_classes = MLPClassifier._process_class_labels(y)

    assert np.array_equal(y_new, [0, 1, 0, 2, 1])
    assert np.array_equal(new_classes, [0, 1, 2])
    assert np.array_equal(counts, [2, 2, 1])
    assert new_to_og == {0: 10, 1: 20, 2: 30, -1: -1}
    assert np.array_equal(og_classes, [10, 20, 30])


def test_single_class_training(mlp_classifier, small_X):
    y = np.zeros(100, dtype=int)
    mlp_classifier.fit(small_X, y)

    assert mlp_classifier.classes_.shape[0] == 1
    preds = mlp_classifier.predict(small_X)
    assert np.all(preds == 0)
    probas = mlp_classifier.predict_proba(small_X)
    assert np.allclose(probas, 1.0)  # 100% confidence in the only class



