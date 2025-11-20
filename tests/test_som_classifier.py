
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError


def test_fit_sets_is_fitted(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)

    assert som_classifier.is_fitted_
    assert som_classifier.n_features_in_ == small_X.shape[1]
    assert som_classifier.som_.codebook.shape == (2, 2, small_X.shape[1])


def test_fit_dimension_mismatch(som_classifier, small_y):
    X_wrong = np.random.rand(10, 4)
    with pytest.raises(ValueError):
        som_classifier.fit(X_wrong, small_y)


def test_annotate_creates_labels(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    assert som_classifier._labeled_data
    assert som_classifier.som_unit_labels_.shape == (2, 2)
    assert set(som_classifier.som_unit_labels_.flatten()).issubset({0, 1})


def test_predict(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    preds = som_classifier.predict(small_X)
    assert preds.shape[0] == small_X.shape[0]
    assert set(preds).issubset({0, 1, -1})


def test_predict_proba(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    proba = som_classifier.predict_proba(small_X)

    assert proba.shape[0] == small_X.shape[0]
    assert proba.shape[1] == len(som_classifier.classes_)
    assert proba.sum(axis=1) == pytest.approx(np.ones(proba.shape[0]))


def test_reset_clears_state(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    som_classifier.reset()

    assert not hasattr(som_classifier, 'is_fitted_')
    assert som_classifier._is_fitted is False
    assert som_classifier.som_ is None
    assert som_classifier._labeled_data is False


def test_transform_shapes(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)

    bmus, scat, unit_ids, radii = som_classifier.transform(small_X)

    assert bmus.shape == (small_X.shape[0], 2)
    assert scat.shape == (small_X.shape[0], 2)
    assert unit_ids.shape == (small_X.shape[0],)
    assert radii.shape == (small_X.shape[0],)


def test_unit_impurity_matrix_shape(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    impurity = som_classifier.unit_impurity()

    assert impurity.shape == (2, 2)
    assert np.all(impurity >= 0)


def test_fit_twice_warns_and_uses_existing_codebook(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    old_codebook = som_classifier.som_.codebook.copy()

    with pytest.warns(UserWarning):
        som_classifier.fit(small_X, small_y)

    # Codebook should change because it trains further — not remain identical
    assert not np.allclose(old_codebook, som_classifier.som_.codebook)


def test_fit_with_unlabeled_data_warns(som_classifier, small_X):
    y_unlabeled = np.full(small_X.shape[0], som_classifier.unlabeled_label)

    with pytest.warns(UserWarning):
        som_classifier.fit(small_X, y_unlabeled)

    assert som_classifier._labeled_data is False
    assert som_classifier.classes_ is None


def test_negative_radius_sets_default(som_classifier, small_X, small_y):
    som_classifier.radius_0 = -0.5
    som_classifier.fit(small_X, small_y)

    # radius_0 replaced based on min dimension
    assert som_classifier.radius_0 == pytest.approx(2 * 0.5)


def test_predict_before_fit_raises(som_classifier, small_X):
    with pytest.raises(NotFittedError):
        som_classifier.predict(small_X)


def test_predict_on_unlabeled_training(som_classifier, small_X):
    y_unlabeled = np.full(small_X.shape[0], som_classifier.unlabeled_label)

    with pytest.warns(UserWarning):
        som_classifier.fit(small_X, y_unlabeled)

    preds = som_classifier.predict(small_X)
    assert np.all(preds == som_classifier.unlabeled_label)


def test_predict_proba_before_fit_raises(som_classifier, small_X):
    with pytest.raises(NotFittedError):
        som_classifier.predict_proba(small_X)


def test_predict_proba_unlabeled_training_warns(som_classifier, small_X):
    y_unlabeled = np.full(small_X.shape[0], som_classifier.unlabeled_label)

    with pytest.warns(UserWarning):
        som_classifier.fit(small_X, y_unlabeled)

    proba = som_classifier.predict_proba(small_X)
    assert proba.shape == (small_X.shape[0], 1)
    assert np.all(proba == som_classifier.unlabeled_label)


def test_annotation_zero_support_units(som_classifier, small_X, small_y):
    # Fewer training samples than SOM units => some units get zero support
    som_classifier.fit(small_X[0:3, :], small_y[0:3])

    labels = som_classifier.som_unit_labels_
    assert labels.shape == (2, 2)
    # Some will be -1
    assert -1 in labels


def test_transform_single_sample(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    x1 = small_X[:1]

    bmus, scat, ids, radii = som_classifier.transform(x1)

    assert bmus.shape == (1, 2)
    assert scat.shape == (1, 2)
    assert ids.shape == (1,)
    assert radii.shape == (1,)


def test_unit_impurity_gini_vs_entropy_positive(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)

    ent = som_classifier.unit_impurity('entropy')
    gin = som_classifier.unit_impurity('gini')

    assert ent.shape == (2, 2)
    assert gin.shape == (2, 2)
    assert np.all(ent >= 0)
    assert np.all(gin >= 0)


def test_unit_impurity_unlabeled_raises_warning(som_classifier, small_X):
    y_unlabeled = np.full(small_X.shape[0], som_classifier.unlabeled_label)
    som_classifier.fit(small_X, y_unlabeled)

    with pytest.warns(UserWarning):
        imp = som_classifier.unit_impurity()

    assert np.isinf(imp).all()


def test_unpredictable_classes_returns_missing(som_classifier, small_X, small_y):

    # Class 2 will not be majority class, should be missing
    X = np.vstack([small_X, small_X[:1, :]])
    y = np.concatenate([small_y, [2]])

    som_classifier.fit(X, y)
    missing = som_classifier.unpredictable_classes()

    assert 2 in missing


def test_unpredictable_classes_no_missing(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    missing = som_classifier.unpredictable_classes()
    assert missing.size == 0


def test_save_and_load_roundtrip(som_classifier, small_X, small_y, tmp_path):
    som_classifier.fit(small_X, small_y)

    fname = tmp_path / "som_cls.pkl"
    som_classifier.save(filename=fname.name, filepath=str(tmp_path))

    loaded = som_classifier.load(filename=fname.name, filepath=str(tmp_path))

    assert isinstance(loaded, type(som_classifier))
    assert loaded.is_fitted_
    assert loaded.som_dimensions == som_classifier.som_dimensions


def test_score_returns_valid_macro_f1(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    score = som_classifier.score(small_X, small_y)
    assert 0.0 <= score <= 1.0


def test_score_with_sample_weights(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    weights = np.ones_like(small_y)
    score = som_classifier.score(small_X, small_y, sample_weight=weights)
    assert 0.0 <= score <= 1.0


def test_activation_frequencies_shape(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    freqs = som_classifier.activation_frequencies(small_X)

    assert freqs.shape == (2, 2)
    assert np.isclose(freqs.sum(), 1.0, atol=1e-6)


def test_activation_frequencies_nonnegative(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    freqs = som_classifier.activation_frequencies(small_X)
    assert np.all(freqs >= 0)


def test_quantization_error_is_nonnegative(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    qe = som_classifier.quantization_error(small_X)
    assert qe >= 0
    assert isinstance(qe, float)


def test_quantization_error_zero_on_exact_codebook(som_classifier, small_X, small_y):
    # Fit to get initialized SOM
    som_classifier.fit(small_X, small_y)

    # Flatten codebook from (rows, cols, features) -> (rows*cols, features)
    codebook_flat = som_classifier.som_.codebook.reshape(-1, som_classifier.n_features_in_)

    qe = som_classifier.quantization_error(codebook_flat)

    assert qe == pytest.approx(0.0)


def test_topographic_error_returns_float(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    te = som_classifier.topographic_error(small_X)
    assert 0.0 <= te <= 1.0
    assert isinstance(te, float)


def test_topographic_error_requires_planar_rectangular(som_classifier, small_X, small_y):
    som_classifier.som_grid_type = 'hexagonal'  # invalid for computation
    som_classifier.fit(small_X, small_y)

    with pytest.raises(NotImplementedError):
        som_classifier.topographic_error(small_X)


def test_mean_impurity_matches_mean_of_unit_impurity(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)

    u = som_classifier.unit_impurity('entropy')
    m = som_classifier.mean_impurity('entropy')

    assert m == pytest.approx(u.mean())


def test_mean_impurity_gini(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    m = som_classifier.mean_impurity('gini')

    assert isinstance(m, float)
    assert m >= 0


def test_mean_impurity_unlabeled_raises_warning(som_classifier, small_X):
    y_unlabeled = np.full(small_X.shape[0], som_classifier.unlabeled_label)

    som_classifier.fit(small_X, y_unlabeled)

    with pytest.warns(UserWarning):
        m = som_classifier.mean_impurity('entropy')

    assert np.isinf(m)


def test_surface_state_and_bmu_somoclu_vs_custom_small(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    som = som_classifier.som_

    surface_state_somoclu = som.get_surface_state(data=small_X)
    surface_state_custom = som_classifier._custom_get_surface_state(data=small_X)

    bmus_somoclu = som.get_bmus(activation_map=surface_state_somoclu)
    bmus_custom = som_classifier._custom_get_bmus(activation_map=surface_state_custom)

    assert np.allclose(surface_state_somoclu, surface_state_custom)
    assert np.allclose(bmus_somoclu, bmus_custom)


def test_surface_state_and_bmu_somoclu_vs_custom(
        large_som_classifier, large_X, large_y
):
    large_som_classifier.fit(large_X, large_y)
    som = large_som_classifier.som_

    surface_state_somoclu = som.get_surface_state(data=large_X)
    surface_state_custom = large_som_classifier._custom_get_surface_state(data=large_X)

    bmus_somoclu = som.get_bmus(activation_map=surface_state_somoclu)
    bmus_custom = large_som_classifier._custom_get_bmus(activation_map=surface_state_custom)

    assert np.allclose(surface_state_somoclu, surface_state_custom)
    assert np.allclose(bmus_somoclu, bmus_custom)


def test_hyperparameter_tuning(som_classifier, small_X, small_y):
    som_classifier.hyperparameter_tuning(
        small_X,
        small_y,
        param_grid={
            'som_dimensions': [(3,3), (4,4)],
            'n_epochs': [5, 10],
        },
        cv=2
    )
    assert som_classifier.grid_search_.best_params_ is not None
    assert isinstance(som_classifier.grid_search_.best_params_, dict)


def test_fit_second_time_different_features_raises(som_classifier, small_X, small_y):
    som_classifier.fit(small_X, small_y)
    X_bigger = np.random.rand(10, small_X.shape[1] + 2)
    with pytest.raises(ValueError):
        som_classifier.fit(X_bigger, small_y)


def test_predict_warns_on_unknown_bmu(som_classifier, small_X, small_y):
    # Force some units to have zero support by using tiny dataset
    som_classifier.fit(small_X[:2], small_y[:2])
    with pytest.warns(UserWarning, match="the BMU has no label"):
        som_classifier.predict(small_X)


def test_annotate_som_warns_on_zero_support_units(som_classifier, small_X, small_y):
    # Fewer training samples than SOM units => some units get zero support
    with pytest.warns(UserWarning, match='SOM nodes are not BMU'):
        som_classifier.fit(small_X[0:3, :], small_y[0:3])




