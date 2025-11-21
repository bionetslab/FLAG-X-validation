
import os
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from pathlib import Path
from flagx import GatingPipeline
from flagx.gating import SomClassifier, MLPClassifier


TEST_DATA_DIR = Path(__file__).parent / 'test_data'
TRAIN_CHANNELS = [
    'FS INT', 'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5',
    'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB',
    'FL10 INT_CD45-KO'
]

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

@pytest.fixture(scope='function', params=['som', 'mlp'])
def gating_method(request):
    return request.param

@pytest.fixture
def gating_method_kwargs(gating_method):
    if gating_method == 'som':
        return dict(som_dimensions=(2, 2), n_epochs=3, verbosity=0)
    else:
        return dict(layer_sizes=(8, 4, 2), n_epochs=1, verbosity=0)


@pytest.fixture
def pipeline_kwargs(test_files, gating_method, gating_method_kwargs, tmp_path):
    """Common kwargs for creating a GatingPipeline instance."""

    return dict(
        train_data_file_path=str(TEST_DATA_DIR),
        train_data_file_names=test_files[0:3],
        train_data_file_type=None,
        save_path=str(tmp_path),
        channels=TRAIN_CHANNELS,
        label_key='label',
        downsampling_kwargs={'target_num_events': 100, 'stratified': True},
        gating_method=gating_method,
        gating_method_kwargs=gating_method_kwargs,
        verbosity=0,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_pipeline_initialization(test_files, tmp_path, pipeline_kwargs):

    pipe = GatingPipeline(**pipeline_kwargs)

    assert pipe.train_data_file_path == str(TEST_DATA_DIR)
    assert pipe.train_data_file_names == test_files[0:3]
    assert pipe.train_data_file_type is None
    assert pipe.save_path == str(tmp_path)
    assert os.path.isdir(pipe.save_path)
    assert pipe.train_data_manager_save_path == str(os.path.join(tmp_path, 'train_data_manager_output'))
    assert pipe.channels == TRAIN_CHANNELS
    assert pipe.label_key == 'label'


def test_train_creates_and_fits_classifier(pipeline_kwargs, gating_method,):
    pipe = GatingPipeline(**pipeline_kwargs)
    pipe.train()

    assert pipe.is_trained_ is True
    assert pipe.gating_module_ is not None
    assert isinstance(pipe.gating_module_, SomClassifier if gating_method == 'som' else MLPClassifier)
    assert hasattr(pipe.gating_module_, 'predict_proba')
    assert pipe.gating_module_.is_fitted_
    assert not pipe.binary_classes_
    assert pipe.prediction_threshold == 0.0


def test_train_sets_binary_flag_if_two_classes(pipeline_kwargs, gating_method,):
    # Relabel to binary labels
    pipeline_kwargs['relabel_data_kwargs'] = {
        'old_to_new_label_mapping': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1},
        'new_label_key': 'binary_labels'
    }
    pipe = GatingPipeline(**pipeline_kwargs)
    pipe.train()

    assert pipe.binary_classes_
    assert pipe.prediction_threshold == 0.5


def test_inference_without_training_raises(pipeline_kwargs, test_files):
    pipe = GatingPipeline(**pipeline_kwargs)

    with pytest.raises(NotFittedError):
        pipe.inference(data_file_path=str(TEST_DATA_DIR), data_file_names=test_files)


def test_inference_runs_after_training(pipeline_kwargs, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)
    pipe.train()

    output_dir = tmp_path / 'inference'
    output_dir.mkdir()

    pipe.inference(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=test_files,
        sample_wise=False,
        gate=True,
        dim_red_methods=('pca', ),
        dim_red_method_kwargs=None,
        save_path=str(output_dir),
        save_filename='out.fcs',
    )

    # Expected output file(s)
    out_file = output_dir / 'out.fcs'
    assert out_file.exists()


def test_inference_sample_wise_saves_multiple_files(pipeline_kwargs, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)

    output_dir = tmp_path / 'inference_sample_wise'
    output_dir.mkdir()

    pipe.inference(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=test_files[0:2],
        sample_wise=True,
        gate=False,
        save_path=str(output_dir),
        save_filename='annotated.fcs',
    )

    for i in range(2):
        name = f'annotated_sample_id_{i+1}.fcs'
        assert (output_dir / name).exists()


@pytest.mark.parametrize(
    'methods', [('umap',), ('pca',), ('tsne', ),('isomap',), ('locallylinearembedding',), ('spectralembedding',), ]
)
def test_dimensionality_reduction_methods(pipeline_kwargs, test_files, tmp_path, methods):
    pipe = GatingPipeline(**pipeline_kwargs)

    # This should not raise
    pipe.inference(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=test_files[0:2],
        sample_wise=False,
        gate=False,
        dim_red_methods=methods,
        save_path=str(tmp_path),
        save_filename='dimred.fcs',
    )

    assert (tmp_path / 'dimred.fcs').exists()


def test_som_dimred(pipeline_kwargs, gating_method, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)
    pipe.train()

    if gating_method == 'som':
        pipe.inference(
            data_file_path=str(TEST_DATA_DIR),
            data_file_names=test_files[0:2],
            gate=False,
            dim_red_methods=('som',),
            save_path=str(tmp_path),
            save_filename='dimred_som.fcs',
        )
        assert (tmp_path / 'dimred_som.fcs').exists()
    else:
        with pytest.raises(ValueError):
            pipe.inference(
                data_file_path=str(TEST_DATA_DIR),
                data_file_names=test_files[0:2],
                gate=False,
                dim_red_methods=('som',),
                save_path=str(tmp_path),
                save_filename='dimred_som.fcs',
            )


def test_save_and_load_pipeline_roundtrip(pipeline_kwargs, gating_method, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)
    pipe.train()

    save_path = tmp_path / 'model'
    save_path.mkdir()

    pipe.save(filepath=str(save_path))

    # The main pickle file must exist
    pkl = save_path / 'gating_pipeline.pkl'
    assert pkl.exists()

    # Load back
    pipe2 = GatingPipeline.load(filepath=str(save_path))

    assert isinstance(pipe2, GatingPipeline)
    assert pipe2.is_trained_ is True
    assert pipe2.gating_method == pipe.gating_method
    assert isinstance(pipe2.gating_module_, SomClassifier if gating_method == 'som' else MLPClassifier)

    # Test inference with loaded pipeline
    output_dir = tmp_path / 'inference_after_loading'
    output_dir.mkdir()

    pipe2.inference(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=test_files[0:2],
        sample_wise=False,
        gate=True,
        dim_red_methods=('pca', ),
        dim_red_method_kwargs=None,
        save_path=str(output_dir),
        save_filename='out.fcs',
    )

    out_file = output_dir / 'out.fcs'
    assert out_file.exists()


def test_no_labels_warns_or_raises(pipeline_kwargs, gating_method, test_files, tmp_path):
    # Remove label_key dict entry
    pipeline_kwargs.pop('label_key')
    pipeline_kwargs['downsampling_kwargs'] = {'target_num_events': 100, }
    pipe = GatingPipeline(**pipeline_kwargs)

    output_dir = tmp_path / 'inference_no_labels'
    output_dir.mkdir()

    if gating_method == 'som':

        pipe.train()

        with pytest.warns(UserWarning):
            pipe.inference(
                data_file_path=str(TEST_DATA_DIR),
                data_file_names=test_files,
                gate=True,   # gating requested, but no labels
                save_path=str(tmp_path),
            )

            pipe.inference(
                data_file_path=str(TEST_DATA_DIR),
                data_file_names=test_files,
                sample_wise=False,
                gate=True,
                dim_red_methods=('pca', ),
                dim_red_method_kwargs=None,
                save_path=str(output_dir),
                save_filename='out.fcs',
            )
    else:
        with pytest.raises(ValueError):
            pipe.train()


def test_dimred_invalid_method(pipeline_kwargs):
    pipe = GatingPipeline(**pipeline_kwargs)

    with pytest.raises(NotImplementedError):
        pipe._reduce_dimension_helper(
            xs=[np.zeros((10, 5))],
            dim_red_method='not_exist'
        )


def test_inference_dimred_mismatch(pipeline_kwargs, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)

    with pytest.raises(ValueError):
        pipe.inference(
            data_file_path=str(TEST_DATA_DIR),
            data_file_names=test_files,
            gate=False,
            dim_red_methods=('umap', 'pca'),
            dim_red_method_kwargs=({'n_neighbors': 5},),  # mismatch
            save_path=str(tmp_path),
        )


def test_inference_scale_channels_extra(pipeline_kwargs, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)

    output = tmp_path / 'scale_test'
    output.mkdir()

    # Should not raise; nonexistent channel just ignored
    pipe.inference(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=test_files,
        gate=False,
        scale_channels=['NOT_A_CHANNEL', 'FS INT'],
        save_path=str(output),
    )

    assert (output / 'annotated_data.fcs').exists()


def test_save_without_gating_module(tmp_path):
    pipe = GatingPipeline(save_path=str(tmp_path))
    pipe.gating_module_ = None
    pipe.is_trained_ = True

    pipe.save(filepath=str(tmp_path))
    assert (tmp_path / 'gating_pipeline.pkl').exists()

    loaded = GatingPipeline.load(filepath=str(tmp_path))
    assert loaded.gating_module_ is None


def test_data_pipeline_autolist(tmp_path):
    pipe = GatingPipeline(
        train_data_file_path=str(TEST_DATA_DIR),
        train_data_file_names=None,  # will autolist
        channels=['FS INT', 'SS INT'],
        label_key='label',
        save_path=str(tmp_path),
        gating_method='som',
        gating_method_kwargs={'som_dimensions': (2, 2), 'n_epochs': 1},
    )

    fdm, x, y = pipe._data_pipeline(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=None,
        data_file_type=None,
        label_key='label',
        downsampling_kwargs=None,
        save_meta_info=False,
    )

    assert isinstance(x, np.ndarray)
    assert len(fdm.anndata_list_) == 5  # number of test_files of type csv or fcs


def test_inference_no_train_no_gate(pipeline_kwargs, test_files, tmp_path):
    pipe = GatingPipeline(**pipeline_kwargs)

    outdir = tmp_path / 'notrain_nogate'
    outdir.mkdir()

    pipe.inference(
        data_file_path=str(TEST_DATA_DIR),
        data_file_names=test_files,
        gate=False,
        dim_red_methods=('pca',),
        save_path=str(outdir),
    )

    assert (outdir / 'annotated_data.fcs').exists()