
import click
import yaml
import os
import matplotlib
matplotlib.use('Agg')

from datetime import datetime
from .pipeline import GatingPipeline



def load_yaml(path):
    """
    Load YAML config and auto-convert specific list fields to tuples:
    - gating_method_kwargs.som_dimensions
    - gating_method_kwargs.layer_sizes
    - dim_red_methods
    - val_range

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Processed configuration dictionary.
    """

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Auto-convert som_dimensions and layer_sizes from list to tuple if present
    if 'gating_method_kwargs' in cfg:

        gating_kwargs = cfg['gating_method_kwargs']

        som_dims = gating_kwargs.get('som_dimensions', None)
        if som_dims is not None and isinstance(som_dims, list):
            gating_kwargs['som_dimensions'] = tuple(som_dims)

        layer_sizes = gating_kwargs.get('layer_sizes', None)
        if layer_sizes is not None and isinstance(layer_sizes, list):
            gating_kwargs['layer_sizes'] = tuple(layer_sizes)

        cfg['gating_method_kwargs'] = gating_kwargs

    # Auto-convert dim_red_methods from list to tuple if present
    dim_red_methods = cfg.get('dim_red_methods', None)
    if dim_red_methods is not None and isinstance(dim_red_methods, list):
        cfg['dim_red_methods'] = tuple(dim_red_methods)

    # Auto-convert dim_red_method_kwargs from list to tuple if present
    dim_red_method_kwargs = cfg.get('dim_red_method_kwargs', None)
    if dim_red_method_kwargs is not None and isinstance(dim_red_method_kwargs, list):
        cfg['dim_red_method_kwargs'] = tuple(dim_red_method_kwargs)

    # Auto-convert val_range from list to tuple if present
    val_range = cfg.get('val_range', None)
    if val_range is not None and isinstance(val_range, list):
        cfg['val_range'] = tuple(val_range)

    return cfg


@click.group()
def cli():
    """CLI for managing GatingPipeline workflows."""


# ----- Step-by-step Commands -----

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for initialization. Includes train parameters.')
@click.option('--save-dir', type=click.Path(), help='Directory to save the pipeline (overrides YAML)')
@click.option('--filename', type=str, help='Filename to save the pipeline (overrides YAML)')
def init(config, save_dir, filename):
    """
    Instantiate a new pipeline and save it.
    Save path and filename can be provided via YAML or overridden by CLI flags.
    """

    # Load the config file
    cfg = load_yaml(config)

    # Precedence: filename cli > filename cfg > default
    filename_cfg = cfg.pop('pipeline_filename', None)
    filename_save = filename or filename_cfg or 'gating_pipeline.pkl'

    # Precedence: save_dir cli > save_dir cfg > default
    save_dir_cfg = cfg.pop('save_dir', None)
    cfg['save_path'] = save_dir or save_dir_cfg or os.getcwd()

    # Instantiate the pipeline and save
    gp = GatingPipeline(**cfg)
    gp.save(filepath=None, filename=filename_save)  # filepath=None => use self.save_path
    click.secho(f"# ### Pipeline instantiated and saved to {os.path.join(gp.save_path, filename_save)}", fg='green')


@cli.command()
@click.option('--load-dir', required=True, type=click.Path(exists=True), help='Directory to load the pipeline from')
@click.option('--filename', type=str, default='gating_pipeline.pkl', help='Filename to load the pipeline')
def train(load_dir, filename):
    """Load a pipeline, train it, and save it back."""
    gp = GatingPipeline.load(filepath=load_dir, filename=filename)
    gp.train()
    filename_save = 'trained_' + filename
    gp.save(filepath=None, filename=filename_save)  # filepath=None => use self.save_path
    click.secho(
        f"# ### Training complete and pipeline saved to {os.path.join(gp.save_path, filename_save)}",
        fg='green'
    )


@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for inference')
@click.option('--load-dir', type=click.Path(), help='Directory to load the pipeline from (overrides YAML)')
@click.option('--load-filename', type=str, help='Filename to load the pipeline from (overrides YAML)')
@click.option('--save-dir', type=click.Path(), help='Directory to save the results to (overrides YAML)')
@click.option('--save-filename', type=str, help='Filename to save the results to (overrides YAML)')
def infer(config, load_dir, load_filename, save_dir, save_filename):
    """
    Load a trained pipeline and run inference.
    Load path and filename can be provided via YAML or overridden by CLI flags.
    """

    # Load the config YAML
    cfg = load_yaml(config)

    # Set save_sample_wise parameter to False, hardcoded for the cli
    cfg['save_sample_wise'] = False

    # Precedence: load_fn cli > load_fn cfg > default
    filename_cfg = cfg.pop('pipeline_filename', None)
    filename_load = load_filename or filename_cfg or 'trained_gating_pipeline.pkl'

    # Precedence: load_dir cli > load_dir cfg > default
    load_dir_cfg = cfg.pop('load_dir', None)
    load_dir_load = load_dir or load_dir_cfg or os.getcwd()

    # Load the pre-trained pipeline
    gp = GatingPipeline.load(filepath=load_dir_load, filename=filename_load)

    # Precedence: filename cli > filename cfg > default, update config
    save_filename_cfg = cfg.pop('save_filename', None)
    filename_save = save_filename or save_filename_cfg or 'annotated_data.fcs'
    cfg['save_filenames'] = filename_save  # kwarg must be filenames, see pipeline .inference()

    # Precedence: save_dir cli > save_path cfg > default, update config, create dir if necessary
    save_dir_cfg = cfg.pop('save_dir', None)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir_default = os.path.join(gp.save_path, 'inference_' + timestamp)

    save_dir_save = save_dir or save_dir_cfg or save_dir_default

    cfg['save_path'] = save_dir_save
    os.makedirs(save_dir_save, exist_ok=True)

    # Inference
    gp.inference(**cfg)

    click.secho(
        f"# ### Inference complete. Results saved to {os.path.join(cfg['save_path'], cfg['save_filenames'])}",
        fg='green'
    )


# ----- Multi-step Workflow Command -----
@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for initialization. Includes train parameters.')
@click.option('--save-dir', type=click.Path(), help='Directory to save the pipeline (overrides YAML)')
@click.option('--filename', type=str, help='Filename to save the pipeline (overrides YAML)')
def init_train(config, save_dir, filename):
    """
    Instantiate a new pipeline, train and save.
    Save path and filename can be provided via YAML or overridden by CLI flags.
    """
    cfg = load_yaml(config)

    # Precedence: filename cli > filename cfg > default
    filename_cfg = cfg.pop('pipeline_filename', None)
    filename_save = filename or filename_cfg or 'trained_gating_pipeline.pkl'

    # Precedence: save_dir cli > save_path cfg > default
    save_dir_cfg = cfg.pop('save_dir', None)
    cfg['save_path'] = save_dir or save_dir_cfg or os.getcwd()

    # Instantiate the pipeline, train and save
    gp = GatingPipeline(**cfg)
    gp.train()
    gp.save(filepath=None, filename=filename_save)  # filepath=None => use self.save_path

    click.secho(
        f"# ### Pipeline instantiated, trained and saved to {os.path.join(gp.save_path, filename_save)}",
        fg='green'
    )


@cli.command()
@click.option('--init-config', required=True, type=click.Path(exists=True), help='YAML for initialization. Includes train parameters.')
@click.option('--infer-config', type=click.Path(exists=True), help='YAML for inference')
@click.option('--save-dir', type=click.Path(), help='Directory to save the pipeline (overrides YAML)')
@click.option('--filename', type=str, help='Filename to save the pipeline (overrides YAML)')
def init_train_infer(init_config, infer_config, save_dir, filename):
    """
    Instantiate a new pipeline, train save, and run inference on the train data.
    Save path and filename can be provided via YAML or overridden by CLI flags.
    """

    cfg_init = load_yaml(init_config)

    # Precedence: filename cli > filename cfg > default
    filename_cfg = cfg_init.pop('pipeline_filename', None)
    filename_save = filename or filename_cfg or 'trained_gating_pipeline.pkl'

    # Precedence: save_dir cli > save_path cfg > default
    save_dir_cfg = cfg_init.pop('save_dir', None)
    cfg_init['save_path'] = save_dir or save_dir_cfg or os.getcwd()

    # Instantiate the pipeline, train and save
    gp = GatingPipeline(**cfg_init)
    gp.train()
    gp.save(filepath=None, filename=filename_save)  # filepath=None => use self.save_path

    # Run inference on the train data
    if infer_config is not None:
        cfg_infer = load_yaml(infer_config)
    else:
        cfg_infer = dict()

    # Set save_sample_wise parameter to False, hardcoded for the cli
    cfg_infer['save_sample_wise'] = False

    # Overwrite the data_dile_path/names parameters to make sure the train data is used
    cfg_infer['data_file_path'] = cfg_init['train_data_file_path']
    cfg_infer['data_file_names'] = cfg_init['train_data_file_names']

    # Overwrite the save_path and save_filenames parameters
    cfg_infer['save_path'] = cfg_init['save_path']
    cfg_infer['save_filenames'] = 'annotated_train_data.fcs'

    # Set some defaults for inference, if none were passed
    cfg_infer.setdefault('gate', True)
    cfg_infer.setdefault('dim_red_methods', ('pca', 'tsne'))
    cfg_infer.setdefault('dim_red_method_kwargs', (None, {'n_jobs': -1}))
    cfg_infer.setdefault('val_range', (0.0, 2**20))
    cfg_infer.setdefault('keep_unscaled', True)
    cfg_infer.setdefault('fcs_metadata_dicts', None)

    gp.inference(**cfg_infer)

    click.secho(
        f"# ### Pipeline instantiated, trained and inference ran on train data. "
        f"Results saved to {save_dir}",
        fg='green'
    )


if __name__ == '__main__':
    cli()
