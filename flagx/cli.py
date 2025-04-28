
import click
import yaml
import os
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
    cfg = load_yaml(config)

    gp_save_dir = cfg.pop('pipeline_save_path', None)
    gp_file_name = cfg.pop('pipeline_filename', None)

    # If CLI flags provided, they override
    save_dir = save_dir or gp_save_dir or '.'
    filename = filename or gp_file_name or 'gating_pipeline.pkl'

    # If 'train_data_manager_save_path' is None set to 'save_dir'
    tdm_sp = cfg.get('train_data_manager_save_path', None)
    if tdm_sp is None:
        cfg['train_data_manager_save_path'] = save_dir

    # Instantiate the pipeline and save
    gp = GatingPipeline(**cfg)
    gp.save(filepath=save_dir, filename=filename)
    click.echo(f"# ### Pipeline instantiated and saved to {os.path.join(save_dir, filename)}")


@cli.command()
@click.option('--load-dir', required=True, type=click.Path(exists=True), help='Directory to load the pipeline from')
@click.option('--filename', type=str, default='gating_pipeline.pkl', help='Filename to load the pipeline')
def train(load_dir, filename):
    """Load a pipeline, train it, and save it back."""
    gp = GatingPipeline.load(filepath=load_dir, filename=filename)
    gp.train()
    filename = 'trained_' + filename
    gp.save(filepath=load_dir, filename=filename)
    click.echo(f"# ### Training complete and pipeline saved to {os.path.join(load_dir, filename)}")


@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for inference')
@click.option('--load-dir', type=click.Path(), help='Directory to load the pipeline from (overrides YAML)')
@click.option('--filename', type=str, help='Filename to load the pipeline from (overrides YAML)')
def infer(config, load_dir, filename):
    """
    Load a trained pipeline and run inference.
    Load path and filename can be provided via YAML or overridden by CLI flags.
    """

    # Load the config YAML
    cfg = load_yaml(config)

    # Try getting load args from YAML first
    gp_load_dir = cfg.pop('pipeline_load_path', None)
    gp_file_name = cfg.pop('pipeline_filename', None)

    load_dir = load_dir or gp_load_dir or '.'
    filename = filename or gp_file_name or 'trained_gating_pipeline.pkl'

    gp = GatingPipeline.load(filepath=load_dir, filename=filename)
    results_path = cfg.get('save_path', os.getcwd())
    gp.inference(**cfg)
    click.echo(f"# ### Inference complete. Results saved to {results_path}")


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

    gp_save_dir = cfg.pop('pipeline_save_path', None)
    gp_file_name = cfg.pop('pipeline_filename', None)

    # If CLI flags provided, they override
    save_dir = save_dir or gp_save_dir or '.'
    filename = filename or gp_file_name or 'trained_gating_pipeline.pkl'

    # Instantiate the pipeline, train and save
    gp = GatingPipeline(**cfg)
    gp.train()
    gp.save(filepath=save_dir, filename=filename)
    click.echo(f"# ### Pipeline instantiated, trained and saved to {os.path.join(save_dir, filename)}")


@cli.command()
@click.option('--init-config', required=True, type=click.Path(exists=True), help='YAML for initialization. Includes train parameters.')
@click.option('--infer-config', required=True, type=click.Path(exists=True), help='YAML for inference')
@click.option('--save-dir', type=click.Path(), help='Directory to save the pipeline (overrides YAML)')
@click.option('--filename', type=str, help='Filename to save the pipeline (overrides YAML)')
def init_train_infer(init_config, infer_config, save_dir, filename):
    """
    Instantiate a new pipeline, train save, and run inference on the train data.
    Save path and filename can be provided via YAML or overridden by CLI flags.
    """

    cfg_init = load_yaml(init_config)

    gp_save_dir = cfg_init.pop('pipeline_save_path', None)
    gp_file_name = cfg_init.pop('pipeline_filename', None)

    # If CLI flags provided, they override
    save_dir = save_dir or gp_save_dir or '.'
    filename = filename or gp_file_name or 'trained_gating_pipeline.pkl'

    # Instantiate the pipeline and save
    gp = GatingPipeline(**cfg_init)
    gp.train()
    gp.save(filepath=save_dir, filename=filename)

    # Run inference on the train data
    cfg_infer = load_yaml(infer_config)

    # Overwrite some of the arguments to make sure the train data is used
    cfg_infer['data_file_path'] = cfg_init['train_data_file_path']
    cfg_infer['data_file_names'] = cfg_init['train_data_file_names']

    # Set some defaults for inference, if none were passed
    cfg_infer.setdefault('gate', True)
    cfg_infer.setdefault('dim_red_methods', ('pca', 'tsne'))
    cfg_infer.setdefault('dim_red_method_kwargs', (None, {'n_jobs': -1}))
    cfg_infer.setdefault('save_sample_wise', False)
    cfg_infer.setdefault('save_path', save_dir)  # Save into train save_dir if no save_dir was passed
    cfg_infer.setdefault('save_filenames', 'train_data.fcs')
    cfg_infer.setdefault('val_range', (0.0, 2**20))
    cfg_infer.setdefault('keep_unscaled', True)
    cfg_infer.setdefault('fcs_metadata_dicts', None)

    gp.inference(**cfg_infer)

    click.secho(
        f"# ### Pipeline instantiated, trained and inference ran on train data. "
        f"Results saved to {os.path.join(save_dir, filename)}",
        fg = 'green'
    )


if __name__ == '__main__':
    cli()
