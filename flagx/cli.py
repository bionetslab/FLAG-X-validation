
import click
import yaml
import os
import matplotlib
matplotlib.use('Agg')

# from datetime import datetime
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


@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for initialization. Includes train parameters.')
def init_train_save(config):
    """
    Instantiate a new pipeline, train and save. All parameters must be provided via config file in YAML format.
    """
    cfg = load_yaml(config)

    filename_pipeline = cfg.pop('pipeline_filename', 'trained_gating_pipeline.pkl')

    # Precedence: save_dir cli > save_path cfg > default
    save_dir_pipeline = cfg.pop('save_dir', './')
    cfg['save_path'] = save_dir_pipeline

    # Instantiate the pipeline, train and save
    gp = GatingPipeline(**cfg)
    gp.train()
    gp.save(filepath=None, filename=filename_pipeline)  # filepath=None => use self.save_path set to save_dir from cfg

    click.secho(
        f'# --- Pipeline instantiated, trained and saved to {os.path.join(gp.save_path, filename_pipeline)} --- #',
        fg='green'
    )


@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for inference')
def load_infer_save(config):
    """
    Load a trained pipeline and run inference.
    Load path and filename can be provided via YAML or overridden by CLI flags.
    """

    # Load the config YAML
    cfg = load_yaml(config)


    filename_pipeline = cfg.pop('filename_pipeline', 'trained_gating_pipeline.pkl')

    load_dir_pipeline = cfg.pop('load_dir_pipeline', './')


    # Load the pre-trained pipeline
    gp = GatingPipeline.load(filepath=load_dir_pipeline, filename=filename_pipeline)


    # Precedence: filename cli > filename cfg > default, update config
    filename_inference = cfg.pop('filename_inference', 'annotated_data.fcs')
    save_dir_inference = cfg.pop('save_dir_inference', './')
    os.makedirs(save_dir_inference, exist_ok=True)

    cfg['save_filename'] = filename_inference
    cfg['save_path'] = save_dir_inference

    # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Inference
    gp.inference(**cfg)

    click.secho(
        f'# --- Inference complete. Results saved to {cfg['save_path']} --- #',
        fg='green'
    )


if __name__ == '__main__':
    cli()
