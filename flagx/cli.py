
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

    # Auto-convert som_dimensions from list to tuple if present
    gating_kwargs = cfg.get('gating_method_kwargs', {})
    som_dims = gating_kwargs.get('som_dimensions', None)
    if som_dims is not None and isinstance(som_dims, list):
        gating_kwargs['som_dimensions'] = tuple(som_dims)

    # Auto-convert layer_sizes from list to tuple if present
    layer_sizes = gating_kwargs.get('layer_sizes', None)
    if layer_sizes is not None and isinstance(layer_sizes, list):
        gating_kwargs['layer_sizes'] = tuple(layer_sizes)

    cfg['gating_method_kwargs'] = gating_kwargs

    # Auto-convert dim_red_methods from list to tuple if present
    dim_red_methods = cfg.get('dim_red_methods', None)
    if dim_red_methods is not None and isinstance(dim_red_methods, list):
        cfg['dim_red_methods'] = tuple(dim_red_methods)

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
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for initialization')
@click.option('--save-dir', required=True, type=click.Path(), help='Directory to save the pipeline')
@click.option('--filename', default='gating_pipeline.pkl', help='Filename to save the pipeline')
def init(config, save_dir, filename):
    """Instantiate a new pipeline and save it."""
    cfg = load_yaml(config)
    gp = GatingPipeline(**cfg)
    gp.save(filepath=save_dir, filename=filename)
    click.echo(f"# ### Pipeline instantiated and saved to {os.path.join(save_dir, filename)}")


@cli.command()
@click.option('--load-dir', required=True, type=click.Path(exists=True), help='Directory to load the pipeline from')
@click.option('--filename', default='gating_pipeline.pkl', help='Filename to load the pipeline')
def train(load_dir, filename, config):
    """Load a pipeline, train it, and save it back."""
    gp = GatingPipeline.load(filepath=load_dir, filename=filename)
    filename = 'trained_' + filename
    gp.save(filepath=load_dir, filename=filename)
    click.echo(f"# ### Training complete and pipeline saved to {os.path.join(load_dir, filename)}")


@cli.command()
@click.option('--load-dir', required=True, type=click.Path(exists=True), help='Directory to load the pipeline from')
@click.option('--filename', default='trained_gating_pipeline.pkl', help='Filename to load the pipeline')
@click.option('--config', required=True, type=click.Path(exists=True), help='YAML for inference')
def infer(load_dir, filename, config):
    """Load a pipeline and run inference."""
    gp = GatingPipeline.load(filepath=load_dir, filename=filename)
    infer_kwargs = load_yaml(config)
    results_path = infer_kwargs.get('save_path', os.getcwd())
    gp.inference(**infer_kwargs)
    click.echo(f"# ### Inference complete. Results saved to {results_path}")


# ----- One-shot Workflow Command -----

@cli.command()
@click.option('--init-config', type=click.Path(exists=True), help='YAML for pipeline initialization')
@click.option('--load-config', type=click.Path(exists=True), help='YAML for loading an existing pipeline')
@click.option('--infer-config', type=click.Path(exists=True), help='YAML for inference step')
def pipeline(init_config, load_config, train_config, infer_config, save_config):
    """
    Full pipeline run: instantiate or load → train → infer
    """

    if init_config and load_config:
        raise click.UsageError("Use only one of --init-config or --load-config.")

    if init_config:
        init_kwargs = load_yaml(init_config)
        gp = GatingPipeline(**init_kwargs)
        click.echo("# ### Pipeline instantiated.")
    elif load_config:
        load_kwargs = load_yaml(load_config)
        gp = GatingPipeline.load(**load_kwargs)
        click.echo("# ### Pipeline loaded.")
    else:
        raise click.UsageError("You must provide either --init-config or --load-config.")

    if train_config:
        gp.train()
        click.echo("# ### Training complete.")

    if infer_config:
        infer_kwargs = load_yaml(infer_config)
        gp.inference(**infer_kwargs)
        click.echo("# ### Inference complete.")

    # Todo: saving


if __name__ == '__main__':
    cli()
