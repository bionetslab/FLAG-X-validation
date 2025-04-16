
import numpy as np
import matplotlib.pyplot as plt
import os

from .._legacy_typing import Union, List


def plot_1d_metric_over_time(
        metric_train: np.ndarray,
        n_epochs: int,
        tracking_interval: int,
        metric_val: Union[np.ndarray, None] = None,
        title: Union[str, None] = None,
        xlabel: Union[str, None] = 'Epoch',
        ylabel: Union[str, None] = 'Metric',
        ax: Union[plt.Axes, None] = None,
        filename: Union[str, None] = None,
        filepath: Union[str, None] = None,
):

    # epochs = range(2, (metric_train.shape[0] + 1) * 2, 2)
    epochs = list(range(tracking_interval, n_epochs + 1, tracking_interval))
    if n_epochs % tracking_interval > 1:
        epochs += [n_epochs, ]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(epochs, metric_train, label='Train', color='blue')

    if metric_val is not None:
        ax.plot(epochs, metric_val, label='Val', color='orange')

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # ax.set_xticks(epochs)
    # ax.set_xticklabels(epochs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.legend()
    # ax.grid(True)

    if filename is not None:
        if filepath is None:
            filepath = os.getcwd()
        plt.savefig(os.path.join(filepath, filename))


def plot_nd_metric_over_time(
        metrics: np.ndarray,
        n_epochs: int,
        tracking_interval: int,
        title: Union[str, None] = None,
        xlabel: Union[str, None] = 'Epoch',
        ylabel: Union[str, None] = 'Metric',
        metric_labels: Union[List[str], None] = None,
        colors: Union[List[str], None] = None,
        ax: Union[plt.Axes, None] = None,
        filename: Union[str, None] = None,
        filepath: Union[str, None] = None,
):
    epochs = list(range(tracking_interval, n_epochs + 1, tracking_interval))
    if n_epochs % tracking_interval > 1:
        epochs += [n_epochs, ]

    n_metrics, _ = metrics.shape
    # epochs = range(2, (n_epochs + 1) * 2, 2)

    if colors is None:
        colors = plt.cm.tab10.colors

    if ax is None:
        fig, ax = plt.subplots()

    # Loop through each metric and plot
    for i in range(n_metrics):
        color = colors[i % len(colors)]
        label_train = f'{metric_labels[i]}' if metric_labels and i < len(metric_labels) else f'{i+1}'
        ax.plot(epochs, metrics[i], label=label_train, color=color, linestyle='-')

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # ax.set_xticks(epochs)
    # ax.set_xticklabels(epochs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.legend()
    # ax.grid(True)

    # Save the plot if filename is provided
    if filename is not None:
        if filepath is None:
            filepath = os.getcwd()
        plt.savefig(os.path.join(filepath, filename))








