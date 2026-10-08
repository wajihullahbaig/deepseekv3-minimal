import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_metrics(train_losses, val_losses, test_losses,
                 train_accuracies=None, val_accuracies=None, test_accuracies=None, out_dir="plots"):
    """Write loss (and accuracy) curves per epoch to plots/training_metrics.png, loss_plot.png, accuracy_plot.png."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    with_acc = all(a is not None for a in (train_accuracies, val_accuracies, test_accuracies))
    series = [("Train", "tab:blue", "o"), ("Validation", "tab:green", "s"), ("Test", "tab:red", "^")]

    def draw(ax, data, ylabel, legend_loc):
        for (name, color, marker), values in zip(series, data):
            ax.plot(epochs, values, color=color, marker=marker, label=f"{name} ({values[-1]:.4f})")
            if len(values) > 3:
                window = min(5, len(values) // 3)
                ax.plot(range(window, len(values) + 1), _moving_average(values, window), color=color,
                        linestyle="--", alpha=0.5)
        ax.set_title(f"{ylabel} vs. epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc=legend_loc)

    fig, axes = plt.subplots(1, 2 if with_acc else 1, figsize=(15 if with_acc else 8, 6), squeeze=False)
    draw(axes[0, 0], (train_losses, val_losses, test_losses), "Loss", "upper right")
    if with_acc:
        draw(axes[0, 1], (train_accuracies, val_accuracies, test_accuracies), "Accuracy", "lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_metrics.png"), dpi=200, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 6))
    draw(ax, (train_losses, val_losses, test_losses), "Loss", "upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_plot.png"), dpi=200, bbox_inches="tight")
    if with_acc:
        fig, ax = plt.subplots(figsize=(8, 6))
        draw(ax, (train_accuracies, val_accuracies, test_accuracies), "Accuracy", "lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "accuracy_plot.png"), dpi=200, bbox_inches="tight")
    plt.close("all")
