import matplotlib.pyplot as plt
import numpy as np

def draw_curve(data, num_epoch, config, ylabel, results_dir):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=True)
    ax.plot(np.arange(0, num_epoch).reshape(num_epoch, 1), data, linewidth=1)
    ax.set_xlabel("epochs", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=10)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(results_dir, dpi=config.dpi)
    plt.close()