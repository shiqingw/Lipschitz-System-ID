import sys
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np 

# draw lip fitting
x = [1.005253077, 1.206303716, 1.447564483, 1.737077355, 2.084492922, 2.501391411]
y_test = [4.000317403, 2.561618658, 1.257392652, 0.300309092, 0.005403201, 0.005403907]
y_train = [4.017039065, 2.572158738, 1.262735015, 0.301797486, 0.005414135, 0.005403261]
xlabel = r"$\gamma$"
save_name = "00lip_fitting_train_test"

system_lipschitz = np.linalg.norm(np.array([[-0.2, 2.0], [-2.0, -0.2]]), 2)

# Draw the curve
print("==> Drawing...")
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({"text.usetex": True,
                        "text.latex.preamble": r"\usepackage{amsmath}"})
plt.rcParams.update({'pdf.fonttype': 42})

plot_height_2d = 6
plot_width_2d = 14
fig = plt.figure(figsize=(14, 6))
cells = [121, 122]
ylabels = ['Training MSE', 'Test MSE']

# Draw train MSE
ax = fig.add_subplot(cells[0])
ax.plot(x, y_train, color="tab:blue", linestyle='solid', linewidth=3, marker='o', 
         markersize=12, zorder=2.0)
ax.vlines(system_lipschitz, 0, max(y_train), colors="black", linestyles="dashed", linewidth=3, zorder=2.0)
ax.text(system_lipschitz, 0.1, r"$K \approx {:.02f}$".format(system_lipschitz), ha='left', va='center', fontsize = 30)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.8, 3])

labelsize = 30
ticksize = 30
ax.set_xlabel(xlabel, fontsize=labelsize)
# ax.set_ylabel(ylabels[0], fontsize=labelsize)
ax.set_title(ylabels[0], fontsize=labelsize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=ticksize)
ax.tick_params(axis='both', which='minor', labelsize=ticksize)
plt.grid()
plt.tight_layout()

# Draw test MSE
ax = fig.add_subplot(cells[1])
ax.plot(x, y_test, color="tab:red", linestyle='solid', linewidth=3, marker='o', 
         markersize=12, zorder=2.0)
ax.vlines(system_lipschitz, 0, max(y_test), colors="black", linestyles="dashed", linewidth=3, zorder=2.0)
ax.text(system_lipschitz, 0.1, r"$K \approx {:.02f}$".format(system_lipschitz), ha='left', va='center', fontsize = 30)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim([0.8, 3])

ax.set_xlabel(xlabel, fontsize=labelsize)
# ax.set_ylabel(ylabels[1], fontsize=labelsize)
ax.set_title(ylabels[1], fontsize=labelsize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=ticksize)
ax.tick_params(axis='both', which='minor', labelsize=ticksize)
plt.grid()
plt.tight_layout()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', fancybox=True, ncol=3, fontsize=ticksize, bbox_to_anchor=(0.5, -0.15))
# plt.show()

# fig_height_2d = 6.3
# fig_width_2d = 14
# bbox_2d = Bbox.from_bounds((plot_width_2d - fig_width_2d)/2, -0.75, fig_width_2d, fig_height_2d)

figure_dir = "{}/eg1_results/{:03d}".format(str(Path(__file__).parent.parent), 0)
plt.savefig(figure_dir + "/" + save_name + ".pdf", bbox_inches="tight")