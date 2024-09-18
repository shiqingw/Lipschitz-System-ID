import sys
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np 

save_name = "00lip_fitting_train_test"
figure_dir = "{}/eg2_results/{:03d}".format(str(Path(__file__).parent.parent), 0)

# draw lip fitting
x = [0.502017498, 1.004034996, 2.008069992, 4.016139984]
y_test = [1.609E+00, 1.414E-02, 2.663E-03, 2.655E-03]
y_train = [1.609E+00, 1.402E-02, 2.680E-03, 2.672E-03]
xlabel = r"$\gamma$"
system_lipschitz = 1.65

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
ax.set_xlim([0.4, 5])

labelsize = 30
ticksize = 30
ax.set_xlabel(xlabel, fontsize=labelsize)
# ax.set_ylabel(ylabels[0], fontsize=labelsize)
ax.set_title(ylabels[0], fontsize=labelsize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=ticksize)
ax.tick_params(axis='y', which='minor', labelsize=ticksize)
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
ax.set_xlim([0.4, 5])

ax.set_xlabel(xlabel, fontsize=labelsize)
# ax.set_ylabel(ylabels[1], fontsize=labelsize)
ax.set_title(ylabels[1], fontsize=labelsize, pad=10)
ax.tick_params(axis='both', which='major', labelsize=ticksize)
ax.tick_params(axis='y', which='minor', labelsize=ticksize)
plt.grid()
plt.tight_layout()

# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', fancybox=True, ncol=3, fontsize=ticksize, bbox_to_anchor=(0.5, -0.15))
# plt.show()

# fig_height_2d = 6.3
# fig_width_2d = 14
# bbox_2d = Bbox.from_bounds((plot_width_2d - fig_width_2d)/2, -0.75, fig_width_2d, fig_height_2d)

plt.savefig(figure_dir + "/" + save_name + ".pdf", bbox_inches="tight")