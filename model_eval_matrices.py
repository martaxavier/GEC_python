import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob

# ===============================
# Load Reordering and Labels
# ===============================

# Load H2N reordering indices
H2N = scipy.io.loadmat('atlas/schaefer_H2N_ind.mat')
H2N_ind = H2N['H2N_ind'].flatten() - 1

# Load labels, then reorder
with open('atlas/Schaefer2018_100ParcelsCerSubcort_7Networks_order_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]
labels = [labels[i] for i in H2N_ind]

# ===============================
# Model Parameters
# ===============================

H2N_ordering = False # Set to True if you want to reorder matrices by H2N indices

# Define thresholds and lambda values
SCthresh_list = ['t10', 't40', 't90']
# model_type = "Grad_fccov"
# lambda_list = ['', 'λ1', 'λ0.1', 'λ0']   # For Grad_fccov
model_type = "pseudoGrad_fccov"
lambda_list = ['', 'λ0.001', 'λ0.0001', 'λ0']  # For pseudoGrad_fccov

# Define directories
os.chdir(os.path.dirname(os.path.abspath(__file__)))
simdir = 'simulations'
plotdir = os.path.join('plots', model_type)

# Yeo 7-network colors/names
network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Default', 'Cont']
network_colors = np.array([
    [0.3010, 0.7450, 0.9330],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.6,    0.6,    0.6],
    [0,      0.4470, 0.7410],
    [0.6350, 0.0780, 0.1840],
    [0.9290, 0.6940, 0.1250]
])

# ===========================
# Helper Functions
# ===========================

def pearson_corr(A, B):
    """Pearson correlation of lower triangle."""
    inds = np.tril_indices_from(A, k=-1)
    a, b = A[inds].flatten(), B[inds].flatten()
    return np.corrcoef(a, b)[0, 1]

def plot_network_bars(ax, labels, network_names, network_colors):
    """Colored bars above/left of matrix for each network."""
    n = len(labels)
    barwidth = 3
    for net_name, net_color in zip(network_names, network_colors):
        inds = [i for i, lbl in enumerate(labels) if net_name in lbl]
        if inds:
            i0, i1 = min(inds), max(inds)
            ax.add_patch(mpatches.Rectangle(
                (i0, n), i1-i0+1, barwidth, color=net_color, clip_on=False, lw=0))
            ax.add_patch(mpatches.Rectangle(
                (-barwidth, i0), barwidth, i1-i0+1, color=net_color, clip_on=False, lw=0))
    ax.set_xlim([-barwidth, n])
    ax.set_ylim([n + barwidth, -barwidth])
    ax.axis('off')

def plot_grid(
    matrix_type, ref_type, title,
    vmin, vmax, cbar_label, filename,
    example_subj=0
):
    """Plot a grid of matrices with colorbar and network bars."""
    
    # Find all result files
    files = sorted(glob.glob(os.path.join(simdir, 'modelFit_SCopt_*.npz')))
    nT, nM = len(SCthresh_list), len(lambda_list)
    
    # Create figure and axes grid; set font sizes
    figsize = (4 * nM, 3.5 * nT)
    fig, axes = plt.subplots(nT, nM, figsize=figsize, squeeze=False)
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 14})

    im = None  # For colorbar reference

    # Loop over thresholds (rows) and lambda values (columns)
    for iT, thresh in enumerate(SCthresh_list):
        for iL, lam in enumerate(lambda_list):
            ax = axes[iT, iL]
            
            # Determine model tag for this subplot
            if iL == 0:
                model_tag = f"SC_{thresh.upper()}"
            else:
                model_tag = f"Grad_fccov_{thresh.upper()}_{lam}"

            # Find corresponding simulation file
            sim_files = [f for f in files if model_tag in f]
            if not sim_files:
                ax.axis('off')  # Hide axis if data missing
                continue

            # Load matrix data for example subject
            data = np.load(sim_files[0], allow_pickle=True)
            mat = data[matrix_type][example_subj]
            ref_mat = data[ref_type][example_subj]

            # Apply H2N reordering if needed
            if H2N_ordering:
                mat = mat[H2N_ind, :][:, H2N_ind]
                ref_mat = ref_mat[H2N_ind, :][:, H2N_ind]

            # Compute lower-triangular Pearson correlation to reference
            corval = pearson_corr(mat, ref_mat)

            # Plot the matrix
            im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')

            # Add network bars if using H2N order
            if H2N_ordering:
                plot_network_bars(ax, labels, network_names, network_colors)

            # Set subplot title and row label
            ax.set_title(f"{model_tag}", fontsize=14)
            if iL == 0:
                ax.set_ylabel(f"{thresh.upper()}", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])

    # Adjust grid spacing to minimize vertical whitespace
    fig.subplots_adjust(
        left=0.05, right=0.84, top=0.89, bottom=0.18,
        wspace=0.15, hspace=0.18
    )

    # Add colorbar (shared for all matrices)
    cbar_ax = fig.add_axes([0.86, 0.23, 0.03, 0.58])
    plt.colorbar(im, cax=cbar_ax, label=cbar_label)

    # Add legend for Yeo networks if H2N ordering is used
    if H2N_ordering:
        handles = [
            mpatches.Patch(color=network_colors[i], label=network_names[i])
            for i in range(len(network_names))
        ]
        fig.legend(
            handles, network_names, loc='lower center', ncol=len(network_names),
            bbox_to_anchor=(0.5, 0.11), fontsize=14, frameon=False
        )

    # Add figure title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)

    # Save and display the figure
    plt.savefig(os.path.join(plotdir, filename), dpi=200, bbox_inches='tight')
    plt.show()

# ===========================
# Plot Grids
# ===========================

# GEC (vs SC)
plot_grid(
    matrix_type='GEC_sub',
    ref_type='SC_sub',
    title='Simulated GEC Matrices — Example Subject',
    vmin=0, vmax=2,
    cbar_label='Connection Strength',
    filename='GEC_matrices.png'
)

# Simulated FC
plot_grid(
    matrix_type='FCsim_sub',
    ref_type='FCemp_sub',
    title='Simulated FC Matrices — Example Subject',
    vmin=-1.2, vmax=1.2,
    cbar_label='FC',
    filename='FCsim_matrices.png'
)

# Simulated CovTau 
plot_grid(
    matrix_type='CovTauSim_sub',
    ref_type='CovTauEmp_sub',
    title='Simulated CovTau(τ) Matrices — Example Subject',
    vmin=-1.2, vmax=1.2,
    cbar_label='Cov(τ)',
    filename='CovTauSim_matrices.png'
)
