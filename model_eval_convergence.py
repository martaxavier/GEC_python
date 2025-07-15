import os
import numpy as np
import matplotlib.pyplot as plt

# ==== PARAMETERS ====
# modeltype = "Grad_fccov"    
# lambda_list = [1, 1e-1, 0]

modeltype = "pseudoGrad_fccov"     
lambda_list = [1e-3, 1e-4, 0]

SCthresh_list = ['t10', 't40', 't90']

nThresh = len(SCthresh_list)
nLambda = len(lambda_list)
simdir = "simulations"
plotdir = "plots"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(plotdir, exist_ok=True)

# === FIGURE ===
fig, axes = plt.subplots(nThresh, nLambda, figsize=(14, 9), squeeze=False)
plt.subplots_adjust(hspace=0.3, wspace=0.25)
fig.suptitle("Convergence Curves: Total MSE = MSE$_{FC}$ + MSE$_{COV(\\tau)}$", fontsize=16, fontweight='bold')

for iT, SCthresh in enumerate(SCthresh_list):
    for iL, lam in enumerate(lambda_list):
        thresh_tag = SCthresh.upper()
        lambda_tag = f"λ{lam:g}"
        model_tag = f"{modeltype}_{thresh_tag}_{lambda_tag}"
        sim_filename = f"modelFit_SCopt_{model_tag}.npz"
        sim_path = os.path.join(simdir, sim_filename)
        
        ax = axes[iT, iL]
        
        if not os.path.exists(sim_path):
            ax.set_visible(False)
            continue

        # Load .npz
        dat = np.load(sim_path, allow_pickle=True)
        mse_iter_sub = dat['MSE_iter_sub']
        runtime_sub = dat['Runtime_sub']
        
        # Get MSE for each subject
        mse_iter_sub = dat['MSE_iter_sub']

        # Convert all to 1D arrays
        mse_arrs = [np.array(mse, dtype=float).flatten() for mse in mse_iter_sub]
        min_len = min(len(arr) for arr in mse_arrs)  # min number of iterations completed by all subjects

        # Truncate all to min_len
        mse_mat = np.vstack([arr[:min_len] for arr in mse_arrs])

        # Plot mean and std up to min_len
        mean_mse = np.nanmean(mse_mat, axis=0)
        std_mse = np.nanstd(mse_mat, axis=0)

        ax.plot(mean_mse, color=[0.85, 0.33, 0.1], linewidth=1.5)
        ax.fill_between(np.arange(min_len), mean_mse-std_mse, mean_mse+std_mse, color=[0.85, 0.33, 0.1], alpha=0.18)
        ax.set_xlim([0, min_len-1])
        ax.grid(False)
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-2, top=np.nanmax(mean_mse+std_mse)*1.1)  # dynamic top

        # Title/labels
        ax.set_title(f"{thresh_tag}, λ = {lam:.0e}", fontsize=11)
        if iT == nThresh-1:
            ax.set_xlabel("Iteration", fontsize=11)
        if iL == 0:
            ax.set_ylabel("Total Error", fontsize=11)
        
        # Add mean runtime annotation in bottom-right
        mean_runtime = np.mean(runtime_sub)
        ax.text(len(mean_mse)*0.95, np.nanmin(mean_mse+std_mse)*1.05, f"{mean_runtime:.2f}s",
                fontsize=11, color=(0, 0.447, 0.741),
                va='bottom', ha='right', fontweight='bold')

plt.savefig(os.path.join(plotdir, f"Convergence_curves_{modeltype}.png"), dpi=120, bbox_inches='tight')
plt.show()

