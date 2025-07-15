# =============================================================================
#    Generative Effective Connectivity (GEC) Estimation with pseudo-Gradients
#    -----------------------------------------------------------
#    For each subject:
#        - Loads subject's thresholded SC and frequency vector (f_diff)
#        - Loads subject's BOLD time series
#        - Computes empirical FC and Cov(τ) from the BOLD data
#        - Initializes GEC as SC
#        - Optimizes GEC with pseudo-gradient-based loss (FC + Cov(τ) + reg)
#        - Plots loss and saves results
# =============================================================================

# Import necessary libraries
import time
import os
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import optax
import numpy as np
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.signal import detrend, correlate


# Safeguard maximum threads for parallel libraries
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# Change to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create directories if not exist
simdir = 'simulations'
plotdir = 'plots/pseudoGrad_fccov'
os.makedirs(simdir, exist_ok=True)
os.makedirs(plotdir, exist_ok=True)

# ===========================
#       PARAMETERS
# ===========================

# Data parameters
TR = 1.26                # TR in seconds

# Hopf model parameters
sigma = 0.01             # Noise SD
tau = 2*TR               # Lag in seconds
maxC = float('Inf')      # Max connection strength
a = -0.02                # Hopf bifurcation parameter

# Optimization parameters
num_steps = 1000         # Number of optimization steps
epsFC = 4e-3             # FC loss weight
epsCovTau = 1e-3         # Cov(τ) loss weight
tol = 1e-4               # Relative tolerance for convergence
patience = 5             # Patience for early stopping

# Adjustable parameters
#lambdas = [5e-3, 1e-3, 1e-4, 0]
lambdas = [5e-3]
thresholds = ['t90', 't40', 't10']  

# ===========================
#       LOAD DATA
# ===========================

print("Loading data...")
f_diff = scipy.io.loadmat('empirical_mig_n2treat.mat')['f_diff'].flatten()
mat_ts = scipy.io.loadmat('mig_n2treat_func.mat')
ts_all = mat_ts['mig_n2treat_func']  # Each cell is (N, T) array

# ===========================
#     EMPIRICAL FC/COVTAU
# ===========================

def compute_covtau(ts, Tau):
    N, T = ts.shape
    tst = ts.T
    COV0 = np.cov(tst, rowvar=False)
    sigratio = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sigratio[i, j] = 1.0 / np.sqrt(COV0[i, i]) / np.sqrt(COV0[j, j])
    COVtau = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            clag = correlate(tst[:, i], tst[:, j], mode='full')
            lags = np.arange(-T + 1, T)
            lag_idx = np.where(lags == Tau)[0][0]
            COVtau[i, j] = clag[lag_idx] / T
    COVtau = COVtau * sigratio
    return COVtau

def compute_empirical_fc_covtau(ts, Tau):
    N, T = ts.shape
    ts_proc = np.zeros_like(ts)
    for i in range(N):
        ts_proc[i, :] = detrend(ts[i, :] - np.mean(ts[i, :]))
    FCemp = np.corrcoef(ts_proc)
    COVtauemp = compute_covtau(ts_proc, Tau)
    return FCemp, COVtauemp

# ===========================
#     MODEL FUNCTIONS
# ===========================

def get_jacobian(G, a, omega):
    """
    Construct the Jacobian matrix for the Hopf model.
    (All numpy version)
    """
    N = G.shape[0]
    S = np.sum(G, axis=1)
    diag_a_S = np.diag(a - S)
    J_xx = diag_a_S + G
    J_yy = diag_a_S + G
    J_xy = np.diag(omega)
    J_yx = -np.diag(omega)
    J_top = np.concatenate([J_xx, J_xy], axis=1)
    J_bottom = np.concatenate([J_yx, J_yy], axis=1)
    J = np.concatenate([J_top, J_bottom], axis=0)
    return J

def lyap_solve(A, Q):
    """
    Solve the continuous Lyapunov equation using SciPy.
    (No jax, only numpy/scipy)
    """
    P = scipy.linalg.solve_continuous_lyapunov(A, -Q)
    return P

def get_FC_sim(Sigma, N):
    """
    Compute simulated FC from covariance matrix (numpy version).
    """
    Sigma_xx = Sigma[:N, :N]
    epsilon = 1e-10
    stddev = np.sqrt(np.abs(np.diag(Sigma_xx))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    FC_sim = Sigma_xx / norm
    if np.any(np.isnan(norm)):
        print("NaN in norm in FC_sim or Cov_tau_sim normalization!")
    if np.any(norm == 0):
        print("Zero in norm in FC_sim or Cov_tau_sim normalization!")   
    return FC_sim

def get_Cov_tau_sim(J, Sigma, tau, N):
    """
    Compute simulated Cov(τ) from model (numpy version).
    """
    expJtau = scipy.linalg.expm(J * tau)
    Cov_full = expJtau @ Sigma
    Cov_tau_xx = Cov_full[:N, :N]
    Sigma_xx = Sigma[:N, :N]
    epsilon = 1e-10
    stddev = np.sqrt(np.abs(np.diag(Sigma_xx))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    Cov_tau_normed = Cov_tau_xx / norm
    if np.any(np.isnan(norm)):
        print("NaN in norm in FC_sim or Cov_tau_sim normalization!")
    if np.any(norm == 0):
        print("Zero in norm in FC_sim or Cov_tau_sim normalization!")
    return Cov_tau_normed

# ===========================
#     PLOTTING FUNCTIONS
# ===========================

def plot_results(G, SC, FC_emp, FC_sim, Cov_tau_emp, Cov_tau_sim, mse_total, plotdir, regstr, tstr):
    """
    Plot and save results for one subject.
    """
    # Consistent vmax
    nonzero_g = G[G > 0]
    nonzero_sc = SC[SC > 0]
    vmax = max(np.percentile(nonzero_g, 99), np.percentile(nonzero_sc, 99)) if (nonzero_g.size > 0 and nonzero_sc.size > 0) else 1.0
    # SC plot
    plt.figure(figsize=(6, 5))
    plt.imshow(SC, cmap='viridis', vmin=0, vmax=vmax)
    plt.title(f"Initial SC{regstr}{tstr}")
    plt.colorbar(label='Connection strength')
    plt.tight_layout()
    plt.savefig(f"{plotdir}/SC_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()
    # GEC plot
    plt.figure(figsize=(6, 5))
    plt.imshow(G, cmap='viridis', vmin=0, vmax=vmax)
    plt.title(f"GEC{regstr}{tstr}")
    plt.colorbar(label='Connection strength')
    plt.tight_layout()
    plt.savefig(f"{plotdir}/GEC_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()
    # Empirical FC
    plt.figure(figsize=(6, 5))
    plt.imshow(FC_emp, cmap='viridis', vmin=-1, vmax=1)
    plt.title(f"Empirical FC{regstr}{tstr}")
    plt.colorbar(label='Correlation')
    plt.tight_layout()
    plt.savefig(f"{plotdir}/FCemp_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()
    # Simulated FC
    plt.figure(figsize=(6, 5))
    plt.imshow(FC_sim, cmap='viridis', vmin=-1, vmax=1)
    plt.title(f"Simulated FC{regstr}{tstr}")
    plt.colorbar(label='Correlation')
    plt.tight_layout()
    plt.savefig(f"{plotdir}/FCsim_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()
    # Empirical CovTau
    plt.figure(figsize=(6, 5))
    plt.imshow(Cov_tau_emp, cmap='viridis')
    plt.title(f"Empirical CovTau{regstr}{tstr}")
    plt.colorbar(label='Cov(τ)')
    plt.tight_layout()
    plt.savefig(f"{plotdir}/CovTauEmp_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()
    # Simulated CovTau
    plt.figure(figsize=(6, 5))
    plt.imshow(Cov_tau_sim, cmap='viridis')
    plt.title(f"Simulated CovTau{regstr}{tstr}")
    plt.colorbar(label='Cov(τ)')
    plt.tight_layout()
    plt.savefig(f"{plotdir}/CovTauSim_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()
    # MSE plot
    plt.figure()
    plt.plot(mse_total)
    plt.xlabel("Iteration")
    plt.ylabel("Total MSE")
    plt.title(f"MSE{regstr}{tstr}")
    plt.tight_layout()
    plt.savefig(f"{plotdir}/MSE_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()

# ===============================
#   GEC ESTIMATION PER SUBJECT
# ===============================

def estimate_GEC_per_subject_pseudograd(
    subj_idx, SC_all, ts_all, f_diff, maxC, tau, TR, sigma,
    lambda_reg, epsFC, epsCovTau, num_steps, tol, patience
):
    start_time = time.time()
    SC = np.array(SC_all[:, :, subj_idx])
    sc_max = SC.max()
    if np.isfinite(maxC) and sc_max > 0:
        SC = SC / sc_max * maxC
    N = SC.shape[0]

    # Update mask: nonzero SC or homologous
    homologous = np.zeros((N, N), dtype=bool)
    half = N // 2
    for i in range(N):
        for j in range(N):
            if (i < half and j == i + half) or (j < half and i == j + half):
                homologous[i, j] = True
    mask = (SC > 0) | homologous

    ts = ts_all[0, subj_idx] if ts_all.shape[0] == 1 else ts_all[subj_idx, 0]
    ts = np.array(ts)
    Tau_in_samples = int(np.round(tau / TR))
    FCemp, CovTauEmp = compute_empirical_fc_covtau(ts, Tau_in_samples)

    C = SC.copy()
    mse_total, mse_fc_hist, mse_cov_hist = [], [], []
    old_best = np.inf
    patience_counter = 0

    omega = f_diff[:N] * (2 * np.pi)
    a_arr = -0.02 * np.ones(N)
    Q = sigma**2 * np.eye(2*N)

    for step in range(num_steps):
        # Model Jacobian & simulation
        S = np.sum(C, axis=1)
        diag_a_S = np.diag(a_arr - S)
        J_xx = diag_a_S + C
        J_yy = diag_a_S + C
        J_xy = np.diag(omega)
        J_yx = -np.diag(omega)
        J_top = np.concatenate([J_xx, J_xy], axis=1)
        J_bottom = np.concatenate([J_yx, J_yy], axis=1)
        J = np.concatenate([J_top, J_bottom], axis=0)
        Sigma = scipy.linalg.solve_continuous_lyapunov(J, -Q)
        Sigma_xx = Sigma[:N, :N]
        stddev = np.sqrt(np.abs(np.diag(Sigma_xx))) + 1e-10
        FCsim = Sigma_xx / (stddev[:, None] * stddev[None, :])
        expJtau = scipy.linalg.expm(J * tau)
        Cov_tau_sim = expJtau @ Sigma
        Cov_tau_sim = Cov_tau_sim[:N, :N]
        Cov_tau_sim = Cov_tau_sim / (stddev[:, None] * stddev[None, :])

        mse_fc = np.mean((FCemp - FCsim)**2)
        mse_cov = np.mean((CovTauEmp - Cov_tau_sim)**2)
        mse = mse_fc + mse_cov + lambda_reg * np.mean((C - SC)**2)
        mse_total.append(mse)
        mse_fc_hist.append(mse_fc)
        mse_cov_hist.append(mse_cov)

        # Early stopping 
        if mse < old_best - tol:
            old_best = mse
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            break

        # Pseudo-gradient update
        for i in range(N):
            for j in range(N):
                if not mask[i, j]:
                    continue
                delta = epsFC * (FCemp[i, j] - FCsim[i, j]) - lambda_reg * (C[i, j] - SC[i, j])
                delta += epsCovTau * (CovTauEmp[i, j] - Cov_tau_sim[i, j])
                C[i, j] += delta
                C[i, j] = max(0, C[i, j])
        # Rescale to maxC
        Cmax = C.max()
        if np.isfinite(maxC) and sc_max > 0:
            C = C / Cmax * maxC

    runtime_sec = time.time() - start_time

    return {
        "FC_emp": FCemp,
        "Cov_tau_emp": CovTauEmp,
        "FC_sim": FCsim,
        "Cov_tau_sim": Cov_tau_sim,
        "SC": SC,
        "GEC": C,
        "mse_total": np.array(mse_total),
        "mse_fc": np.array(mse_fc_hist),
        "mse_cov": np.array(mse_cov_hist),
        "runtime_sec": runtime_sec,
        "max_iter": step+1
    }

# ===============================
#        MAIN PIPELINE
# ===============================

if __name__ == "__main__":
    for SCthresh in thresholds:
        sc_all = scipy.io.loadmat(f'mig_n2treat_SC_{SCthresh}.mat')['SC_all']
        N, _, NSUB = sc_all.shape
        for lambda_reg in lambdas:
            thresh_tag = SCthresh.upper()
            lambda_tag = f"λ{lambda_reg:g}"
            model_tag = f"pseudoGrad_fccov_{thresh_tag}_{lambda_tag}"
            sim_filename = f"modelFit_SCopt_{model_tag}.npz"
            sim_path = os.path.join(simdir, sim_filename)
            print(f"\n==== Running: {model_tag} ====")

            FCemp_sub, CovTauEmp_sub = [], []
            FCsim_sub, CovTauSim_sub = [], []
            SC_sub, GEC_sub = [], []
            Loss_iter_sub, MaxIter_sub = [], []
            MSE_iter_sub = []
            Runtime_sub = []

            for subj in range(NSUB): 
                out = estimate_GEC_per_subject_pseudograd(
                    subj, sc_all, ts_all, f_diff, maxC, tau, TR, sigma,
                    lambda_reg, epsFC, epsCovTau, num_steps, tol, patience
                )
                FCemp_sub.append(out['FC_emp'])
                CovTauEmp_sub.append(out['Cov_tau_emp'])
                FCsim_sub.append(out['FC_sim'])
                CovTauSim_sub.append(out['Cov_tau_sim'])
                SC_sub.append(out['SC'])
                GEC_sub.append(out['GEC'])
                Loss_iter_sub.append(out['mse_total'])
                MSE_iter_sub.append(out['mse_total'])
                MaxIter_sub.append(out['max_iter'])
                Runtime_sub.append(out['runtime_sec'])

                if subj == 0:
                    regstr = f"_λ{lambda_reg:g}"
                    tstr = f"_thr{SCthresh[1:]}"
                    plot_results(
                        out['GEC'], out['SC'], out['FC_emp'], out['FC_sim'],
                        out['Cov_tau_emp'], out['Cov_tau_sim'], out['mse_total'],
                        plotdir, regstr, tstr
                    )
            np.savez_compressed(
                sim_path,
                FCemp_sub=np.array(FCemp_sub),
                CovTauEmp_sub=np.array(CovTauEmp_sub),
                FCsim_sub=np.array(FCsim_sub),
                CovTauSim_sub=np.array(CovTauSim_sub),
                SC_sub=np.array(SC_sub),
                GEC_sub=np.array(GEC_sub),
                Loss_iter_sub=np.array(Loss_iter_sub, dtype=object),
                MSE_iter_sub=np.array(MSE_iter_sub, dtype=object),
                MaxIter_sub=np.array(MaxIter_sub),
                maxIter=num_steps,
                Runtime_sub=np.array(Runtime_sub),
                maxC=maxC,
                TR=TR,
                tau=tau,
                tag=model_tag
            )
            print(f"Saved simulation results to {sim_path}")

# =============================================================================
#    END OF SCRIPT
# =============================================================================