# ======================================================================
#                           Fixed-SC Model
#    -----------------------------------------------------------
#
#   This script simulates the fixed structural connectivity (SC) model
#   and computes empirical and simulated functional connectivity (FC) 
#   and covariance at a given time lag.
# =======================================================================

# Import necessary libraries
import os
import numpy as np
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.signal import detrend, correlate
import time

# Safeguard maximum threads for parallel libraries
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# Change to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create directories if not exist
simdir = 'simulations'
plotdir = 'plots/FixedSC'
os.makedirs(simdir, exist_ok=True)
os.makedirs(plotdir, exist_ok=True)

# ===========================
#       PARAMETERS
# ===========================

TR = 1.26
sigma = 0.01
tau = 2 * TR
maxC = 0.2 # float('Inf') # Use float('Inf') for no maxC scaling
thresholds = ['t10', 't40', 't90']

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
    """
    Compute the covariance at a given time lag Tau for time series ts.
    """
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
    """    
    Compute empirical functional connectivity (FC) 
    and covariance at a given time lag Tau.
    """
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
    Compute the Jacobian matrix for the model.
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
    Solve the continuous Lyapunov equation A * P + P * A^T = -Q.
    """
    return scipy.linalg.solve_continuous_lyapunov(A, -Q)

def get_FC_sim(Sigma, N):
    """ 
    Compute the simulated functional connectivity (FC) 
    from the covariance matrix Sigma.
    """
    Sigma_xx = Sigma[:N, :N]
    epsilon = 1e-10
    stddev = np.sqrt(np.abs(np.diag(Sigma_xx))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    FC_sim = Sigma_xx / norm
    return FC_sim

def get_Cov_tau_sim(J, Sigma, tau, N):
    """
    Compute the covariance at a given time lag tau
    from the Jacobian J and covariance matrix Sigma."""
    expJtau = scipy.linalg.expm(J * tau)
    Cov_full = expJtau @ Sigma
    Cov_tau_xx = Cov_full[:N, :N]
    Sigma_xx = Sigma[:N, :N]
    epsilon = 1e-10
    stddev = np.sqrt(np.abs(np.diag(Sigma_xx))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    Cov_tau_normed = Cov_tau_xx / norm
    return Cov_tau_normed

# ===========================
#     PLOTTING FUNCTIONS
# ===========================

def plot_results(G, SC, FC_emp, FC_sim, Cov_tau_emp, Cov_tau_sim, plotdir, regstr, tstr):
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

# ===============================
#        MAIN PIPELINE
# ===============================

for SCthresh in thresholds:

    # Load structural connectivity data for the current threshold
    print(f"\n--- Fixed-SC: {SCthresh.upper()} ---")
    sc_all = scipy.io.loadmat(f'mig_n2treat_SC_{SCthresh}.mat')['SC_all']
    N, _, NSUB = sc_all.shape

    FCemp_sub, CovTauEmp_sub = [], []
    FCsim_sub, CovTauSim_sub = [], []
    SC_sub, GEC_sub = [], []
    MaxIter_sub = []
    Runtime_sub = []

    # Run the model for each subject
    for subj in range(NSUB):
        start_time = time.time()
        SC = np.array(sc_all[:, :, subj])
        sc_max = SC.max()
        if np.isfinite(maxC) and sc_max > 0:
            SC = SC / sc_max * maxC
        ts = ts_all[0, subj] if ts_all.shape[0] == 1 else ts_all[subj, 0]
        ts = np.array(ts)
        Tau_in_samples = int(np.round(tau / TR))

        # Empirical
        FC_emp, Cov_tau_emp = compute_empirical_fc_covtau(ts, Tau_in_samples)

        # Model simulation (no optimization, GEC = SC)
        N = SC.shape[0]
        omega = f_diff[:N] * (2 * np.pi)
        a_arr = -0.02 * np.ones(N)
        Q = sigma ** 2 * np.eye(2 * N)

        G = SC.copy()  # GEC = SC for fixed-SC

        J = get_jacobian(G, a_arr, omega)
        Sigma = lyap_solve(J, Q)
        FC_sim = get_FC_sim(Sigma, N)
        Cov_tau_sim = get_Cov_tau_sim(J, Sigma, tau, N)

        FCemp_sub.append(FC_emp)
        CovTauEmp_sub.append(Cov_tau_emp)
        FCsim_sub.append(FC_sim)
        CovTauSim_sub.append(Cov_tau_sim)
        SC_sub.append(SC)
        GEC_sub.append(G)
        MaxIter_sub.append(1)
        Runtime_sub.append(time.time() - start_time)

        # ---- Plot only for subject 1 ----
        if subj == 0:
            regstr = ""  # No lambda_reg in fixed-SC, so leave empty 
            tstr = f"_thr{SCthresh[1:]}"
            plot_results(
                G, SC, FC_emp, FC_sim,
                Cov_tau_emp, Cov_tau_sim,
                plotdir, regstr, tstr
            )       

    model_tag = f"SC_{SCthresh.upper()}"
    sim_filename = f"modelFit_SCopt_{model_tag}.npz"
    sim_path = os.path.join(simdir, sim_filename)

    np.savez_compressed(
        sim_path,
        FCemp_sub=np.array(FCemp_sub),
        CovTauEmp_sub=np.array(CovTauEmp_sub),
        FCsim_sub=np.array(FCsim_sub),
        CovTauSim_sub=np.array(CovTauSim_sub),
        SC_sub=np.array(SC_sub),
        GEC_sub=np.array(GEC_sub),
        MaxIter_sub=np.array(MaxIter_sub),
        Runtime_sub=np.array(Runtime_sub),
        tag=model_tag,
        maxC=maxC,
        TR=TR,
        tau=tau,
    )
    print(f"Saved SC results to {sim_path}")

print("\nAll fixed-SC models complete.")
