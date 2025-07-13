# =============================================================================
#    Generative Effective Connectivity (GEC) Estimation with JAX
#    -----------------------------------------------------------
#    For each subject:
#        - Loads subject's thresholded SC and frequency vector (f_diff)
#        - Loads BOLD time series
#        - Computes empirical FC and Cov(τ) from the BOLD data
#        - Initializes GEC as SC
#        - Optimizes GEC with gradient-based true loss (FC + Cov(τ) + reg)
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
plotdir = 'plots/Grad_fccov'
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
learning_rate = 0.01     # Adam learning rate
num_steps = 500          # Number of optimization steps
alpha = 1.0              # FC loss weight
beta = 0.25              # Cov(τ) loss weight
tol = 5e-3               # Relative tolerance for convergence
patience = 5             # Patience for early stopping

# Adjustable parameters
lambdas = [1, 5e-1, 1e-1, 0]
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
    """
    Compute cross-covariance at lag Tau, normalized by zero-lag covariance.
    """
    N, T = ts.shape
    tst = ts.T  # (T, N)

    # Empirical zero-lag covariance for normalization
    COV0 = np.cov(tst, rowvar=False)

    # Normalization matrix
    sigratio = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            sigratio[i, j] = 1.0 / np.sqrt(COV0[i, i]) / np.sqrt(COV0[j, j])

    # Cross-covariance at lag Tau
    COVtau = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            clag = correlate(tst[:, i], tst[:, j], mode='full')
            lags = np.arange(-T + 1, T)
            lag_idx = np.where(lags == Tau)[0][0]
            COVtau[i, j] = clag[lag_idx] / T

    # Apply normalization
    COVtau = COVtau * sigratio
    return COVtau

def compute_empirical_fc_covtau(ts, Tau):
    """
    Compute empirical FC and Cov(τ) from time series.
    """
    N, T = ts.shape
    # Demean and detrend
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
    """
    N = G.shape[0]
    S = jnp.sum(G, axis=1)
    diag_a_S = jnp.diag(a - S)
    J_xx = diag_a_S + G
    J_yy = diag_a_S + G
    J_xy = jnp.diag(omega)
    J_yx = -jnp.diag(omega)
    J_top = jnp.concatenate([J_xx, J_xy], axis=1)
    J_bottom = jnp.concatenate([J_yx, J_yy], axis=1)
    J = jnp.concatenate([J_top, J_bottom], axis=0)
    return J

# ---- Custom VJP Lyapunov solver ----
def lyap_solve(A, Q):
    """
    Solve the continuous Lyapunov equation using SciPy.
    """
    # Convert from JAX array to NumPy array (even if tracer)
    A_np = np.array(jax.device_get(A))
    Q_np = np.array(jax.device_get(Q))
    P = scipy.linalg.solve_continuous_lyapunov(A_np, -Q_np)
    return jnp.array(P)

def lyap_solve_fwd(A, Q):
    """
    Forward pass for custom VJP Lyapunov solver.
    """
    P = lyap_solve(A, Q)
    # Save all needed for backward pass
    return P, (A, Q, P)

def lyap_solve_bwd(res, dP):
    """
    Backward pass for custom VJP Lyapunov solver.
    """
    A, Q, P = res
    A_np = np.array(jax.device_get(A))
    dP_np = np.array(jax.device_get(dP))

    if np.any(np.isnan(A_np)) or np.any(np.isnan(dP_np)) or np.any(np.isinf(A_np)) or np.any(np.isinf(dP_np)):
        raise ValueError("Backward pass received NaN or Inf in A or dP!")
    
    # Solve the adjoint Lyapunov equation: S A + A^T S = -dP
    S = scipy.linalg.solve_continuous_lyapunov(A_np.T, -dP_np)
    S = jnp.array(S)
    dA = S @ P.T + S.T @ P
    dQ = S
    return (dA, dQ)

# Register with JAX
lyap_solve_jax = jax.custom_vjp(lyap_solve)
lyap_solve_jax.defvjp(lyap_solve_fwd, lyap_solve_bwd)
# ---- end Lyapunov solver ----

def get_FC_sim(Sigma, N):
    """
    Compute simulated FC from covariance matrix.
    """
    Sigma_xx = Sigma[:N, :N]
    epsilon = 1e-10
    stddev = jnp.sqrt(jnp.abs(jnp.diag(Sigma_xx))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    FC_sim = Sigma_xx / norm
    if jnp.any(jnp.isnan(norm)):
        print("NaN in norm in FC_sim or Cov_tau_sim normalization!")
    if jnp.any(norm == 0):
        print("Zero in norm in FC_sim or Cov_tau_sim normalization!")   
    return FC_sim

def get_Cov_tau_sim(J, Sigma, tau, N):
    """
    Compute simulated Cov(τ) from model.
    """
    expJtau = jax.scipy.linalg.expm(J * tau)
    Cov_full = expJtau @ Sigma
    Cov_tau_xx = Cov_full[:N, :N]
    Sigma_xx = Sigma[:N, :N]
    epsilon = 1e-10
    stddev = jnp.sqrt(jnp.abs(jnp.diag(Sigma_xx))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    Cov_tau_normed = Cov_tau_xx / norm
    if jnp.any(jnp.isnan(norm)):
        print("NaN in norm in FC_sim or Cov_tau_sim normalization!")
    if jnp.any(norm == 0):
        print("Zero in norm in FC_sim or Cov_tau_sim normalization!")
    return Cov_tau_normed

def loss_fn(G, a, omega, Q, tau, FC_emp, Cov_tau_emp, SC, alpha, beta, lambd):
    """
    Loss function for GEC optimization.
    """
    N = G.shape[0]
    J = get_jacobian(G, a, omega)
    Sigma = lyap_solve_jax(J, Q)
    FC_sim = get_FC_sim(Sigma, N)
    Cov_tau_sim = get_Cov_tau_sim(J, Sigma, tau, N)
    loss_FC = jnp.sum((FC_emp - FC_sim) ** 2)
    loss_Cov = jnp.sum((Cov_tau_emp - Cov_tau_sim) ** 2)
    loss_reg = jnp.sum((G - SC) ** 2)
    total_loss = alpha * loss_FC + beta * loss_Cov + lambd * loss_reg

    if jnp.any(jnp.isnan(FC_sim)):
        print("NaN in FC_sim!")
    if jnp.any(jnp.isnan(Cov_tau_sim)):
        print("NaN in Cov_tau_sim!")
    if jnp.any(jnp.isnan(Sigma)):
        print("NaN in Sigma!")
    if jnp.isnan(total_loss):
        print("NaN in total_loss!")
 
    return total_loss

# ===========================
#     PLOTTING FUNCTIONS
# ===========================

def plot_results(G, SC, FC_emp, FC_sim, Cov_tau_emp, Cov_tau_sim, losses, plotdir, regstr, tstr):
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
    # Loss plot
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss{regstr}{tstr}")
    plt.tight_layout()
    plt.savefig(f"{plotdir}/Loss_subject1{regstr}{tstr}.png", dpi=150)
    plt.close()

# ===============================
#   GEC ESTIMATION PER SUBJECT
# ===============================

def estimate_GEC_per_subject(
    subj_idx, SC_all, ts_all, f_diff, maxC, tau, TR, sigma,
    learning_rate, num_steps, alpha, beta, lambda_reg, tol, patience
):
    """
    Estimate GEC for a single subject using gradient-based optimization.

    Args:
        subj_idx:    Index of the subject.
        SC_all:      Structural connectivity array, shape (N, N, NSUB).
        ts_all:      List of BOLD time series, each (N, T).
        f_diff:      Frequency differences, (N,).
        maxC:        Maximum connection strength for normalization.
        tau:         Covariance lag (seconds).
        TR:          Repetition time (seconds).
        sigma:       Noise SD.
        learning_rate, num_steps, alpha, beta, lambda_reg, tol, patience: Optimization hyperparameters.

    Returns:
        Dictionary of results for this subject.
    """
    start_time = time.time()

    # ---- 1. Load and normalize SC ----
    SC = np.array(SC_all[:, :, subj_idx])
    sc_max = SC.max()
    if np.isfinite(maxC) and sc_max > 0:
        SC = SC / sc_max * maxC

    N = SC.shape[0]

    # ---- 2. Build update mask (SC > 0 or homologous regions) ----
    homologous = np.zeros((N, N), dtype=bool)
    half = N // 2
    for i in range(N):
        for j in range(N):
            if (i < half and j == i + half) or (j < half and i == j + half):
                homologous[i, j] = True
    sc_mask = (SC > 0)
    mask = np.logical_or(sc_mask, homologous).astype(np.float32)
    mask = jnp.array(mask)
    SC_jax = jnp.array(SC)

    # ---- 3. Prepare subject-specific data and model parameters ----
    # Load time series for subject
    ts = ts_all[0, subj_idx] if ts_all.shape[0] == 1 else ts_all[subj_idx, 0]
    ts = np.array(ts)

    # Empirical FC and Cov(tau)
    Tau_in_samples = int(np.round(tau / TR))
    FC_emp, Cov_tau_emp = compute_empirical_fc_covtau(ts, Tau_in_samples)

    # Model parameters for this subject
    omega = f_diff[:N] * (2 * np.pi)
    Q = sigma**2 * np.eye(2*N)
    a_jax = jnp.array(a * np.ones(N))
    omega_jax = jnp.array(omega)
    Q_jax = jnp.array(Q)
    FC_emp_jax = jnp.array(FC_emp)
    Cov_tau_emp_jax = jnp.array(Cov_tau_emp)

    # ---- 4. Initialize optimizer and variables ----
    G_init = SC_jax.copy()
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(G_init)

    # ---- 5. Optimization loop ----
    def step(G, opt_state):
        """
        Perform one Adam optimization step.
        """
        loss, grad = jax.value_and_grad(loss_fn)(
            G, a_jax, omega_jax, Q_jax, tau,
            FC_emp_jax, Cov_tau_emp_jax, SC_jax,
            alpha, beta, lambda_reg
        )
        # Only update allowed entries
        grad = grad * mask
        updates, opt_state = optimizer.update(grad, opt_state)
        G = optax.apply_updates(G, updates)
        # Project to feasible set (mask, non-negative)
        G = G * mask
        G = jnp.clip(G, 0, None)
        return G, opt_state, float(loss)

    G = G_init
    losses = []
    mse_total = []
    patience_counter = 0
    best_loss = float('inf')

    for i in range(num_steps):
        G, opt_state, loss = step(G, opt_state)
        losses.append(loss)

        # Compute current model FC and CovTau
        J = get_jacobian(G, a_jax, omega_jax)
        Sigma = lyap_solve_jax(J, Q_jax)
        FC_sim = get_FC_sim(Sigma, N)
        Cov_tau_sim = get_Cov_tau_sim(J, Sigma, tau, N)
        mse_fc = np.mean((np.array(FC_emp) - np.array(FC_sim))**2)
        mse_covtau = np.mean((np.array(Cov_tau_emp) - np.array(Cov_tau_sim))**2)
        mse_total.append(mse_fc + mse_covtau)

        # Early stopping: reset patience on improvement
        if loss < best_loss * (1 - tol):
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter > patience:
            break

    runtime_sec = time.time() - start_time

    # ---- 6. Final forward simulation using optimized G ----
    J_final = get_jacobian(G, a_jax, omega_jax)
    Sigma_final = np.array(lyap_solve_jax(J_final, Q_jax))
    FC_sim_final = np.corrcoef(Sigma_final[:N, :N])

    # Lagged covariance and normalization
    expJtau = scipy.linalg.expm(np.array(J_final) * tau)
    Cov_tau_sim = expJtau @ Sigma_final
    Cov_tau_sim = Cov_tau_sim[:N, :N]
    epsilon = 1e-10
    stddev = np.sqrt(np.abs(np.diag(Sigma_final[:N, :N]))) + epsilon
    norm = stddev[:, None] * stddev[None, :]
    Cov_tau_sim_final = Cov_tau_sim / norm

    # ---- 7. Collect results ----
    return {
        "FC_emp": FC_emp,
        "Cov_tau_emp": Cov_tau_emp,
        "FC_sim": FC_sim_final,
        "Cov_tau_sim": Cov_tau_sim_final,
        "SC": SC,
        "GEC": np.array(G),
        "losses": np.array(losses),
        "mse_total": np.array(mse_total),
        "max_iter": i+1,
        "runtime_sec": runtime_sec
    }

# ===============================
#        MAIN PIPELINE
# ===============================

if __name__ == "__main__":

    # Loop over all thresholds (e.g. t10, t40, t90)
    for SCthresh in thresholds:

        # ---- Load structural connectivity for this threshold ----
        sc_all = scipy.io.loadmat(f'mig_n2treat_SC_{SCthresh}.mat')['SC_all']
        N, _, NSUB = sc_all.shape

        # Loop over all regularization values (lambdas)
        for lambda_reg in lambdas:

            # ---- Prepare tags and output filenames ----
            thresh_tag = SCthresh.upper()
            lambda_tag = f"λ{lambda_reg:g}"
            model_tag = f"Grad_fccov_{thresh_tag}_{lambda_tag}"
            sim_filename = f"modelFit_SCopt_{model_tag}.npz"
            sim_path = os.path.join(simdir, sim_filename)
            print(f"\n==== Running: {model_tag} ====")

            # ---- Storage for results across all subjects ----
            FCemp_sub, CovTauEmp_sub = [], []
            FCsim_sub, CovTauSim_sub = [], []
            SC_sub, GEC_sub = [], []
            Loss_iter_sub, MaxIter_sub = [], []
            MSE_iter_sub = []
            Runtime_sub = []

            # ---- Loop over all subjects ----
            for subj in range(NSUB): # Change to range(NSUB) for all subjects

                out = estimate_GEC_per_subject(
                    subj, sc_all, ts_all, f_diff, maxC, tau, TR, sigma,
                    learning_rate, num_steps, alpha, beta, lambda_reg, tol, patience
                )

                # Collect all outputs
                FCemp_sub.append(out['FC_emp'])
                CovTauEmp_sub.append(out['Cov_tau_emp'])
                FCsim_sub.append(out['FC_sim'])
                CovTauSim_sub.append(out['Cov_tau_sim'])
                SC_sub.append(out['SC'])
                GEC_sub.append(out['GEC'])
                Loss_iter_sub.append(out['losses'])
                MSE_iter_sub.append(out['mse_total'])
                MaxIter_sub.append(out['max_iter'])
                Runtime_sub.append(out['runtime_sec'])

                # ---- Plot only for subject 1 ----
                if subj == 0:
                    regstr = f"_λ{lambda_reg:g}"
                    tstr = f"_thr{SCthresh[1:]}"
                    plot_results(
                        out['GEC'], out['SC'], out['FC_emp'], out['FC_sim'],
                        out['Cov_tau_emp'], out['Cov_tau_sim'], out['losses'],
                        plotdir, regstr, tstr
                    )

            # ---- Save all simulation results for this (threshold, lambda) ----
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
