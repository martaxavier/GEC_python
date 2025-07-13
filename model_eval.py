import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm

# ============  SETUP  =============
os.chdir(os.path.dirname(os.path.abspath(__file__)))

simdir = 'simulations'
plotdir = 'plots'

# ============  METRIC FUNCTIONS  =============

def pearson_corr(A, B):
    """Returns Pearson correlation between flattened A and B."""
    A, B = np.asarray(A).flatten(), np.asarray(B).flatten()
    return np.corrcoef(A, B)[0, 1]

# def cos_sim(A, B):
#     """Cosine similarity."""
#     A, B = np.asarray(A).flatten(), np.asarray(B).flatten()
#     return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# ==========  COLORS & MARKERS  ===========

def parse_tag(tag):
    tag = str(tag)
    splits = tag.split('_')
    modeltype = splits[0]
    thresh = next((s for s in splits if s.startswith('T')), '')
    lambdapart = [s for s in splits if s.startswith('λ')]
    lambdastr = lambdapart[0] if lambdapart else ''
    return modeltype, thresh, lambdastr

def assign_colors_markers(tags):
    model_types = sorted({m for m, t, l in tags})
    thresholds = sorted({t for m, t, l in tags})
    lambdas = sorted({l for m, t, l in tags if l}, key=lambda x: float(x[1:]) if x else 0, reverse=True)

    # Use a "darker" end of colormaps for more vivid points:
    def crop_cmap(cmap, vmin=0.2, vmax=0.8):
        return lambda x: cmap(vmin + x*(vmax-vmin))

    base_cmaps = {
        'T10': crop_cmap(cm.Blues),
        'T40': crop_cmap(cm.Greens),
        'T90': crop_cmap(cm.Oranges),
    }
    marker_map = {'SC': 'o', 'pseudoGrad_fccov': 's', 'Grad_fccov': '^'}
    return model_types, thresholds, lambdas, base_cmaps, marker_map

def get_color_marker(thresh, lambda_idx, model_type, base_cmaps, marker_map, lambdas):
    cmap = base_cmaps.get(thresh, lambda x: (0.5, 0.5, 0.5, 1))
    # Avoid pure white: only use middle of color spectrum
    if lambda_idx is not None and lambda_idx >= 0 and len(lambdas) > 1:
        color = cmap(lambda_idx / (len(lambdas)-1))
    else:
        color = cmap(0.0)
    marker = marker_map.get(model_type, 'x')
    return marker, color

def crop_cmap(cmap, vmin=0.35, vmax=0.8):
    return lambda x: cmap(vmin + x * (vmax - vmin))

def lambda_from_label(label):
    import re
    # Looks for _λ1, _λ0.1, etc
    match = re.search(r'_λ([0-9.]+)', label)
    if match:
        return float(match.group(1))
    if label.endswith('_λ0'):
        return 0.0
    return float('inf')  # SC etc.

def legend_sort_key(label):
    # Extract threshold and lambda
    # E.g. Grad_T90_λ1 -> ('T90', 1)
    parts = label.split('_')
    thresh = [s for s in parts if s.startswith('T')][0] if any(s.startswith('T') for s in parts) else ''
    lam = lambda_from_label(label)
    return (thresh, -lam)  # negative for descending

# ===========  LOAD AND EVALUATE  ===========

N = 100
Isubdiag = np.tril_indices(N, -1)
files = sorted(glob.glob(os.path.join(simdir, 'modelFit_SCopt_*.npz')))

tags_all = []
for f in files:
    data = np.load(f, allow_pickle=True)
    tag = data['tag'].item() if hasattr(data['tag'], 'item') else str(data['tag'])
    tags_all.append(parse_tag(tag))

model_types, thresholds, lambdas, base_cmaps, marker_map = assign_colors_markers(tags_all)

results = []
for f in files:
    data = np.load(f, allow_pickle=True)
    tag = data['tag'].item() if hasattr(data['tag'], 'item') else str(data['tag'])
    model_type, thresh, lambdastr = parse_tag(tag)
    lambda_idx = lambdas.index(lambdastr) if lambdastr else None

    FCemp_sub = data['FCemp_sub']
    FCsim_sub = data['FCsim_sub']
    CovTauEmp_sub = data['CovTauEmp_sub']
    CovTauSim_sub = data['CovTauSim_sub']
    SC_sub = data['SC_sub']
    GEC_sub = data['GEC_sub']

    pearson_fc, pearson_covtau, sc_dev = [], [], []
    for i in range(len(FCemp_sub)):
        FCemp = FCemp_sub[i]
        FCsim = FCsim_sub[i]
        CovEmp = CovTauEmp_sub[i]
        CovSim = CovTauSim_sub[i]
        SC = SC_sub[i]
        GEC = GEC_sub[i]

        # Pearson between lower-triangle vectors
        pearson_fc.append(pearson_corr(FCemp[Isubdiag], FCsim[Isubdiag]))
        pearson_covtau.append(pearson_corr(CovEmp[Isubdiag], CovSim[Isubdiag]))
        sc_dev.append(1 - pearson_corr(SC[Isubdiag], GEC[Isubdiag]))

    results.append(dict(
        tag=tag, model_type=model_type, thresh=thresh, lambdastr=lambdastr, lambda_idx=lambda_idx,
        mean_fc=np.mean(pearson_fc), mean_covtau=np.mean(pearson_covtau), mean_sc_dev=np.mean(sc_dev),
    ))

# ===============  PLOTTING: FC  ================

plt.figure(figsize=(10, 7))
plotted = set()
for r in results:
    marker, color = get_color_marker(r['thresh'], r['lambda_idx'], r['model_type'], base_cmaps, marker_map, lambdas)
    label = f"{r['model_type']}_{r['thresh']}"
    if r['lambdastr']:
        label += f"_{r['lambdastr']}"
    # Only one legend entry per unique label
    show_label = label if label not in plotted else None
    edgecolor = 'none' if marker == 'o' else 'k'
    plt.scatter(r['mean_sc_dev'], r['mean_fc'], s=90, marker=marker, color=color, edgecolor=edgecolor, label=show_label, zorder=3)
    plotted.add(label)

# -- Draw Pareto lines for each threshold --
for thresh in thresholds:
    group = [r for r in results if r['thresh'] == thresh]
    # Sort by lambda_idx (None goes last)
    group_sorted = sorted(group, key=lambda r: (r['lambda_idx'] is None, r['lambda_idx']))
    xs = [r['mean_sc_dev'] for r in group_sorted]
    ys = [r['mean_fc'] for r in group_sorted]
    # Get base color for this threshold
    color = base_cmaps[thresh](0.7)  # 0.7 is darker end
    plt.plot(xs, ys, '-', color=color, linewidth=0.5, alpha=0.8, zorder=2)

plt.xlabel('1 - Corr(GEC, SC)')
plt.ylabel('Corr(FC_emp, FC_sim)')
plt.title('FC Fit vs SC Deviation (Pearson)')
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
# Sort labels by (threshold, lambda) order
sorted_labels = sorted(unique.keys(), key=legend_sort_key)
plt.legend([unique[l] for l in sorted_labels], sorted_labels,
           bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(plotdir, 'Pareto_FC_vs_SCdev.png'), dpi=150)
plt.show()

# ===============  PLOTTING: CovTau()  ================

plt.figure(figsize=(10, 7))
plotted = set()
for r in results:
    marker, color = get_color_marker(r['thresh'], r['lambda_idx'], r['model_type'], base_cmaps, marker_map, lambdas)
    label = f"{r['model_type']}_{r['thresh']}"
    if r['lambdastr']:
        label += f"_{r['lambdastr']}"
    show_label = label if label not in plotted else None
    edgecolor = 'none' if marker == 'o' else 'k'
    plt.scatter(r['mean_sc_dev'], r['mean_covtau'], s=90, marker=marker, color=color, edgecolor=edgecolor, label=show_label, zorder=3)
    plotted.add(label)

# -- Draw Pareto lines for each threshold --
for thresh in thresholds:
    group = [r for r in results if r['thresh'] == thresh]
    group_sorted = sorted(group, key=lambda r: (r['lambda_idx'] is None, r['lambda_idx']))
    xs = [r['mean_sc_dev'] for r in group_sorted]
    ys = [r['mean_covtau'] for r in group_sorted]
    color = base_cmaps[thresh](0.7)  # 0.7 is darker end
    plt.plot(xs, ys, '-', color=color, linewidth=0.5, alpha=0.8, zorder=2)

plt.xlabel('1 - Corr(GEC, SC)')
plt.ylabel('Corr(CovTau_emp, CovTau_sim)')
plt.title('CovTau Fit vs SC Deviation (Pearson)')
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
sorted_labels = sorted(unique.keys(), key=legend_sort_key)
plt.legend([unique[l] for l in sorted_labels], sorted_labels,
           bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(plotdir, 'Pareto_CovTau_vs_SCdev.png'), dpi=150)
plt.show()