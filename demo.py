import os, json, shutil
from concurrent.futures import ProcessPoolExecutor
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.sparse import csr_matrix
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq
NP_TRAPEZOID = getattr(np, 'trapezoid', np.trapz)

def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}

RUN_FIG2 = env_flag('RUN_FIG2', True)
RUN_FIG3 = env_flag('RUN_FIG3', True)
RUN_FIG4 = env_flag('RUN_FIG4', True)
RUN_FIG5 = env_flag('RUN_FIG5', True)
RUN_FIG6 = env_flag('RUN_FIG6', True)
RUN_CONVERGENCE_CHECKS = env_flag('RUN_CONVERGENCE_CHECKS', False)
N_WORKERS = int(os.environ.get('N_WORKERS', max(1, os.cpu_count() - 2)))
a = 0.7
b = 0.8
c = 20.0
OSCILLATORY_Z = -a / b
SYNTH_EDGE_LOW = 0
SYNTH_EDGE_HIGH = 0.01
BENCHMARK_CONNECTION_PROBABILITY = 0.02
BENCHMARK_WEIGHT_MODE = os.environ.get('BENCHMARK_WEIGHT_MODE', 'uniform').strip().lower()
EMPIRICAL_WEIGHT_NORMALIZATION = 'total_weight'
USE_RAW_EMPIRICAL_WEIGHTS = False
ER_N = 200
ER_P = BENCHMARK_CONNECTION_PROBABILITY
ER_M = int(round(ER_P * ER_N * (ER_N - 1) / 2.0))
BENCHMARK_NOMINAL_MEAN_DEGREE = ER_P * (ER_N - 1)
ER_SEED = 47
FIG4_N = ER_N
FIG4_ER_M = ER_M
FIG4_ER_P = ER_P
FIG4_BA_M = int(round(0.5 * (FIG4_N - np.sqrt(FIG4_N ** 2 - 2.0 * BENCHMARK_CONNECTION_PROBABILITY * FIG4_N * (FIG4_N - 1)))))
FIG4_SW_K = int(2 * round(BENCHMARK_CONNECTION_PROBABILITY * (FIG4_N - 1) / 2.0))
FIG4_SW_P = 0.1
FIG4_NET_SEEDS = {'ER': ER_SEED, 'BA': 42, 'SW': 99}
FIG4_CONNECTION_PROBABILITIES = {'ER': FIG4_ER_P, 'BA': 2.0 * FIG4_BA_M * (FIG4_N - FIG4_BA_M) / (FIG4_N * (FIG4_N - 1)), 'SW': FIG4_SW_K / (FIG4_N - 1)}
if max((abs(value - BENCHMARK_CONNECTION_PROBABILITY) for value in FIG4_CONNECTION_PROBABILITIES.values())) > 1.0 / FIG4_N:
    raise ValueError('ER, BA and WS connection probabilities are not aligned.')
FIG5_N_RANDOM_REPEATS = int(os.environ.get('FIG5_N_RANDOM_REPEATS', 12))
FIG5_K_STEP = int(os.environ.get('FIG5_K_STEP', 2))
FIG5_FIXED_K_FRAC = float(os.environ.get('FIG5_FIXED_K_FRAC', 0.25))
FIG5_GROUP_K_FRACS = tuple((float(v) for v in os.environ.get('FIG5_GROUP_K_FRACS', '0.35,0.475,0.60').split(',') if v.strip()))
FIG5_DOSE_K_FRACS = tuple((float(v) for v in os.environ.get('FIG5_DOSE_K_FRACS', '0.25,0.375,0.475,0.60').split(',') if v.strip()))
FIG5_DOSE_GRID = tuple((float(v) for v in os.environ.get('FIG5_DOSE_GRID', '0,0.25,0.5,0.75,1,1.25,1.5').split(',') if v.strip()))
FIG5_BA_N = FIG4_N
FIG5_BA_M = FIG4_BA_M
FIG5_BA_SEED = FIG4_NET_SEEDS['BA']
FIG5_REFERENCE_DOSE = float(os.environ.get('FIG5_REFERENCE_DOSE', 1.0))
NET_X0, NET_Y0 = (1, 1)
INIT_VAR = 0.01
INIT_STD = float(np.sqrt(INIT_VAR))
USE_HETER_Z = True
Z_STD = 0.01
FIG2_STABLE_Z = 0.0
ABC_PERTURB = True
ABC_REL_HALF_RANGE = 0.05
tStart = 0.0
tEnd = float(os.environ.get('T_END', 150.0))
N_TIME = int(os.environ.get('N_TIME', 10001))
t_eval = np.linspace(tStart, tEnd, N_TIME)
discard_ratio = 0.2
FINAL_WINDOW_RATIO = 0.2
tail_amp_thr = 0.15
tail_std_thr = 0.03
min_peaks = 2
peak_prom = 0.06
peak_dist_ratio = 0.005
N_SCAN = int(os.environ.get('N_SCAN', 80))
N_ZSCAN_TRIALS = int(os.environ.get('N_ZSCAN_TRIALS', 24))
N_SENS = int(os.environ.get('N_SENS', 61))
N_CONTROL_PROB_TRIALS = int(os.environ.get('N_CONTROL_PROB_TRIALS', 80))
BRAIN_CONTROL_PROB_TRIALS = int(os.environ.get('BRAIN_CONTROL_PROB_TRIALS', 40))
FIG6_TOPK_TRIALS = int(os.environ.get('FIG6_TOPK_TRIALS', 40))
FIG6_LANDSCAPE_TRIALS = int(os.environ.get('FIG6_LANDSCAPE_TRIALS', 32))
FIG6_INTERVAL_SCAN_POINTS = int(os.environ.get('FIG6_INTERVAL_SCAN_POINTS', 25))
FIG6_INTERVAL_SCAN_TRIALS = int(os.environ.get('FIG6_INTERVAL_SCAN_TRIALS', 8))
CONTROL_PROB_THRESHOLD = float(os.environ.get('CONTROL_PROB_THRESHOLD', 1.0))
CONTROL_PROB_STRICT_THRESHOLD = float(os.environ.get('CONTROL_PROB_STRICT_THRESHOLD', 1.0))
FIG4_K_STEP = int(os.environ.get('FIG4_K_STEP', 5))
FIG2_DIAG_Z_HALF_WIDTH = 0.18
FIG2_SEED_SHIFT = int(os.environ.get('FIG2_SEED_SHIFT', 17))
REFERENCE_DRIVER_DOSE = float(os.environ.get('REFERENCE_DRIVER_DOSE', os.environ.get('MAX_DRIVER_Z', 1.0)))
MAX_DRIVER_Z = REFERENCE_DRIVER_DOSE
CONTROL_INPUT_POLICY = os.environ.get('CONTROL_INPUT_POLICY', 'fixed_dose').strip().lower()
FIG4_DOSE_GRID = tuple((float(v) for v in os.environ.get('FIG4_DOSE_GRID', '0.5,0.75,1,1.25,1.5').split(',') if v.strip()))
FIG4_LANDSCAPE_DOSE_GRID = tuple((float(v) for v in os.environ.get('FIG4_LANDSCAPE_DOSE_GRID', '0,0.25,0.5,0.75,1,1.25,1.5').split(',') if v.strip()))
FIG6_LANDSCAPE_DOSE_GRID = tuple((float(v) for v in os.environ.get('FIG6_LANDSCAPE_DOSE_GRID', '0.2,0.4,0.6,0.8,1,1.5,2,3,4,6').split(',') if v.strip()))
FIG6_TOPK_FRACS = tuple((float(v) for v in os.environ.get('FIG6_TOPK_FRACS', '0,0.02,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.65,0.675,0.70,0.7125,0.725,0.7375,0.75,0.7625,0.775,0.7875,0.80,0.825,0.85,0.90,1.0').split(',') if v.strip()))
N_SENS_Z = int(os.environ.get('N_SENS_Z', 49))
N_SENS_TRIALS = int(os.environ.get('N_SENS_TRIALS', 16))
SENS_SMOOTH_SIGMA = float(os.environ.get('SENS_SMOOTH_SIGMA', 1.05))
SENS_Z_ABS_MAX = float(os.environ.get('SENS_Z_ABS_MAX', 3.6))
STABLE_TARGET_MARGIN = 0.1
N_Z_PERIOD = int(os.environ.get('N_Z_PERIOD', 15))
dpi_out = 400
FIG_EXT = '.pdf'
FIG_EXTS = tuple((ext if ext.startswith('.') else f'.{ext}' for ext in os.environ.get('FIG_EXTS', 'svg,pdf').split(',') if ext.strip()))
CMAP_NODES = LinearSegmentedColormap.from_list('node_soft_blue_teal', ['#D5E8F1', '#ABD7DF', '#CAEBE7', '#A9D9BB', '#90B4CF', '#4FB1B2', '#337BAC']).resampled(80)
ALPHA_NODES = 0.12
LW_NODES = 0.5
COLOR_XW = '#C73D47'
LW_XW = 2.2
COLOR_XEFF = '#2C6BB3'
LW_XEFF = 2.0
LS_XEFF = '--'
COLOR_THEORY_FP = '#EB7F24'
COLOR_EMP_FP = '#4FB1B2'
COLOR_ERR_MEAN = '#D33934'
COLOR_NODE_MEAN = '#272727'
COLOR_STABLE = '#4FB1B2'
COLOR_OSC = '#C73D47'
COLOR_STABLE_FILL = '#DDECF2'
COLOR_OSC_FILL = '#F2CFD4'
CMAP_SUCCESS = LinearSegmentedColormap.from_list('success_blue_to_blush', ['#99DEFC', '#FD9EAA'], N=256)
SUCCESS_LEVELS = np.linspace(0, 1, 256)
CMAP_OSC_PROB = LinearSegmentedColormap.from_list('oscillation_probability', ['#D5E8F1', '#F5E4CC', '#F2CFD4', COLOR_OSC])
CMAP_OUTSTRENGTH = LinearSegmentedColormap.from_list('outstrength_rank_sci24', ['#D5E8F1', '#CAEBE7', '#ABD7DF', '#A9D9BB', '#4FB1B2', '#90B4CF', '#337BAC'])
COLOR_BRAIN = {'fly_core80': '#C73D47', 'fly_core120': '#2C6BB3', 'fly_core150': '#4FB1B2'}
COLOR_REAL_BRAIN = {'allen_mouse': '#5C87B1', 'marmoset': '#38C08F', 'hcp': '#9AA8B4', 'gw': '#A7B8AE', 'fly_core150': COLOR_BRAIN['fly_core150'], 'drosophila': '#C73D47', 'drosophila_full': '#C73D47'}
COLOR_NET = {'ER': '#5C87B1', 'BA': '#38C08F', 'SW': '#EB7F24'}
COLOR_STRATEGY = {'out': '#2C6BB3', 'in': '#C73D47', 'random': '#6FBA4F'}
COLOR_ALLNODE_CONTROL = '#5F6368'
COLOR_NO_CONTROL = '#A8B3BD'
FIG6_REGION_LABEL = {'isocortex': 'Isocortex', 'cortex': 'Cortex', 'frontal': 'Frontal', 'prefrontal': 'Prefrontal', 'premotor': 'Premotor', 'medial_cingulate': 'Medial cingulate', 'orbital_insular': 'Orbital/insular', 'temporal': 'Temporal', 'parietal': 'Parietal', 'occipital': 'Occipital', 'cingulate': 'Cingulate', 'insula': 'Insula', 'hippocampal': 'Hippocampal', 'thalamus': 'Thalamus', 'anterior_thalamic': 'Anterior thalamic nuclei', 'lateral_hypothalamus': 'Lateral hypothalamus', 'cerebellar_nuclei': 'Cerebellar nuclei', 'striatum': 'Striatum', 'hypothalamus': 'Hypothalamus', 'midbrain': 'Midbrain', 'hindbrain': 'Hindbrain', 'olfactory': 'Olfactory', 'sensorimotor': 'Sensorimotor', 'motor': 'Motor output', 'sensory_all': 'Sensory all', 'sensory_visual': 'Visual', 'sensory_olfactory': 'Olfactory', 'sensory_gustatory': 'Gustatory', 'mushroom_body': 'Mushroom body', 'interneuron': 'Interneurons', 'projection': 'Projection', 'descending_sez': 'Descending SEZ', 'random_eq_motor': 'Random motor-sized', 'all': 'All nodes', 'whole_thalamus': 'Whole thalamus', 'thalamo_hippocampal': 'Thalamus + hippocampal', 'thalamo_cortical': 'Thalamocortical', 'thalamo_hypothalamic': 'Thalamus + hypothalamus', 'thalamo_cerebellar': 'Thalamus + cerebellar nuclei', 'thalamo_cortico_hypothalamic': 'Thalamo-cortico-hypothalamic'}
COLOR_REGION = {'isocortex': '#5C87B1', 'cortex': '#5C87B1', 'frontal': '#5C87B1', 'prefrontal': '#337BAC', 'premotor': '#38C08F', 'medial_cingulate': '#7A9EC2', 'orbital_insular': '#4FB1B2', 'temporal': '#90B4CF', 'parietal': '#337BAC', 'occipital': '#ABD7DF', 'cingulate': '#7A9EC2', 'insula': '#4FB1B2', 'hippocampal': '#38C08F', 'thalamus': '#A9D9BB', 'anterior_thalamic': '#6FBA4F', 'lateral_hypothalamus': '#DD7597', 'cerebellar_nuclei': '#9AA8B4', 'striatum': '#F18E00', 'hypothalamus': '#DD7597', 'midbrain': '#A958B3', 'hindbrain': '#9AA8B4', 'olfactory': '#FFD83E', 'sensorimotor': '#DB4751', 'motor': '#DB4751', 'sensory_all': '#F18E00', 'sensory_visual': '#F18E00', 'sensory_olfactory': '#F18E00', 'sensory_gustatory': '#F18E00', 'mushroom_body': '#FFD83E', 'interneuron': '#00BB80', 'projection': '#007CCB', 'descending_sez': '#A958B3', 'random': '#BDC3C7', 'all': COLOR_ALLNODE_CONTROL, 'whole_thalamus': '#A9D9BB', 'thalamo_hippocampal': '#0C8F7A', 'thalamo_cortical': '#2C6BB3', 'thalamo_hypothalamic': '#C45A8A', 'thalamo_cerebellar': '#7A7F87', 'thalamo_cortico_hypothalamic': '#413A97'}
output_dir = os.environ.get('OUTPUT_DIR', 'nature_output_U(0,0.01)+p0.02')
DIRS = {}
for name in ['fig2', 'fig3', 'fig4', 'fig5', 'fig6']:
    d = os.path.join(output_dir, name)
    os.makedirs(d, exist_ok=True)
    DIRS[name] = d

def reset_main_output_dirs():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for name in ['fig2', 'fig3', 'fig4', 'fig5', 'fig6']:
        d = os.path.join(output_dir, name)
        os.makedirs(d, exist_ok=True)
        DIRS[name] = d
plt.ioff()
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 8, 'mathtext.fontset': 'stix', 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False, 'legend.fontsize': 7})

def save_sub(fig, folder, name):
    if not getattr(fig, '_skip_tight_layout', False):
        fig.tight_layout()
    first_path = None
    for ext in FIG_EXTS:
        p = os.path.join(folder, f'{name}{ext}')
        fig.savefig(p, dpi=dpi_out, bbox_inches='tight', facecolor='white')
        if first_path is None:
            first_path = p
    plt.close(fig)
    print(f'  Saved: {first_path}')

def save_json(obj, path):

    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=conv)

def cleanup_outputs(folder, names):
    exts = [FIG_EXT, '.pdf', '.png', '.svg', '.jpg', '.jpeg', '.json', '.csv']
    for name in names:
        candidates = [name] if os.path.splitext(name)[1] else [name + ext for ext in exts]
        for p in candidates:
            fp = os.path.join(folder, p)
            if os.path.exists(fp):
                os.remove(fp)

def mfig(figsize=(6, 4.5)):
    return plt.subplots(1, 1, figsize=figsize)

def safe_label(text):
    return str(text).replace('_', ' ')

def label_line_end(ax, x, y, label, color, dx=0.012, fontsize=7, va='center'):
    if len(x) == 0 or len(y) == 0:
        return
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return
    xx = x[ok][-1]
    yy = y[ok][-1]
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.text(xx + dx * span, yy, label, color=color, fontsize=fontsize, va=va, ha='left', clip_on=False)

def annotate_threshold_point(ax, x, y, label, color):
    if np.isfinite(x) and np.isfinite(y):
        ax.scatter([x], [y], s=46, c=color, edgecolors='k', lw=0.45, zorder=5)
        ax.text(x, y, f'  {label}', color=color, fontsize=7, ha='left', va='center')

def legend_clean(ax, handles=None, labels=None, **kwargs):
    defaults = dict(fontsize=6.5, frameon=True, fancybox=False, framealpha=0.9, edgecolor='#D7D7D7', facecolor='white', borderpad=0.35, handlelength=1.4, handletextpad=0.45, labelspacing=0.35, columnspacing=0.9)
    defaults.update(kwargs)
    if handles is not None:
        leg = ax.legend(handles=handles, labels=labels, **defaults)
    else:
        leg = ax.legend(**defaults)
    if leg is not None:
        leg.get_frame().set_linewidth(0.45)
    return leg

def benchmark_edge_weights(size, seed):
    size = int(size)
    if BENCHMARK_WEIGHT_MODE == 'uniform':
        return np.random.default_rng(int(seed)).uniform(SYNTH_EDGE_LOW, SYNTH_EDGE_HIGH, size=size)
    elif BENCHMARK_WEIGHT_MODE == 'unit':
        return np.full(size, 0.5 * (SYNTH_EDGE_LOW + SYNTH_EDGE_HIGH), dtype=float)
    else:
        raise ValueError("BENCHMARK_WEIGHT_MODE must be 'uniform' or 'unit'.")

def orient_undirected_skeleton_50_50(G, n, direction_seed, weight_seed):
    edges = list(G.edges())
    rg_direction = np.random.default_rng(int(direction_seed))
    weights = benchmark_edge_weights(len(edges), int(weight_seed))
    A = np.zeros((int(n), int(n)), dtype=float)
    for edge_idx, (u, v) in enumerate(edges):
        if rg_direction.random() < 0.5:
            src, dst = (u, v)
        else:
            src, dst = (v, u)
        A[int(dst), int(src)] = weights[edge_idx]
    return A

def make_directed_er_fixed_edges(n, m_edges, seed):
    n = int(n)
    m_edges = int(m_edges)
    max_edges = n * (n - 1) // 2
    if not 0 <= m_edges <= max_edges:
        raise ValueError(f'm_edges must be in [0, {max_edges}].')
    G = nx.gnm_random_graph(n, m_edges, seed=int(seed), directed=False)
    return orient_undirected_skeleton_50_50(G, n, direction_seed=int(seed) + 777, weight_seed=int(seed) + 1009)

def make_shared_fig234_network():
    return make_directed_er_fixed_edges(ER_N, ER_M, ER_SEED)

def make_directed_ba(n, m, seed):
    n = int(n)
    m = int(np.clip(int(m), 1, max(1, n - 1)))
    G = nx.barabasi_albert_graph(n, m, seed=int(seed))
    return orient_undirected_skeleton_50_50(G, n, direction_seed=int(seed) + 777, weight_seed=int(seed) + 1777)

def make_directed_sw(n, k, p, seed):
    G = nx.watts_strogatz_graph(int(n), int(k), float(p), seed=int(seed))
    return orient_undirected_skeleton_50_50(G, int(n), direction_seed=int(seed) + 333, weight_seed=int(seed) + 1333)

def make_fig4_network(net_type, n=None, seed=None):
    n = n or FIG4_N
    seed = seed or FIG4_NET_SEEDS.get(net_type, 42)
    if net_type == 'ER':
        A = make_directed_er_fixed_edges(n, FIG4_ER_M, seed)
    elif net_type == 'BA':
        A = make_directed_ba(n, FIG4_BA_M, seed)
    elif net_type == 'SW':
        A = make_directed_sw(n, FIG4_SW_K, FIG4_SW_P, seed)
    else:
        raise ValueError(f'Unknown network type: {net_type}')
    return A

def rng(seed):
    return np.random.default_rng(int(seed))

def degree_weights(A):
    s = A.sum(axis=0)
    d = float(s.sum())
    return s / d if d > 1e-14 else np.ones(A.shape[0]) / A.shape[0]

def effective_coupling(A):
    kin = A.sum(axis=1)
    kout = A.sum(axis=0)
    d = float(kout.sum())
    return float(np.dot(kout, kin) / d) if d > 1e-14 else 0.0

def benchmark_operating_point_info(A, label='network'):
    A = np.asarray(A, dtype=float)
    Ae = effective_coupling(A)
    if Ae <= 1e-14:
        raise ValueError('Synthetic benchmark requires positive A_eff.')
    theory = derive_z_interval(Ae)
    width = max(float(theory['width']), 1e-12)
    oscillatory_z_in_interval = bool(theory['z_left'] <= OSCILLATORY_Z <= theory['z_right'])
    if not oscillatory_z_in_interval:
        raise ValueError(f'{label}: z=-a/b={OSCILLATORY_Z:.6g} is outside the predicted oscillatory interval.')
    return (A.copy(), {'label': label, 'normalization': 'direct_edge_sampling_no_realization_rescaling', 'edge_rule': 'A_ij~U(edge_low,edge_high) on nonzero synthetic edges', 'weight_mode': BENCHMARK_WEIGHT_MODE, 'edge_weight_support': [SYNTH_EDGE_LOW, SYNTH_EDGE_HIGH], 'mean_degree_selection': 'fixed a priori synthetic-benchmark setting', 'nominal_mean_degree': float(BENCHMARK_NOMINAL_MEAN_DEGREE), 'A_eff': float(Ae), 'oscillatory_z': float(OSCILLATORY_Z), 'control_baseline': float(OSCILLATORY_Z), 'z_interval': {'left': float(theory['z_left']), 'right': float(theory['z_right']), 'width': float(theory['width'])}, 'oscillatory_z_in_interval': oscillatory_z_in_interval, 'oscillatory_z_rel_margin': float(min(OSCILLATORY_Z - theory['z_left'], theory['z_right'] - OSCILLATORY_Z) / width), 'zero_in_interval': bool(theory['z_left'] <= 0.0 <= theory['z_right'])})

def matrix_for_aeff_sensitivity(A, target_aeff):
    A = np.asarray(A, dtype=float)
    current = effective_coupling(A)
    target = float(target_aeff)
    if current <= 1e-14:
        if abs(target) <= 1e-14:
            return A.copy()
        raise ValueError('Cannot scan nonzero A_eff from a zero-coupling network.')
    return A * (target / current)

def weighted_series(X, w):
    return X @ w

def make_ic(n, seed_val):
    rg = rng(seed_val)
    return (rg.normal(loc=NET_X0, scale=INIT_STD, size=int(n)), rg.normal(loc=NET_Y0, scale=INIT_STD, size=int(n)))

def make_z_vec(z0, w, seed_val):
    n = len(w)
    if not USE_HETER_Z or Z_STD <= 0:
        return np.full(n, z0)
    rg_ = rng(seed_val)
    dz = rg_.normal(0, Z_STD, n)
    dz -= float(np.dot(w, dz))
    return np.full(n, z0) + dz

def _bounded_weighted_relative_perturbation(mean_value, n, w, rg_, half_range=ABC_REL_HALF_RANGE):
    eta = rg_.uniform(-half_range, half_range, size=int(n))
    eta -= float(np.dot(w, eta)) / max(float(np.sum(w)), 1e-12)
    eta = np.clip(eta, -half_range, half_range)
    eta -= float(np.dot(w, eta)) / max(float(np.sum(w)), 1e-12)
    return mean_value * (1.0 + eta)

def make_abc(n, w, seed_val):
    if not ABC_PERTURB:
        return (np.full(n, a), np.full(n, b), np.full(n, c))
    rg_ = rng(seed_val)
    av = _bounded_weighted_relative_perturbation(a, n, w, rg_)
    bv = _bounded_weighted_relative_perturbation(b, n, w, rg_)
    cv = _bounded_weighted_relative_perturbation(c, n, w, rg_)
    return (av, bv, cv)

def z_eff_of_x(x, Ae):
    return x ** 3 / 3.0 - (1.0 + Ae - 1.0 / b) * x - a / b

def derive_z_interval(Ae):
    alpha = 1.0 + Ae - 1.0 / b
    beta = 1.0 + Ae - b / c ** 2
    Ac = 4.0 / (3.0 * b) - b / (3.0 * c ** 2) - 1.0
    out = {'A_eff': float(Ae), 'A_cross': float(Ac), 'z1': None, 'z2': None, 'sel': None}
    if alpha > 0:
        amp = 2.0 / 3.0 * alpha ** 1.5
        out['z1'] = (float(-amp - a / b), float(amp - a / b))
    if beta > 0:
        xh = np.sqrt(beta)
        z2l, z2r = sorted([z_eff_of_x(-xh, Ae), z_eff_of_x(xh, Ae)])
        out['z2'] = (float(z2l), float(z2r))
    if Ae < Ac:
        out['sel'] = out['z2'] or out['z1']
    else:
        out['sel'] = out['z1'] or out['z2']
    if out['sel'] is None:
        raise ValueError('No oscillatory interval.')
    out['z_left'], out['z_right'] = out['sel']
    out['width'] = out['z_right'] - out['z_left']
    return out

def stable_target_margin_info(theory, margin_fraction=STABLE_TARGET_MARGIN):
    width = max(float(theory['width']), 1e-12)
    eps = float(ABC_REL_HALF_RANGE) if ABC_PERTURB else 0.0
    if eps >= 1.0:
        raise ValueError('ABC_REL_HALF_RANGE must be smaller than one.')
    ab_shift_bound = a / b * (2.0 * eps / (1.0 - eps)) if eps > 0.0 else 0.0
    z_noise_bound = 3.0 * float(Z_STD) if USE_HETER_Z else 0.0
    fractional_buffer = float(margin_fraction) * width
    absolute_buffer = max(fractional_buffer, ab_shift_bound + z_noise_bound)
    return {'method': 'max(fraction*interval_width, worst_case_a_over_b_shift+3sigma_z)', 'fraction': float(margin_fraction), 'fractional_buffer': float(fractional_buffer), 'a_over_b_shift_bound': float(ab_shift_bound), 'three_sigma_z': float(z_noise_bound), 'absolute_buffer': float(absolute_buffer)}

def analytic_nonoscillatory_target(theory, margin=STABLE_TARGET_MARGIN, side='right'):
    zl = float(theory['z_left'])
    zr = float(theory['z_right'])
    margin = float(margin)
    if margin < 0.0:
        raise ValueError('Stable-target margin must be nonnegative.')
    buffer = stable_target_margin_info(theory, margin)['absolute_buffer']
    if side == 'right':
        return float(zr + buffer)
    if side == 'left':
        return float(zl - buffer)
    raise ValueError("Analytical stable-target side must be 'left' or 'right'.")

def eff_fixed_points(Ae, z0):
    alpha = 1.0 + Ae - 1.0 / b
    coeff = [1, 0, -3 * alpha, -3 * (a / b + z0)]
    roots = np.roots(coeff)
    pts = []
    for r in roots:
        if abs(r.imag) < 1e-08:
            x = float(r.real)
            y = float((a - x) / b)
            J = np.array([[c * (1 - x * x + Ae), c], [-1 / c, -b / c]])
            ev = np.linalg.eigvals(J)
            pts.append({'x': x, 'y': y, 'stable': bool(np.max(np.real(ev)) < 0), 'eig_re': [float(np.real(e)) for e in ev], 'eig_im': [float(np.imag(e)) for e in ev]})
    return sorted(pts, key=lambda d: d['x'])

def choose_nonoscillatory_target(theory, margin=STABLE_TARGET_MARGIN, side='nearest'):
    if side in ('nearest', 'mildest'):
        left = analytic_nonoscillatory_target(theory, margin, 'left')
        right = analytic_nonoscillatory_target(theory, margin, 'right')
        return float(left if abs(left) <= abs(right) else right)
    return analytic_nonoscillatory_target(theory, margin, side)

def network_operating_point_info(A, label='network'):
    A = np.asarray(A, dtype=float)
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    width = max(float(theory['width']), 1e-12)
    return (A.copy(), {'label': label, 'A_eff': float(Ae), 'z_interval': {'left': float(theory['z_left']), 'right': float(theory['z_right']), 'width': float(theory['width'])}, 'oscillatory_z': float(OSCILLATORY_Z), 'oscillatory_z_in_interval': bool(theory['z_left'] <= OSCILLATORY_Z <= theory['z_right']), 'oscillatory_z_rel_margin': float(min(OSCILLATORY_Z - theory['z_left'], theory['z_right'] - OSCILLATORY_Z) / width), 'zero_in_interval': bool(theory['z_left'] <= 0.0 <= theory['z_right']), 'control_baseline': float(OSCILLATORY_Z)})

def make_oscillatory_z_vec(n):
    return np.full(int(n), float(OSCILLATORY_Z), dtype=float)

def make_weighted_mean_z_vec(z_eff_target, w, seed_val, std=Z_STD):
    w = np.asarray(w, dtype=float)
    n = len(w)
    if std <= 0:
        return np.full(n, float(z_eff_target))
    rg_ = rng(seed_val)
    dz = rg_.normal(0.0, float(std), n)
    dz -= float(np.dot(w, dz))
    return np.full(n, float(z_eff_target)) + dz

def z_eff_value(zv, w):
    return float(np.dot(w, zv))

def driver_weight(w, drivers):
    if len(drivers) == 0:
        return 0.0
    return float(np.sum(np.asarray(w)[np.asarray(drivers, dtype=int)]))

def driver_z_from_dose(z_eff_target, dose, baseline_z=OSCILLATORY_Z):
    direction = np.sign(float(z_eff_target) - float(baseline_z))
    if direction == 0.0:
        direction = 1.0
    return float(baseline_z + direction * abs(float(dose)))

def realized_z_eff_from_driver(W_D, z_driver, baseline_z=OSCILLATORY_Z):
    W_D = float(W_D)
    baseline_z = float(baseline_z)
    return float(baseline_z + W_D * (float(z_driver) - baseline_z))

def solve_baseline_driver_z(w, drivers, z_eff_target, reference_dose=None, baseline_z=OSCILLATORY_Z):
    W_D = driver_weight(w, drivers)
    baseline_z = float(baseline_z)
    target = float(z_eff_target)
    dose_limit = float(MAX_DRIVER_Z if reference_dose is None else abs(reference_dose))
    if W_D <= 1e-14:
        return {'valid': False, 'z_driver': np.nan, 'W_D': float(W_D), 'target_feasible': False, 'z_driver_unbounded': np.nan, 'delta_z_driver_unbounded': np.nan, 'z_eff_target': target, 'z_eff_realized': baseline_z, 'baseline_z': baseline_z, 'reference_dose': dose_limit}
    non_driver_eff = (1.0 - float(W_D)) * baseline_z
    z_driver_unbounded = float((target - non_driver_eff) / float(W_D))
    delta_unbounded = float(z_driver_unbounded - baseline_z)
    policy = CONTROL_INPUT_POLICY
    if policy in {'fixed', 'fixed_dose', 'dose', 'max'}:
        z_driver = driver_z_from_dose(target, dose_limit, baseline_z) if dose_limit > 0 else z_driver_unbounded
    elif dose_limit > 0:
        delta_applied = float(np.clip(delta_unbounded, -dose_limit, dose_limit))
        z_driver = float(baseline_z + delta_applied)
    else:
        z_driver = z_driver_unbounded
    target_feasible = bool(abs(delta_unbounded) <= dose_limit) if dose_limit > 0 else True
    z_eff_realized = float(non_driver_eff + float(W_D) * z_driver)
    return {'valid': True, 'z_driver': z_driver, 'W_D': float(W_D), 'control_policy': policy, 'target_feasible': target_feasible, 'z_driver_unbounded': z_driver_unbounded, 'delta_z_driver_unbounded': delta_unbounded, 'z_eff_target': target, 'z_eff_realized': z_eff_realized, 'baseline_z': baseline_z, 'reference_dose': dose_limit}

def _bracketed_roots(func, x0, x1, n_scan=4097, tol=1e-10):
    lo, hi = sorted((float(x0), float(x1)))
    grid = np.linspace(lo, hi, max(33, int(n_scan)))
    values = np.asarray(func(grid), dtype=float)
    if not np.all(np.isfinite(values)):
        return [np.nan]
    roots = []
    near = np.flatnonzero(np.abs(values) <= tol)
    roots.extend((float(grid[idx]) for idx in near))
    for idx in np.flatnonzero(values[:-1] * values[1:] < 0.0):
        roots.append(float(brentq(func, grid[idx], grid[idx + 1])))
    roots.sort()
    unique = []
    for root in roots:
        if not unique or not np.isfinite(root) or abs(root - unique[-1]) > 1e-07:
            unique.append(root)
    return unique

def _relaxation_geometry(Ae, z0):
    if 1.0 + Ae <= 1e-12 or b <= 0.0:
        return {'valid': False, 'flow_valid': False}
    xf = float(np.sqrt(1.0 + Ae))
    z_total = float(z0)

    def hnull(x):
        x = np.asarray(x, dtype=float)
        return x ** 3 / 3.0 - (1.0 + Ae) * x - z_total

    def gslow(x):
        x = np.asarray(x, dtype=float)
        return x - a + b * hnull(x)

    def outer_root(y_target, side):
        coeff = [1.0, 0.0, -3.0 * (1.0 + Ae), -3.0 * (z_total + y_target)]
        roots = sorted((float(root.real) for root in np.roots(coeff) if abs(root.imag) < 1e-08))
        if side == 'right':
            candidates = [root for root in roots if root > xf + 1e-09]
            return max(candidates) if candidates else None
        candidates = [root for root in roots if root < -xf - 1e-09]
        return min(candidates) if candidates else None
    y_left_fold = float(hnull(-xf))
    y_right_fold = float(hnull(xf))
    x_right = outer_root(y_left_fold, 'right')
    x_left = outer_root(y_right_fold, 'left')
    if x_right is None or x_left is None:
        return {'valid': False, 'flow_valid': False}
    right_roots = _bracketed_roots(gslow, xf, x_right)
    left_roots = _bracketed_roots(gslow, x_left, -xf)
    g_right_fold = float(gslow(xf))
    g_left_fold = float(gslow(-xf))
    flow_valid = bool(g_right_fold > 0.0 and g_left_fold < 0.0 and (not right_roots) and (not left_roots))
    return {'valid': bool(flow_valid), 'flow_valid': bool(flow_valid), 'x_left': float(x_left), 'x_right': float(x_right), 'x_fold_left': float(-xf), 'x_fold_right': float(xf), 'g_fold_left': g_left_fold, 'g_fold_right': g_right_fold, 'left_branch_roots': left_roots, 'right_branch_roots': right_roots}

def relaxation_period(Ae, z0, nq=20000):
    geometry = _relaxation_geometry(Ae, z0)
    if not geometry.get('valid', False):
        return {**geometry, 'valid': False, 'flow_valid': False, 'T': np.nan}

    def hnull(x):
        return x ** 3 / 3.0 - (1.0 + Ae) * x - float(z0)

    def hprime(x):
        return x * x - (1.0 + Ae)

    def branch_time(x0, x1):
        x_grid = np.linspace(float(x0), float(x1), max(100, int(nq)))
        denominator = x_grid - a + b * hnull(x_grid)
        if not np.all(np.isfinite(denominator)) or np.any(np.abs(denominator) <= 1e-10):
            return np.nan
        integrand = c * np.abs(hprime(x_grid) / denominator)
        return float(abs(NP_TRAPEZOID(integrand, x_grid)))
    t_right = branch_time(geometry['x_right'], geometry['x_fold_right'])
    t_left = branch_time(geometry['x_left'], geometry['x_fold_left'])
    if not np.isfinite(t_right) or not np.isfinite(t_left):
        return {**geometry, 'valid': False, 'flow_valid': True, 'T': np.nan}
    return {'valid': True, 'flow_valid': True, 'T': float(t_right + t_left), 'T_left': float(t_left), 'T_right': float(t_right), **geometry}

def relaxation_amplitude(Ae, z0):
    geometry = _relaxation_geometry(Ae, z0)
    if not geometry.get('valid', False):
        return {**geometry, 'valid': False, 'flow_valid': False, 'A': np.nan}
    return {'valid': True, 'flow_valid': True, 'A': float(geometry['x_right'] - geometry['x_left']), **geometry}

def rhs_net(x, y, A_csr, zv, av, bv, cv):
    Ax = A_csr.dot(x)
    return (cv * (y + x - x ** 3 / 3 + zv + Ax), -(x - av + bv * y) / cv)

def rhs_red(xe, ye, z0, Ae):
    return (c * (ye + xe - xe ** 3 / 3 + z0 + Ae * xe), -(xe - a + b * ye) / c)

def ensure_csr(A):
    return A if hasattr(A, 'format') and A.format == 'csr' else csr_matrix(A)

def rk4_net(A, x0, y0, zv, tev, av=None, bv=None, cv=None):
    n = A.shape[0]
    dt = float(tev[1] - tev[0])
    T = len(tev)
    if av is None:
        av = np.full(n, a)
    if bv is None:
        bv = np.full(n, b)
    if cv is None:
        cv = np.full(n, c)
    Ac = ensure_csr(A)
    x, y = (x0.copy(), y0.copy())
    xa, ya = (np.empty((T, n)), np.empty((T, n)))
    xa[0], ya[0] = (x, y)
    for i in range(1, T):
        k1x, k1y = rhs_net(x, y, Ac, zv, av, bv, cv)
        k2x, k2y = rhs_net(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, Ac, zv, av, bv, cv)
        k3x, k3y = rhs_net(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, Ac, zv, av, bv, cv)
        k4x, k4y = rhs_net(x + dt * k3x, y + dt * k3y, Ac, zv, av, bv, cv)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        xa[i], ya[i] = (x, y)
    return {'t': tev, 'x_all': xa, 'y_all': ya, 'z_vec': zv.copy()}

def rk4_net_collective(A, x0, y0, zv, tev, w, av=None, bv=None, cv=None):
    n = A.shape[0]
    dt = float(tev[1] - tev[0])
    T = len(tev)
    if av is None:
        av = np.full(n, a)
    if bv is None:
        bv = np.full(n, b)
    if cv is None:
        cv = np.full(n, c)
    Ac = ensure_csr(A)
    x, y = (x0.copy(), y0.copy())
    w = np.asarray(w, dtype=float)
    xw, yw = (np.empty(T), np.empty(T))
    xw[0], yw[0] = (float(np.dot(x, w)), float(np.dot(y, w)))
    for i in range(1, T):
        k1x, k1y = rhs_net(x, y, Ac, zv, av, bv, cv)
        k2x, k2y = rhs_net(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, Ac, zv, av, bv, cv)
        k3x, k3y = rhs_net(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, Ac, zv, av, bv, cv)
        k4x, k4y = rhs_net(x + dt * k3x, y + dt * k3y, Ac, zv, av, bv, cv)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        xw[i], yw[i] = (float(np.dot(x, w)), float(np.dot(y, w)))
    return {'t': tev, 'xw': xw, 'yw': yw, 'z_vec': np.asarray(zv).copy()}

def rk4_red(x0, y0, z0, Ae, tev):
    dt = float(tev[1] - tev[0])
    x, y = (float(x0), float(y0))
    xs, ys = (np.empty(len(tev)), np.empty(len(tev)))
    xs[0], ys[0] = (x, y)
    for i in range(1, len(tev)):
        k1x, k1y = rhs_red(x, y, z0, Ae)
        k2x, k2y = rhs_red(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z0, Ae)
        k3x, k3y = rhs_red(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z0, Ae)
        k4x, k4y = rhs_red(x + dt * k3x, y + dt * k3y, z0, Ae)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        y += dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        xs[i], ys[i] = (x, y)
    return {'t': tev, 'x': xs, 'y': ys}

def tail(s):
    n0 = int(discard_ratio * len(s))
    return (s[n0:], n0)

def final_window(s, fraction=FINAL_WINDOW_RATIO):
    n = len(s)
    n0 = max(0, n - max(1, int(np.ceil(float(fraction) * n))))
    return (s[n0:], n0)

def _smooth_for_peak_detection(st):
    st = np.asarray(st, dtype=float)
    if len(st) < 15 or not np.all(np.isfinite(st)):
        return st
    win = max(5, int(round(0.003 * len(st))))
    if win % 2 == 0:
        win += 1
    win = min(win, len(st) if len(st) % 2 == 1 else len(st) - 1)
    if win < 5:
        return st
    kernel = np.ones(win, dtype=float) / float(win)
    half = win // 2
    padded = np.pad(st, (half, half), mode='edge')
    return np.convolve(padded, kernel, mode='valid')

def detect_osc(series, t):
    st, n0 = tail(series)
    if len(st) == 0 or not np.all(np.isfinite(st)):
        return {'is_osc': False, 'valid': False, 'amp': np.nan, 'n_peaks': 0, 'peaks': np.array([], dtype=int), 'n0': n0}
    amp = float(np.max(st) - np.min(st))
    std = float(np.std(st))
    md = max(1, int(len(st) * peak_dist_ratio))
    st_peak = _smooth_for_peak_detection(st)
    pks, _ = find_peaks(st_peak, prominence=peak_prom, distance=md)
    return {'is_osc': bool(amp > tail_amp_thr and std > tail_std_thr and (len(pks) >= min_peaks)), 'valid': True, 'amp': amp, 'n_peaks': len(pks), 'peaks': pks, 'n0': n0}

def collective_state_metrics(series, t):
    det = detect_osc(series, t)
    st, _ = tail(series)
    state = 'invalid' if not det.get('valid', True) else 'oscillatory' if det['is_osc'] else 'nonoscillatory'
    return {'state': state, 'is_osc': bool(det['is_osc']), 'valid': bool(det.get('valid', True)), 'tail_amp': float(det['amp']), 'tail_std': float(np.std(st)) if np.all(np.isfinite(st)) else np.inf, 'n_peaks': int(det['n_peaks'])}

def summarize_control_probability(rows):
    rows = list(rows)
    baseline_valid = [row for row in rows if row.get('baseline_valid', True)]
    control_valid = [row for row in rows if row.get('controlled_valid', True)]
    paired_valid = [row for row in rows if row.get('baseline_valid', True) and row.get('controlled_valid', True)]
    return {'success_probability': float(np.mean([row['success'] for row in paired_valid])) if paired_valid else np.nan, 'controlled_nonoscillatory_probability': float(np.mean([row['nonoscillatory'] for row in control_valid])) if control_valid else np.nan, 'baseline_oscillation_probability': float(np.mean([row['baseline_osc'] for row in baseline_valid])) if baseline_valid else np.nan, 'n_total': int(len(rows)), 'n_valid_pairs': int(len(paired_valid)), 'n_invalid_baseline': int(len(rows) - len(baseline_valid)), 'n_invalid_control': int(len(rows) - len(control_valid))}

def tail_amplitude(series):
    st, _ = tail(np.asarray(series))
    return float(np.max(st) - np.min(st))

def normalized_collective_error(xw, yw, xr, yr):
    xw = np.asarray(xw, dtype=float)
    yw = np.asarray(yw, dtype=float)
    xr = np.asarray(xr, dtype=float)
    yr = np.asarray(yr, dtype=float)
    denom = max(float(np.sqrt(np.mean(xr ** 2 + yr ** 2))), 1e-12)
    return np.sqrt((xw - xr) ** 2 + (yw - yr) ** 2) / denom

def est_period(series, t):
    st, n0 = tail(series)
    md = max(1, int(len(st) * peak_dist_ratio))
    st_peak = _smooth_for_peak_detection(st)
    pks, _ = find_peaks(st_peak, prominence=peak_prom, distance=md)
    if len(pks) < 2:
        return {'valid': False, 'T_mean': np.nan}
    pt = t[n0:][pks]
    periods = np.diff(pt)
    return {'valid': True, 'T_mean': float(np.mean(periods)), 'T_std': float(np.std(periods)), 'pt': pt}

def run_sim(A, z0, seed_shift=0, z_vec=None, use_z_heter=True):
    n = A.shape[0]
    w = degree_weights(A)
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    x0, y0 = make_ic(n, ER_SEED + 10000 + seed_shift)
    if z_vec is None:
        zv = make_z_vec(z0, w, ER_SEED + 20000 + seed_shift) if use_z_heter else np.full(n, z0)
    else:
        zv = np.asarray(z_vec, dtype=float).copy()
    z_eff = z_eff_value(zv, w)
    av, bv, cv = make_abc(n, w, ER_SEED + 30000 + seed_shift)
    print(f'  Sim N={n}, z_eff={z_eff:.4f}, Ae={Ae:.6f}...')
    sn = rk4_net(A, x0, y0, zv, t_eval, av, bv, cv)
    xw = weighted_series(sn['x_all'], w)
    yw = weighted_series(sn['y_all'], w)
    sr = rk4_red(xw[0], yw[0], z_eff, Ae, t_eval)
    return {'sn': sn, 'sr': sr, 'xw': xw, 'yw': yw, 'w': w, 'Ae': Ae, 'theory': theory, 'z0': z0, 'z_eff': z_eff, 'A': A}

def run_sim_collective(A, z0, seed_shift=0, z_vec=None, use_z_heter=True):
    n = A.shape[0]
    w = degree_weights(A)
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    x0, y0 = make_ic(n, ER_SEED + 10000 + seed_shift)
    if z_vec is None:
        zv = make_z_vec(z0, w, ER_SEED + 20000 + seed_shift) if use_z_heter else np.full(n, z0)
    else:
        zv = np.asarray(z_vec, dtype=float).copy()
    z_eff = z_eff_value(zv, w)
    av, bv, cv = make_abc(n, w, ER_SEED + 30000 + seed_shift)
    sn = rk4_net_collective(A, x0, y0, zv, t_eval, w, av, bv, cv)
    sr = rk4_red(sn['xw'][0], sn['yw'][0], z_eff, Ae, t_eval)
    return {'sn': sn, 'sr': sr, 'xw': sn['xw'], 'yw': sn['yw'], 'w': w, 'Ae': Ae, 'theory': theory, 'z0': z0, 'z_eff': z_eff, 'A': A}

def node_colors(N):
    cm = matplotlib.colormaps.get_cmap('Spectral').resampled(N)
    return [cm(i) for i in range(N)]

def _plot_nodes_phase(ax, xa, ya, N, nc):
    for i in range(N):
        ax.plot(xa[:, i], ya[:, i], color=nc[i], alpha=ALPHA_NODES, lw=LW_NODES)

def _plot_nodes_time(ax, t, sa, N, nc):
    for i in range(N):
        ax.plot(t, sa[:, i], color=nc[i], alpha=ALPHA_NODES, lw=LW_NODES)

def normalized_node_reduced_error(xa, ya, xr, yr):
    xa = np.asarray(xa, dtype=float)
    ya = np.asarray(ya, dtype=float)
    xr = np.asarray(xr, dtype=float)
    yr = np.asarray(yr, dtype=float)
    denom = max(float(np.sqrt(np.mean(xr ** 2 + yr ** 2))), 1e-12)
    return np.sqrt((xa - xr[:, None]) ** 2 + (ya - yr[:, None]) ** 2) / denom

def plot_error_structure(ax, t, e_nodes, e_eff, nc, title=None):
    n_nodes = e_nodes.shape[1]
    for i in range(n_nodes):
        ax.plot(t, e_nodes[:, i], color=nc[i], alpha=0.1, lw=0.45, zorder=1)
    e_node_mean = np.mean(e_nodes, axis=1)
    ax.plot(t, e_node_mean, color=COLOR_NODE_MEAN, lw=1.8, label='$\\langle E_i(t)\\rangle$', zorder=3)
    ax.plot(t, e_eff, color=COLOR_ERR_MEAN, lw=2.0, label='$E_{\\mathrm{eff}}(t)$', zorder=4)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Normalized distance')
    if title:
        ax.set_title(title, fontsize=8)
    finite_errors = np.r_[e_nodes.ravel(), e_eff]
    finite_errors = finite_errors[np.isfinite(finite_errors)]
    if finite_errors.size:
        ax.set_ylim(0, float(np.max(finite_errors)) * 1.05)
    handles = [Line2D([0], [0], color=CMAP_NODES(0.55), lw=0.8, alpha=0.7, label='$E_i(t)$'), Line2D([0], [0], color=COLOR_NODE_MEAN, lw=1.8, label='$\\langle E_i(t)\\rangle$'), Line2D([0], [0], color=COLOR_ERR_MEAN, lw=2.0, label='$E_{\\mathrm{eff}}(t)$')]
    legend_clean(ax, handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3)
    return {'node_mean_final': float(np.mean(final_window(e_node_mean)[0])), 'eff_final': float(np.mean(final_window(e_eff)[0])), 'node_mean_early': float(np.mean(e_node_mean[:max(1, int(0.2 * len(e_node_mean)))])), 'eff_early': float(np.mean(e_eff[:max(1, int(0.2 * len(e_eff)))]))}

def _cycle_aligned_reduction_error(t, xo, yo, xw, yw, xr, yr, w, n_phase=320):

    def peak_idx(series):
        st, n0 = tail(series)
        if len(st) == 0 or not np.all(np.isfinite(st)):
            return np.array([], dtype=int)
        md = max(1, int(len(st) * peak_dist_ratio))
        pks, _ = find_peaks(_smooth_for_peak_detection(st), prominence=peak_prom, distance=md)
        return pks + n0
    p_net = peak_idx(xw)
    p_red = peak_idx(xr)
    theta = np.linspace(0.0, 1.0, int(n_phase))
    if len(p_net) < 2 or len(p_red) < 2:
        rms = max(float(np.sqrt(np.mean(xr ** 2 + yr ** 2))), 1e-12)
        e_raw = np.sqrt((xw - xr) ** 2 + (yw - yr) ** 2) / rms
        disp_raw = np.sqrt(((xo - xw[:, None]) ** 2 + (yo - yw[:, None]) ** 2) @ w) / rms
        return (theta, np.interp(theta, np.linspace(0, 1, len(e_raw)), e_raw), np.interp(theta, np.linspace(0, 1, len(disp_raw)), disp_raw), {'mode': 'fallback_raw_time', 'mean_error': float(np.mean(e_raw)), 'max_error': float(np.max(e_raw))})
    n0, n1 = (int(p_net[-2]), int(p_net[-1]))
    r0, r1 = (int(p_red[-2]), int(p_red[-1]))
    tn = np.linspace(t[n0], t[n1], int(n_phase))
    tr = np.linspace(t[r0], t[r1], int(n_phase))
    xwn = np.interp(tn, t, xw)
    ywn = np.interp(tn, t, yw)
    xrr = np.interp(tr, t, xr)
    yrr = np.interp(tr, t, yr)
    rms = max(float(np.sqrt(np.mean(xrr ** 2 + yrr ** 2))), 1e-12)
    e_cycle = np.sqrt((xwn - xrr) ** 2 + (ywn - yrr) ** 2) / rms
    cycle_center_network = np.array([np.mean(xwn), np.mean(ywn)], dtype=float)
    cycle_center_reduced = np.array([np.mean(xrr), np.mean(yrr)], dtype=float)
    center_shift = float(np.linalg.norm(cycle_center_network - cycle_center_reduced) / rms)
    dx_centered = xwn - cycle_center_network[0] - (xrr - cycle_center_reduced[0])
    dy_centered = ywn - cycle_center_network[1] - (yrr - cycle_center_reduced[1])
    centered_shape_error = np.sqrt(dx_centered ** 2 + dy_centered ** 2) / rms
    disp = np.zeros_like(theta)
    for j, tj in enumerate(tn):
        x_nodes = np.array([np.interp(tj, t, xo[:, i]) for i in range(xo.shape[1])])
        y_nodes = np.array([np.interp(tj, t, yo[:, i]) for i in range(yo.shape[1])])
        disp[j] = np.sqrt(np.dot(w, (x_nodes - xwn[j]) ** 2 + (y_nodes - ywn[j]) ** 2)) / rms
    return (theta, e_cycle, disp, {'mode': 'tail_cycle_phase_aligned', 'network_cycle': [float(t[n0]), float(t[n1])], 'reduced_cycle': [float(t[r0]), float(t[r1])], 'mean_error': float(np.mean(e_cycle)), 'max_error': float(np.max(e_cycle)), 'mean_dispersion': float(np.mean(disp)), 'max_dispersion': float(np.max(disp)), 'cycle_center_network': [float(v) for v in cycle_center_network], 'cycle_center_reduced': [float(v) for v in cycle_center_reduced], 'normalized_center_shift': center_shift, 'centered_shape_mean_error': float(np.mean(centered_shape_error))})

def _weighted_pearson(x, y, weights):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return np.nan
    x = x[mask]
    y = y[mask]
    weights = weights[mask]
    total = float(np.sum(weights))
    if total <= 1e-14:
        return np.nan
    weights = weights / total
    mx = float(np.dot(weights, x))
    my = float(np.dot(weights, y))
    dx = x - mx
    dy = y - my
    vx = float(np.dot(weights, dx * dx))
    vy = float(np.dot(weights, dy * dy))
    if vx <= 1e-20 or vy <= 1e-20:
        return np.nan
    return float(np.dot(weights, dx * dy) / np.sqrt(vx * vy))

def _directed_degree_correlation_audit(A):
    A = np.asarray(A, dtype=float)
    dst, src = np.nonzero(A)
    if len(src) == 0:
        return {'edge_count': 0, 'degree_unweighted': {}, 'strength_edge_weighted': {}}
    edge_weight = A[dst, src]
    unit_weight = np.ones(len(src), dtype=float)
    in_degree = np.count_nonzero(A, axis=1).astype(float)
    out_degree = np.count_nonzero(A, axis=0).astype(float)
    in_strength = np.sum(A, axis=1)
    out_strength = np.sum(A, axis=0)

    def four_correlations(source_in, source_out, target_in, target_out, weights):
        return {'source_out__target_in': _weighted_pearson(source_out[src], target_in[dst], weights), 'source_in__target_in': _weighted_pearson(source_in[src], target_in[dst], weights), 'source_out__target_out': _weighted_pearson(source_out[src], target_out[dst], weights), 'source_in__target_out': _weighted_pearson(source_in[src], target_out[dst], weights)}
    return {'edge_count': int(len(src)), 'orientation': 'A[i,j] > 0 means source j -> target i', 'primary_correlation': 'source_out__target_in', 'degree_unweighted': four_correlations(in_degree, out_degree, in_degree, out_degree, unit_weight), 'strength_edge_weighted': four_correlations(in_strength, out_strength, in_strength, out_strength, edge_weight)}

def topology_reduction_assumption_audit(A, w, Ae):
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    left_action = np.asarray(w @ A, dtype=float)
    reduced_action = float(Ae) * w
    residual = left_action - reduced_action
    l2_scale = max(float(np.linalg.norm(left_action)), 1e-14)
    linf_scale = max(float(np.max(np.abs(left_action))), 1e-14)
    in_strength = np.sum(A, axis=1)
    mean_in_strength = float(np.mean(in_strength))
    observed_cv = float(np.std(in_strength) / mean_in_strength) if mean_in_strength > 1e-14 else np.nan
    neff = float(1.0 / max(np.dot(w, w), 1e-14))
    return {'closure_identity': 'w^T A approximately equals A_eff w^T', 'left_action_relative_residual_l2': float(np.linalg.norm(residual) / l2_scale), 'left_action_relative_residual_linf': float(np.max(np.abs(residual)) / linf_scale), 'effective_node_number': neff, 'effective_node_fraction': float(neff / max(len(w), 1)), 'observed_in_strength_cv': observed_cv, 'degree_correlations': _directed_degree_correlation_audit(A)}

def _finite_distribution_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {'mean': None, 'p95': None, 'max': None, 'rms': None}
    return {'mean': float(np.mean(values)), 'p95': float(np.quantile(values, 0.95)), 'max': float(np.max(values)), 'rms': float(np.sqrt(np.mean(values * values)))}

def _assumption_series_stats(values, tail_start):
    values = np.asarray(values, dtype=float)
    tail_start = int(np.clip(tail_start, 0, max(len(values) - 1, 0)))
    return {'all': _finite_distribution_stats(values), 'post_transient': _finite_distribution_stats(values[tail_start:])}

def _globally_normalized_residual(lhs, rhs, tail_start):
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    tail_start = int(np.clip(tail_start, 0, max(len(lhs) - 1, 0)))
    lhs_tail = lhs[tail_start:]
    rhs_tail = rhs[tail_start:]
    scale = max(float(np.sqrt(np.mean(lhs_tail * lhs_tail))), float(np.sqrt(np.mean(rhs_tail * rhs_tail))), 1e-14)
    return (np.abs(lhs - rhs) / scale, float(scale))

def dynamic_reduction_assumption_audit(data):
    t = np.asarray(data['sn']['t'], dtype=float)
    xo = np.asarray(data['sn']['x_all'], dtype=float)
    yo = np.asarray(data['sn']['y_all'], dtype=float)
    w = np.asarray(data['w'], dtype=float)
    X = np.asarray(data['xw'], dtype=float)
    Y = np.asarray(data['yw'], dtype=float)
    A = np.asarray(data['A'], dtype=float)
    Ae = float(data['Ae'])
    tail_start = int(discard_ratio * len(t))
    Lx2 = np.einsum('ti,i->t', xo * xo, w, optimize=True)
    Ly2 = np.einsum('ti,i->t', yo * yo, w, optimize=True)
    Lxy = np.einsum('ti,i->t', xo * yo, w, optimize=True)
    Lx3 = np.einsum('ti,i->t', xo * xo * xo, w, optimize=True)
    eps_x2, scale_x2 = _globally_normalized_residual(Lx2, X * X, tail_start)
    eps_xy, scale_xy = _globally_normalized_residual(Lxy, X * Y, tail_start)
    eps_x3, scale_x3 = _globally_normalized_residual(Lx3, X * X * X, tail_start)
    left_action = np.asarray(w @ A, dtype=float)
    LAx = xo @ left_action
    eps_A, scale_A = _globally_normalized_residual(LAx, Ae * X, tail_start)
    state_scale = max(float(np.sqrt(np.mean(X[tail_start:] ** 2 + Y[tail_start:] ** 2))), 1e-14)
    weighted_state_variance = np.maximum(Lx2 - X * X + (Ly2 - Y * Y), 0.0)
    synchronization_dispersion = np.sqrt(weighted_state_variance) / state_scale
    L_combo = np.einsum('ti,i->t', 2.0 * xo - yo, w, optimize=True)
    eps_linear, scale_linear = _globally_normalized_residual(L_combo, 2.0 * X - Y, tail_start)
    series = {'linearity': eps_linear, 'hadamard_x2': eps_x2, 'hadamard_xy': eps_xy, 'fhn_cubic_x3': eps_x3, 'network_A': eps_A, 'synchronization': synchronization_dispersion}
    summary = {'tail_start_index': int(tail_start), 'tail_start_time': float(t[tail_start]), 'linearity': {'identity': 'L(2x-y)=2L(x)-L(y)', 'weight_sum_error': float(abs(np.sum(w) - 1.0)), 'normalization_scale': scale_linear, 'residual': _assumption_series_stats(eps_linear, tail_start), 'status': 'exact_by_construction_up_to_floating_point_roundoff'}, 'hadamard_closure': {'normalization': 'absolute residual divided by the larger post-transient RMS of the two compared terms', 'x_squared': {'identity': 'L(x^2) approximately equals L(x)^2', 'normalization_scale': scale_x2, 'residual': _assumption_series_stats(eps_x2, tail_start)}, 'x_times_y': {'identity': 'L(x*y) approximately equals L(x)L(y)', 'normalization_scale': scale_xy, 'residual': _assumption_series_stats(eps_xy, tail_start)}, 'fhn_cubic': {'identity': 'L(x^3) approximately equals L(x)^3', 'exact_difference': '3*L(x)*Var_w(x)+weighted_third_central_moment', 'normalization_scale': scale_x3, 'residual': _assumption_series_stats(eps_x3, tail_start)}}, 'network_closure': {'identity': 'L(Ax) approximately equals A_eff L(x)', 'normalization_scale': scale_A, 'dynamic_residual': _assumption_series_stats(eps_A, tail_start)}, 'macroscopic_synchronization': {'definition': 'sqrt(L((x-X)^2+(y-Y)^2))/post-transient RMS(X,Y)', 'normalization_scale': state_scale, 'dispersion': _assumption_series_stats(synchronization_dispersion, tail_start)}}
    return (series, summary)

def _plot_assumption_residuals(ax, t, series, summary, title, show_legend=False):
    start = int(summary['tail_start_index'])
    tt = np.asarray(t, dtype=float)[start:]
    curves = [('fhn_cubic_x3', '$\\epsilon_{x^3}$ cubic closure', COLOR_XW), ('network_A', '$\\epsilon_A$ network closure', COLOR_XEFF), ('synchronization', '$D_{\\mathrm{sync}}$', COLOR_EMP_FP)]
    for key, label, color in curves:
        values = np.maximum(np.asarray(series[key], dtype=float)[start:], 1e-10)
        mean_value = _finite_distribution_stats(values)['mean']
        mean_text = 'nan' if mean_value is None else f'{mean_value:.2g}'
        ax.plot(tt, values, lw=1.15, color=color, label=f'{label} (mean {mean_text})')
    ax.set_yscale('log')
    ax.set_ylabel('Normalized residual')
    ax.set_title(title, fontsize=8)
    if show_legend:
        legend_clean(ax, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=6.0)

def _save_assumption_source_data(path, t, osc_series, stable_series, osc_summary, stable_summary):
    t = np.asarray(t, dtype=float)

    def regime_rows(code, series, summary):
        tail_flag = (np.arange(len(t)) >= int(summary['tail_start_index'])).astype(float)
        return np.column_stack([np.full(len(t), float(code)), t, tail_flag, series['linearity'], series['hadamard_x2'], series['hadamard_xy'], series['fhn_cubic_x3'], series['network_A'], series['synchronization']])
    table = np.vstack([regime_rows(1, osc_series, osc_summary), regime_rows(0, stable_series, stable_summary)])
    np.savetxt(path, table, delimiter=',', header='regime_code,time,post_transient,linearity_residual,hadamard_x2_residual,hadamard_xy_residual,fhn_cubic_x3_residual,network_A_residual,synchronization_dispersion', comments='')

def plot_fig2(data_osc, data_stable, sd):
    N = data_osc['sn']['x_all'].shape[1]
    t = data_osc['sn']['t']
    nc = node_colors(N)
    Ae = data_osc['Ae']
    xo, yo = (data_osc['sn']['x_all'], data_osc['sn']['y_all'])
    xwo, ywo = (data_osc['xw'], data_osc['yw'])
    xro, yro = (data_osc['sr']['x'], data_osc['sr']['y'])
    xs, ys = (data_stable['sn']['x_all'], data_stable['sn']['y_all'])
    xws, yws = (data_stable['xw'], data_stable['yw'])
    xrs, yrs = (data_stable['sr']['x'], data_stable['sr']['y'])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    _plot_nodes_time(ax0, t, xo, N, nc)
    ax0.plot(t, xwo, color=COLOR_XW, lw=LW_XW, label='$\\bar{x}_w(t)$')
    ax0.plot(t, xro, color=COLOR_XEFF, lw=LW_XEFF, ls=LS_XEFF, label='Reduced')
    ax0.set_ylabel('$x$')
    legend_clean(ax0, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2)
    _plot_nodes_time(ax1, t, yo, N, nc)
    ax1.plot(t, ywo, color=COLOR_XW, lw=LW_XW, label='$\\bar{y}_w(t)$')
    ax1.plot(t, yro, color=COLOR_XEFF, lw=LW_XEFF, ls=LS_XEFF, label='Reduced')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$y$')
    save_sub(fig, sd, 'fig2a')
    fig, ax = mfig((6, 5))
    _plot_nodes_phase(ax, xo, yo, N, nc)
    ax.plot(xwo, ywo, color=COLOR_XW, lw=LW_XW, label='$\\bar{x}_w,\\bar{y}_w$', zorder=3)
    ax.plot(xro, yro, color=COLOR_XEFF, lw=LW_XEFF, ls=LS_XEFF, label='Reduced', zorder=4)
    ax.text(0.03, 0.04, f'$z_i=-a/b={OSCILLATORY_Z:.3f}$', transform=ax.transAxes, fontsize=7, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.72))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    legend_clean(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    save_sub(fig, sd, 'fig2b')
    _theta, _cycle_error, _cycle_dispersion, cycle_info = _cycle_aligned_reduction_error(t, xo, yo, xwo, ywo, xro, yro, data_osc['w'])
    err_o = normalized_collective_error(xwo, ywo, xro, yro)
    node_err_o = normalized_node_reduced_error(xo, yo, xro, yro)
    fig, ax = mfig((7.4, 3.7))
    err_o_time_info = plot_error_structure(ax, t, node_err_o, err_o, nc, title='Oscillatory regime (pointwise; phase drift retained)')
    ax.text(0.99, 0.95, f"$\\langle E_{{\\mathrm{{eff}}}}\\rangle_{{final}}={err_o_time_info['eff_final']:.3g}$" + '\n' + f"$\\langle E_i\\rangle_{{final}}={err_o_time_info['node_mean_final']:.3g}$" + '\n' + f"$\\langle E_{{\\mathrm{{cycle}}}}\\rangle={cycle_info['mean_error']:.3g}$", transform=ax.transAxes, fontsize=7, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.82))
    save_sub(fig, sd, 'fig2c')
    save_json(cycle_info, os.path.join(sd, 'fig2_cycle_error.json'))
    err_o_info = {'pointwise_time': err_o_time_info, 'phase_aligned_cycle': cycle_info}
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    fps = eff_fixed_points(Ae, data_stable['z_eff'])
    tfp = next((f for f in fps if f['stable']), None)
    _plot_nodes_time(ax0, t, xs, N, nc)
    ax0.plot(t, xws, color=COLOR_XW, lw=LW_XW, label='$\\bar{x}(t)$')
    ax0.plot(t, xrs, color=COLOR_XEFF, lw=LW_XEFF, ls=LS_XEFF, label='Reduced')
    if tfp:
        ax0.axhline(tfp['x'], color=COLOR_THEORY_FP, ls=':', lw=1, alpha=0.7)
    ax0.set_ylabel('$x$')
    legend_clean(ax0, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2)
    _plot_nodes_time(ax1, t, ys, N, nc)
    ax1.plot(t, yws, color=COLOR_XW, lw=LW_XW, label='$\\bar{y}(t)$')
    ax1.plot(t, yrs, color=COLOR_XEFF, lw=LW_XEFF, ls=LS_XEFF, label='Reduced')
    if tfp:
        ax1.axhline(tfp['y'], color=COLOR_THEORY_FP, ls=':', lw=1, alpha=0.7)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$y$')
    save_sub(fig, sd, 'fig2d')
    fps = eff_fixed_points(Ae, data_stable['z_eff'])
    tfp = next((f for f in fps if f['stable']), None)
    xws_final, n0 = final_window(xws)
    yws_final, _ = final_window(yws)
    xrs_final, _ = final_window(xrs)
    yrs_final, _ = final_window(yrs)
    efp = {'x': float(np.mean(xws_final)), 'y': float(np.mean(yws_final))}
    fp_err = np.nan
    fig, ax = mfig((6, 5))
    _plot_nodes_phase(ax, xs, ys, N, nc)
    ax.plot(xws, yws, color=COLOR_XW, lw=LW_XW, label='Network mean', zorder=3)
    ax.plot(xrs, yrs, color=COLOR_XEFF, lw=LW_XEFF, ls=LS_XEFF, label='Reduced', zorder=4)
    ax.plot(xws_final, yws_final, color=COLOR_XW, lw=LW_XW + 0.6, alpha=0.95, zorder=5)
    ax.plot(xrs_final, yrs_final, color=COLOR_XEFF, lw=LW_XEFF + 0.5, ls=LS_XEFF, alpha=0.95, zorder=5)
    if tfp:
        ax.scatter(tfp['x'], tfp['y'], marker='*', s=200, c=COLOR_THEORY_FP, edgecolors='k', lw=0.8, zorder=5, label='Theory FP')
    ax.scatter(efp['x'], efp['y'], marker='o', s=100, c=COLOR_EMP_FP, edgecolors='k', lw=0.8, zorder=5, label='Sim. FP')
    if tfp:
        fp_err = float(np.sqrt((efp['x'] - tfp['x']) ** 2 + (efp['y'] - tfp['y']) ** 2))
        ax.text(0.03, 0.04, f"$u_{{\\mathrm{{eff}}}}={data_stable['z_eff']:.3f}$" + '\n' + f"Theory FP=({tfp['x']:.2f},{tfp['y']:.2f})" + '\n' + f"Sim. FP=({efp['x']:.2f},{efp['y']:.2f})" + '\n' + f'$E_{{\\mathrm{{FP}}}}={fp_err:.3g}$', transform=ax.transAxes, fontsize=6, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.72))
        ax.annotate(f"({tfp['x']:.2f},{tfp['y']:.2f})", xy=(tfp['x'], tfp['y']), xytext=(8, 10), textcoords='offset points', fontsize=5.8, color=COLOR_THEORY_FP, arrowprops=dict(arrowstyle='-', color=COLOR_THEORY_FP, lw=0.45, shrinkA=0, shrinkB=4))
        ax.annotate(f"({efp['x']:.2f},{efp['y']:.2f})", xy=(efp['x'], efp['y']), xytext=(8, -13), textcoords='offset points', fontsize=5.8, color=COLOR_EMP_FP, arrowprops=dict(arrowstyle='-', color=COLOR_EMP_FP, lw=0.45, shrinkA=0, shrinkB=4))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    legend_clean(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    save_sub(fig, sd, 'fig2e')
    err_s = normalized_collective_error(xws, yws, xrs, yrs)
    node_err_s = normalized_node_reduced_error(xs, ys, xrs, yrs)
    fig, ax = mfig((7.4, 3.7))
    err_s_info = plot_error_structure(ax, t, node_err_s, err_s, nc, title='Non-oscillatory regime')
    ax.text(0.99, 0.95, f"$\\langle E_{{\\mathrm{{eff}}}}\\rangle_{{final}}={err_s_info['eff_final']:.3g}$" + '\n' + f"$\\langle E_i\\rangle_{{final}}={err_s_info['node_mean_final']:.3g}$", transform=ax.transAxes, fontsize=7, va='top', ha='right', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.82))
    save_sub(fig, sd, 'fig2f')
    save_json({'mode': 'symmetric_pointwise_time_error_plus_phase_aligned_cycle_diagnostic', 'normalization': 'sqrt(mean(x_reduced^2 + y_reduced^2))', 'oscillatory': err_o_info, 'nonoscillatory': err_s_info, 'fixed_point_method': 'theory fixed points solve the biased reduced cubic; the simulation fixed point is the mean over the final 20% of weighted network variables', 'theory_fixed_point': tfp, 'simulation_fixed_point': efp, 'fixed_point_error': float(fp_err) if np.isfinite(fp_err) else None}, os.path.join(sd, 'fig2_time_error.json'))
    print(f'  Computing period/amplitude ratio diagnostics for fig2g/fig2i ({N_WORKERS} workers)...')
    info = data_osc['theory']
    zl, zr = (info['z_left'], info['z_right'])
    mg = 0.05 * (zr - zl)
    zlo = max(zl + mg, OSCILLATORY_Z - FIG2_DIAG_Z_HALF_WIDTH)
    zhi = min(zr - mg, OSCILLATORY_Z + FIG2_DIAG_Z_HALF_WIDTH)
    if zlo >= zhi:
        zlo, zhi = (zl + mg, zr - mg)
    zpts = np.linspace(zlo, zhi, N_Z_PERIOD)
    tasks_pa = [(data_osc['A'], zi, int(abs(zi) * 1000) % 9999) for zi in zpts]
    chunksize = max(1, len(tasks_pa) // max(1, N_WORKERS * 4))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        scan = list(pool.map(_worker_dimred_scan, tasks_pa, chunksize=chunksize))
    save_json(scan, os.path.join(sd, 'fig2_parity_metrics.json'))
    save_json({'period_method': "Slow-fast relaxation-oscillator approximation: integrate c*|h'(x)/(x-a+b*h(x))| along the two outer branches of the FHN cubic critical manifold, where h(x)=x^3/3-(1+A_eff)x-z_eff.", 'amplitude_method': 'Slow-fast fold-jump approximation: x-amplitude is the distance between the two outer landing points obtained by horizontal jumps from the cubic fold points x=+-sqrt(1+A_eff).', 'comparison': 'Fig2g/i plot theory/network ratios. Reduced-model numerical period and amplitude are retained in fig2_parity_metrics.json.'}, os.path.join(sd, 'fig2_theory_method.json'))
    period_rows = [r for r in scan if r.get('theory_period_valid') and np.isfinite(r['T_net']) and np.isfinite(r.get('T_theory_relax', np.nan))]
    amp_rows = [r for r in scan if r.get('theory_amplitude_valid') and np.isfinite(r['A_net']) and np.isfinite(r.get('A_theory_relax', np.nan)) and (r['A_net'] > 1e-12)]
    if len(period_rows) >= 2:
        fig, axp = mfig((5.2, 3.8))
        zc = np.array([r['z0'] for r in period_rows])
        pr = np.array([r['period_ratio_theory_net'] for r in period_rows])
        pr_red = np.array([r.get('period_ratio', np.nan) for r in period_rows])
        order = np.argsort(zc)
        zc = zc[order]
        pr = pr[order]
        pr_red = pr_red[order]
        axp.axhline(1.0, color='#555555', ls='--', lw=0.9)
        if np.any(np.isfinite(pr_red)):
            axp.plot(zc, pr_red, color='#9AA8B4', lw=0.9, ls=':', label='reduced numerical', zorder=1)
        axp.plot(zc, pr, color=COLOR_XEFF, lw=1.2, label='slow-fast theory', zorder=2)
        axp.scatter(zc, pr, c=zc, cmap='viridis', s=30, edgecolors='k', lw=0.25, zorder=3)
        axp.set_title('Slow-fast period theory', fontsize=8)
        axp.set_xlabel('$z_{\\mathrm{eff}}$')
        axp.set_ylabel('$T_{\\mathrm{theory}}/T_{\\mathrm{net}}$')
        if np.any(np.isfinite(pr)):
            combined = np.r_[pr, pr_red[np.isfinite(pr_red)]]
            lo = max(0.0, min(0.75, float(np.nanmin(combined)) - 0.08))
            hi = max(1.22, float(np.nanmax(combined)) + 0.08)
            axp.set_ylim(lo, hi)
        axp.text(0.04, 0.08, f'mean={np.nanmean(pr):.3g}', transform=axp.transAxes, fontsize=6.5)
        legend_clean(axp, loc='lower right', fontsize=6.0)
        save_sub(fig, sd, 'fig2g')
    osc_assumption_series, osc_assumption_summary = dynamic_reduction_assumption_audit(data_osc)
    stable_assumption_series, stable_assumption_summary = dynamic_reduction_assumption_audit(data_stable)
    topology_assumption_summary = topology_reduction_assumption_audit(data_osc['A'], data_osc['w'], data_osc['Ae'])
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    _plot_assumption_residuals(axes[0], t, osc_assumption_series, osc_assumption_summary, 'Oscillatory regime: post-transient assumption residuals', show_legend=True)
    _plot_assumption_residuals(axes[1], t, stable_assumption_series, stable_assumption_summary, 'Non-oscillatory regime: post-transient assumption residuals')
    axes[1].set_xlabel('Time $t$')
    rho_main = topology_assumption_summary['degree_correlations']['degree_unweighted'].get('source_out__target_in', np.nan)
    axes[1].text(0.99, 0.05, f"$\\epsilon_A^{{\\mathrm{{static}}}}={topology_assumption_summary['left_action_relative_residual_l2']:.3g}$\n$\\rho_{{k_{{out}}^{{src}},k_{{in}}^{{dst}}}}={rho_main:.3g}$\n$N_{{\\mathrm{{eff}}}}/N={topology_assumption_summary['effective_node_fraction']:.3g}$", transform=axes[1].transAxes, ha='right', va='bottom', fontsize=6.5, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.82))
    plt.close(fig)
    assumption_payload = {'scope': 'Diagnostics of exact operator linearity, approximate Hadamard closure, macroscopic synchronization, network-term closure, and directed degree correlations.', 'no_binary_pass_fail': 'Residual magnitudes are reported without a universal acceptance threshold; fast-jump maxima must be interpreted separately from post-transient means and 95th percentiles.', 'oscillatory': osc_assumption_summary, 'nonoscillatory': stable_assumption_summary, 'topology': topology_assumption_summary}
    save_json(assumption_payload, os.path.join(sd, 'fig2_assumption_audit.json'))
    _save_assumption_source_data(os.path.join(sd, 'fig2_assumption_audit.csv'), t, osc_assumption_series, stable_assumption_series, osc_assumption_summary, stable_assumption_summary)
    if len(amp_rows) >= 2:
        fig, axa = mfig((5.2, 3.8))
        zc = np.array([r['z0'] for r in amp_rows])
        ar = np.array([r['amplitude_ratio_theory_net'] for r in amp_rows])
        ar_red = np.array([r.get('amplitude_ratio', np.nan) for r in amp_rows])
        order = np.argsort(zc)
        zc = zc[order]
        ar = ar[order]
        ar_red = ar_red[order]
        axa.axhline(1.0, color='#555555', ls='--', lw=0.9)
        if np.any(np.isfinite(ar_red)):
            axa.plot(zc, ar_red, color='#9AA8B4', lw=0.9, ls=':', label='reduced numerical', zorder=1)
        axa.plot(zc, ar, color=COLOR_XW, lw=1.2, label='slow-fast theory', zorder=2)
        axa.scatter(zc, ar, c=zc, cmap='viridis', s=30, edgecolors='k', lw=0.25, zorder=3)
        axa.set_title('Slow-fast amplitude theory', fontsize=8)
        axa.set_xlabel('$z_{\\mathrm{eff}}$')
        axa.set_ylabel('$A_{\\mathrm{theory}}/A_{\\mathrm{net}}$')
        if np.any(np.isfinite(ar)):
            combined = np.r_[ar, ar_red[np.isfinite(ar_red)]]
            lo = max(0.0, min(0.85, float(np.nanmin(combined)) - 0.1))
            hi = max(1.25, float(np.nanmax(combined)) + 0.1)
            axa.set_ylim(lo, hi)
        axa.text(0.04, 0.08, f'mean={np.nanmean(ar):.3g}', transform=axa.transAxes, fontsize=6.5)
        legend_clean(axa, loc='lower right', fontsize=6.0)
        save_sub(fig, sd, 'fig2i')

def run_fig2(A):
    print('\n[Fig 2] ER benchmark dimensionality reduction validation...')
    sd = DIRS['fig2']
    cleanup_outputs(sd, [f'fig2{c}' for c in 'abcdefghij'] + ['fig2_cycle_error', 'fig2_time_error', 'fig2_assumption_audit'])
    A_used, net_info = benchmark_operating_point_info(A, 'fig2_er')
    w = degree_weights(A_used)
    theory = derive_z_interval(effective_coupling(A_used))
    stable_margin = stable_target_margin_info(theory)
    z_nonoscillatory = float(FIG2_STABLE_Z)
    stable_right_threshold = float(theory['z_right'] + stable_margin['absolute_buffer'])
    if z_nonoscillatory < stable_right_threshold:
        raise ValueError('FIG2_STABLE_Z must lie to the right of the analytical oscillatory interval plus its heterogeneity buffer.')
    z0_vec = make_weighted_mean_z_vec(OSCILLATORY_Z, w, ER_SEED + 40000)
    znonoscillatory_vec = make_weighted_mean_z_vec(z_nonoscillatory, w, ER_SEED + 50000)
    print(f"  Ae={net_info['A_eff']:.6f}, interval=[{theory['z_left']:.4f},{theory['z_right']:.4f}]")
    print(f'  z_osc=-a/b={OSCILLATORY_Z:.4f}, z_nonoscillatory_eff={z_nonoscillatory:.4f}')
    save_json({'network_operating_point': net_info, 'control_coordinate': 'direct additive FHN input z', 'z_distribution': 'Gaussian perturbation followed by weighted recentering', 'z_std_target': float(Z_STD), 'z_oscillatory_eff': float(OSCILLATORY_Z), 'z_baseline_eff': float(OSCILLATORY_Z), 'z_oscillatory_weighted_error': float(np.dot(w, z0_vec) - OSCILLATORY_Z), 'z_oscillatory_std': float(np.std(z0_vec - OSCILLATORY_Z)), 'z_nonoscillatory_eff': z_nonoscillatory, 'stable_target_selection': 'fixed zero input verified analytically', 'stable_target_margin': stable_margin, 'stable_right_threshold': stable_right_threshold, 'stable_distance_from_oscillatory_boundary': float(z_nonoscillatory - theory['z_right']), 'z_nonoscillatory_weighted_error': float(np.dot(w, znonoscillatory_vec) - z_nonoscillatory), 'z_nonoscillatory_std': float(np.std(znonoscillatory_vec - z_nonoscillatory))}, os.path.join(sd, 'fig2_control_design.json'))
    data_osc = run_sim(A_used, OSCILLATORY_Z, FIG2_SEED_SHIFT, z_vec=z0_vec)
    data_nonoscillatory = run_sim(A_used, z_nonoscillatory, FIG2_SEED_SHIFT + 100, z_vec=znonoscillatory_vec)
    plot_fig2(data_osc, data_nonoscillatory, sd)
    print('[Fig 2] Done.\n')

def _worker_zscan(args):
    A, zi, seed_shift = args
    out = run_sim_collective(A, zi, seed_shift=seed_shift)
    det = detect_osc(out['xw'], out['sn']['t'])
    if not det.get('valid', True):
        return (zi, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False)
    x_tail, _ = tail(out['xw'])
    y_tail, _ = tail(out['yw'])
    return (zi, 1.0 if det['is_osc'] else 0.0, det['amp'], float(np.min(x_tail)), float(np.max(x_tail)), float(np.min(y_tail)), float(np.max(y_tail)), True)

def _worker_dimred_scan(args):
    A, zi, seed_shift = args
    out = run_sim_collective(A, zi, seed_shift=seed_shift)
    pn = est_period(out['xw'], out['sn']['t'])
    pe = est_period(out['sr']['x'], out['sn']['t'])
    an = tail_amplitude(out['xw'])
    ae = tail_amplitude(out['sr']['x'])
    pth = relaxation_period(out['Ae'], zi)
    ath = relaxation_amplitude(out['Ae'], zi)
    return {'z0': float(zi), 'T_net': pn['T_mean'] if pn['valid'] else np.nan, 'T_eff': pe['T_mean'] if pe['valid'] else np.nan, 'T_theory_relax': pth['T'] if pth['valid'] else np.nan, 'period_valid': bool(pn['valid'] and pe['valid']), 'theory_period_valid': bool(pn['valid'] and pth['valid']), 'slow_flow_valid': bool(pth.get('flow_valid', False) and ath.get('flow_valid', False)), 'g_fold_left': float(pth.get('g_fold_left', np.nan)), 'g_fold_right': float(pth.get('g_fold_right', np.nan)), 'A_net': an, 'A_red': ae, 'A_theory_relax': ath['A'] if ath['valid'] else np.nan, 'theory_amplitude_valid': bool(np.isfinite(an) and an > 1e-12 and ath['valid']), 'period_ratio': float(pe['T_mean'] / pn['T_mean']) if pn['valid'] and pe['valid'] and (pn['T_mean'] > 1e-12) else np.nan, 'period_ratio_theory_net': float(pth['T'] / pn['T_mean']) if pn['valid'] and pth['valid'] and (pn['T_mean'] > 1e-12) else np.nan, 'amplitude_ratio': float(ae / max(an, 1e-12)), 'amplitude_ratio_theory_net': float(ath['A'] / max(an, 1e-12)) if ath['valid'] else np.nan, 'period_error': float(abs(pn['T_mean'] - pe['T_mean']) / pn['T_mean']) if pn['valid'] and pe['valid'] and (pn['T_mean'] > 1e-12) else np.nan, 'period_error_theory': float(abs(pn['T_mean'] - pth['T']) / pn['T_mean']) if pn['valid'] and pth['valid'] and (pn['T_mean'] > 1e-12) else np.nan, 'amplitude_error': float(abs(an - ae) / max(an, 1e-12)), 'amplitude_error_theory': float(abs(an - ath['A']) / max(an, 1e-12)) if ath['valid'] else np.nan}

def _worker_sensitivity(args):
    A, N, z_sens_p, pname, pv, b_def, c_def, Acoup, t_ev, dr, trial_idx = args
    seed = (ER_SEED + 10000 + int(trial_idx) * 1009 + int(abs(float(z_sens_p)) * 10000) + int(abs(float(pv)) * 1000)) % 999999
    x0_, y0_ = make_ic(N, seed)
    w_ = degree_weights(A)
    b_ = b_def
    c_ = c_def
    if pname == 'b':
        b_ = pv
    elif pname == 'c':
        c_ = pv
    av_ = np.full(N, a)
    bv_ = np.full(N, b_)
    cv_ = np.full(N, c_)
    zv_ = np.full(N, z_sens_p)
    A_scaled = A if pname in {'b', 'c'} else matrix_for_aeff_sensitivity(A, float(pv))
    sn_ = rk4_net_collective(A_scaled, x0_, y0_, zv_, t_ev, w_, av_, bv_, cv_)
    xw_ = sn_['xw']
    yw_ = sn_['yw']
    n0_ = int(dr * len(xw_))
    det_ = detect_osc(xw_, t_ev)
    if not det_.get('valid', True):
        return (pv, float(z_sens_p), int(trial_idx), False, np.nan, np.nan, False)
    dx = float(np.max(xw_[n0_:]) - np.min(xw_[n0_:]))
    dy = float(np.max(yw_[n0_:]) - np.min(yw_[n0_:]))
    return (pv, float(z_sens_p), int(trial_idx), det_['is_osc'], dx, dy, True)

def _worker_baseline_probability(args):
    A, w, seed_base, trial_idx = args
    N = A.shape[0]
    x0, y0 = make_ic(N, seed_base + trial_idx * 13 + 1)
    zv = make_oscillatory_z_vec(N)
    av_, bv_, cv_ = make_abc(N, w, seed_base + trial_idx * 13 + 3)
    sn = rk4_net_collective(A, x0, y0, zv, t_eval, w, av_, bv_, cv_)
    mx = collective_state_metrics(sn['xw'], t_eval)
    return {'trial': int(trial_idx), 'baseline_osc': bool(mx['is_osc']), 'baseline_valid': bool(mx['valid']), 'baseline_tail_amp': mx['tail_amp']}

def _worker_control_probability(args):
    A, w, ctrl, z_driver, seed_base, trial_idx, baseline_osc, baseline_amp, k = args
    N = A.shape[0]
    baseline_valid = bool(np.isfinite(baseline_amp))
    if len(ctrl) == 0:
        nonoscillatory = bool(baseline_valid and (not baseline_osc))
        return {'trial': int(trial_idx), 'k': int(k), 'success': False, 'nonoscillatory': nonoscillatory, 'controlled_valid': baseline_valid, 'baseline_valid': baseline_valid, 'x_tail_amp': float(baseline_amp), 'W_ctrl': 0.0, 'baseline_osc': bool(baseline_osc), 'baseline_tail_amp': float(baseline_amp)}
    if not np.isfinite(z_driver):
        return {'trial': int(trial_idx), 'k': int(k), 'success': False, 'nonoscillatory': False, 'controlled_valid': False, 'baseline_valid': baseline_valid, 'x_tail_amp': np.nan, 'W_ctrl': driver_weight(w, ctrl), 'baseline_osc': bool(baseline_osc), 'baseline_tail_amp': float(baseline_amp)}
    x0, y0 = make_ic(N, seed_base + trial_idx * 13 + 1)
    zv = make_oscillatory_z_vec(N)
    av_, bv_, cv_ = make_abc(N, w, seed_base + trial_idx * 13 + 3)
    sn = rk4_net_collective_pinned(A, x0, y0, zv, t_eval, w, ctrl, z_driver, av_, bv_, cv_)
    xw_ = sn['xw']
    mx = collective_state_metrics(xw_, t_eval)
    nonoscillatory = bool(mx['valid'] and (not mx['is_osc']))
    return {'trial': int(trial_idx), 'k': int(k), 'success': bool(baseline_valid and baseline_osc and nonoscillatory), 'nonoscillatory': nonoscillatory, 'controlled_valid': bool(mx['valid']), 'baseline_valid': baseline_valid, 'x_tail_amp': mx['tail_amp'], 'W_ctrl': driver_weight(w, ctrl), 'baseline_osc': bool(baseline_osc), 'baseline_tail_amp': float(baseline_amp)}

def _worker_control_probability_dose(args):
    A, w, ctrl, z_driver, seed_base, trial_idx, baseline_osc, baseline_amp, k, dose = args
    row = _worker_control_probability((A, w, ctrl, z_driver, seed_base, trial_idx, baseline_osc, baseline_amp, k))
    row['dose'] = float(dose)
    return row

def find_k_eff_success(summary, threshold=CONTROL_PROB_THRESHOLD):
    rows = sorted(summary, key=lambda r: r['k'])
    return next((r for r in rows if np.isfinite(r.get('success_probability', np.nan)) and r['success_probability'] >= threshold), None)

def success_threshold_table(summary, thresholds=None):
    thresholds = tuple(thresholds or (CONTROL_PROB_THRESHOLD, CONTROL_PROB_STRICT_THRESHOLD))
    keep = ('k', 'k_frac', 'success_probability', 'W_ctrl', 'z_driver', 'z_driver_unbounded', 'z_eff_realized', 'target_feasible', 'n_total', 'n_valid_pairs', 'n_invalid_baseline', 'n_invalid_control')
    table = {}
    for threshold in thresholds:
        row = find_k_eff_success(summary, float(threshold))
        key = f'{float(threshold):.3g}'
        if row is None:
            table[key] = None
        else:
            table[key] = {k: row[k] for k in keep if k in row}
    return table

def run_fig3(A):
    print('\n[Fig 3] Oscillation interval & parameter sensitivity...')
    sd = DIRS['fig3']
    cleanup_outputs(sd, ['fig3f', 'fig3g', 'fig3h', 'fig3i', 'fig3j'])
    A, net_info = benchmark_operating_point_info(A, 'fig3_er')
    save_json(net_info, os.path.join(sd, 'fig3_network_operating_point.json'))
    w = degree_weights(A)
    N = A.shape[0]
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    zl, zr = (theory['z_left'], theory['z_right'])
    print(f'  Ae={Ae:.6f}, interval=[{zl:.4f},{zr:.4f}]')
    print(f'  fig3a: z-scan ({N_SCAN} points x {N_ZSCAN_TRIALS} trials, {N_WORKERS} workers)...')
    z_arr = np.linspace(zl - 0.3 * (zr - zl), zr + 0.3 * (zr - zl), N_SCAN)
    tasks = [(A, zi, int(abs(zi) * 10000.0 + trial * 101 + 17) % 999999) for zi in z_arr for trial in range(N_ZSCAN_TRIALS)]
    osc_flags = np.zeros(N_SCAN)
    amps_ = np.zeros(N_SCAN)
    xmins = np.zeros(N_SCAN)
    xmaxs = np.zeros(N_SCAN)
    ymins = np.zeros(N_SCAN)
    ymaxs = np.zeros(N_SCAN)
    chunksize = max(1, len(tasks) // max(1, N_WORKERS * 4))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_worker_zscan, tasks, chunksize=chunksize))
    zscan_rows = []
    for idx, zi in enumerate(z_arr):
        rows = [r for r in results if abs(r[0] - zi) < 1e-12]
        valid_rows = [r for r in rows if bool(r[7])]
        osc_vals = np.array([r[1] for r in valid_rows], dtype=float)
        amp_vals = np.array([r[2] for r in valid_rows], dtype=float)
        osc_flags[idx] = float(np.mean(osc_vals)) if len(osc_vals) else np.nan
        amps_[idx] = float(np.mean(amp_vals)) if len(amp_vals) else np.nan
        osc_rows = [r for r in valid_rows if r[1] > 0.5]
        env_rows = osc_rows if osc_rows else valid_rows
        xmins[idx] = float(np.mean([r[3] for r in env_rows])) if env_rows else np.nan
        xmaxs[idx] = float(np.mean([r[4] for r in env_rows])) if env_rows else np.nan
        ymins[idx] = float(np.mean([r[5] for r in env_rows])) if env_rows else np.nan
        ymaxs[idx] = float(np.mean([r[6] for r in env_rows])) if env_rows else np.nan
        zscan_rows.append({'z': float(zi), 'oscillation_probability': float(osc_flags[idx]), 'mean_tail_amplitude': float(amps_[idx]), 'x_min': float(xmins[idx]), 'x_max': float(xmaxs[idx]), 'y_min': float(ymins[idx]), 'y_max': float(ymaxs[idx]), 'n_total': int(len(rows)), 'n_valid': int(len(valid_rows)), 'n_invalid': int(len(rows) - len(valid_rows))})
    save_json({'n_trials': N_ZSCAN_TRIALS, 'rows': zscan_rows}, os.path.join(sd, 'fig3_zscan_probability.json'))
    fig, ax = mfig((7, 4))
    mo = osc_flags > 0.5
    ms_ = ~mo
    ax.axvspan(z_arr.min(), zl, color=COLOR_STABLE_FILL, alpha=0.45, zorder=0)
    ax.axvspan(zl, zr, color=COLOR_OSC_FILL, alpha=0.55, zorder=0)
    ax.axvspan(zr, z_arr.max(), color=COLOR_STABLE_FILL, alpha=0.45, zorder=0)
    ax.plot(z_arr, osc_flags, color=COLOR_XW, lw=1.5, zorder=2)
    sc = ax.scatter(z_arr, osc_flags, s=30, c=osc_flags, cmap=CMAP_OSC_PROB, vmin=0, vmax=1, edgecolors='k', lw=0.25, zorder=3)
    ax.axvline(zl, color=COLOR_THEORY_FP, ls='--', lw=1.2, label='$z_l, z_r$')
    ax.axvline(zr, color=COLOR_THEORY_FP, ls='--', lw=1.2)
    ax.text((zl + zr) / 2, 1.02, 'analytical candidate envelope', fontsize=7, color=COLOR_XW, ha='center', va='bottom')
    ax.set_xlabel('$z_{\\mathrm{eff}}$')
    ax.set_ylabel('Oscillation probability')
    ax.set_ylim(-0.05, 1.08)
    legend_clean(ax, loc='lower right')
    save_sub(fig, sd, 'fig3a')
    print('  fig3b: phase diagram...')
    fig, ax = mfig((7, 5))
    Ac_val = 4.0 / (3.0 * b) - b / (3.0 * c ** 2) - 1.0
    A_range = np.linspace(0.001, max(Ac_val * 1.5, Ae * 1.2, 1.0), 400)
    zl1, zr1, zl2, zr2 = ([], [], [], [])
    for Ai in A_range:
        alpha_ = 1.0 + Ai - 1.0 / b
        beta_ = 1.0 + Ai - b / c ** 2
        if alpha_ > 0:
            amp_ = 2.0 / 3.0 * alpha_ ** 1.5
            zl1.append(-amp_ - a / b)
            zr1.append(amp_ - a / b)
        else:
            zl1.append(np.nan)
            zr1.append(np.nan)
        if beta_ > 0:
            xh_ = np.sqrt(beta_)

            def zx_(x__, Ai_=Ai):
                return x__ ** 3 / 3.0 - (1.0 + Ai_ - 1.0 / b) * x__ - a / b
            v1, v2 = (zx_(-xh_), zx_(xh_))
            zl2.append(min(v1, v2))
            zr2.append(max(v1, v2))
        else:
            zl2.append(np.nan)
            zr2.append(np.nan)
    zl1, zr1, zl2, zr2 = (np.array(zl1), np.array(zr1), np.array(zl2), np.array(zr2))
    ok1 = ~(np.isnan(zl1) | np.isnan(zr1))
    ok2 = ~(np.isnan(zl2) | np.isnan(zr2))
    ax.fill_betweenx(A_range[ok1], zl1[ok1], zr1[ok1], color=COLOR_OSC_FILL, alpha=0.65, label='$z_1$ interval')
    ax.plot(zl1[ok1], A_range[ok1], color=COLOR_XW, lw=1.2, ls='-')
    ax.plot(zr1[ok1], A_range[ok1], color=COLOR_XW, lw=1.2, ls='-')
    ax.fill_betweenx(A_range[ok2], zl2[ok2], zr2[ok2], color=COLOR_STABLE_FILL, alpha=0.65, label='$z_2$ interval')
    ax.plot(zl2[ok2], A_range[ok2], color=COLOR_XEFF, lw=1.2, ls='--')
    ax.plot(zr2[ok2], A_range[ok2], color=COLOR_XEFF, lw=1.2, ls='--')
    ax.axhline(Ac_val, color=COLOR_THEORY_FP, ls='-.', lw=1.8, label=f'$A_{{\\mathrm{{cross}}}}={Ac_val:.3f}$')
    legend_clean(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    save_sub(fig, sd, 'fig3b')
    print('  fig3c: amplitude envelope vs z_eff...')
    fig, ax = mfig((7, 4))
    stable_tail = 0.5 * (xmins + xmaxs)
    ax.axvspan(z_arr.min(), zl, color=COLOR_STABLE_FILL, alpha=0.32, zorder=0)
    ax.axvspan(zl, zr, color=COLOR_OSC_FILL, alpha=0.3, zorder=0)
    ax.axvspan(zr, z_arr.max(), color=COLOR_STABLE_FILL, alpha=0.32, zorder=0)
    ax.fill_between(z_arr, xmins, xmaxs, where=mo, interpolate=True, color=COLOR_XW, alpha=0.24, zorder=2, label='Oscillatory tail envelope')
    ax.plot(z_arr[mo], xmins[mo], color=COLOR_XW, lw=1.1, alpha=0.88, zorder=3)
    ax.plot(z_arr[mo], xmaxs[mo], color=COLOR_XW, lw=1.1, alpha=0.88, zorder=3)
    ax.vlines(z_arr[mo], xmins[mo], xmaxs[mo], color=COLOR_XW, lw=0.55, alpha=0.26, zorder=2)
    ax.scatter(z_arr[ms_], stable_tail[ms_], s=26, c=COLOR_EMP_FP, edgecolors='k', lw=0.25, alpha=0.92, label='Non-oscillatory tail state', zorder=4)
    ax.axvline(zl, color=COLOR_THEORY_FP, ls='--', lw=1.0, label='Candidate boundary')
    ax.axvline(zr, color=COLOR_THEORY_FP, ls='--', lw=1.0)
    ax.set_xlabel('$z_{\\mathrm{eff}}$')
    ax.set_ylabel('Tail $\\bar{x}$ state / envelope')
    ax.set_title('Non-oscillatory tail state and oscillatory amplitude envelope', fontsize=8)
    yvals = np.r_[xmins[np.isfinite(xmins)], xmaxs[np.isfinite(xmaxs)], stable_tail[np.isfinite(stable_tail)]]
    if yvals.size:
        pad = 0.06 * max(float(np.nanmax(yvals) - np.nanmin(yvals)), 1e-09)
        ax.set_ylim(float(np.nanmin(yvals) - pad), float(np.nanmax(yvals) + pad))
    legend_clean(ax, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    save_sub(fig, sd, 'fig3c')
    print('  fig3d-e: parameter sensitivity maps (b-z_eff and c-z_eff)...')
    Acoup = effective_coupling(A)

    def _z_interval_for(b_, c_, Ae_):
        alpha_ = 1.0 + Ae_ - 1.0 / b_
        beta_ = 1.0 + Ae_ - b_ / c_ ** 2
        Ac_ = 4.0 / (3.0 * b_) - b_ / (3.0 * c_ ** 2) - 1.0
        z1_ = None
        z2_ = None
        if alpha_ > 0:
            amp_ = 2.0 / 3.0 * alpha_ ** 1.5
            z1_ = (-amp_ - a / b_, amp_ - a / b_)
        if beta_ > 0:
            xh_ = np.sqrt(beta_)
            coef_ = 1.0 + Ae_ - 1.0 / b_
            vl = (-xh_) ** 3 / 3.0 - coef_ * -xh_ - a / b_
            vr = xh_ ** 3 / 3.0 - coef_ * xh_ - a / b_
            z2_ = (min(vl, vr), max(vl, vr))
        return z2_ or z1_ if Ae_ < Ac_ else z1_ or z2_
    params = [('b', np.linspace(0.3, 1.5, N_SENS), b, 'fig3d', '$b$'), ('c', np.linspace(0.8, c, N_SENS), c, 'fig3e', '$c$')]
    sens_summary = []
    for pname, pvals, pdef, fname, plbl in params:

        def _get_iv(pv):
            b_, c_, Ae_ = (b, c, Ae)
            if pname == 'b':
                b_ = pv
            elif pname == 'c':
                c_ = pv
            else:
                Ae_ = pv
            return _z_interval_for(b_, c_, Ae_)
        intervals = [_get_iv(pv) for pv in pvals]
        bounds = [v for iv in intervals if iv is not None for v in iv if np.isfinite(v)]
        if len(bounds) >= 2:
            span = max(bounds) - min(bounds)
            z_min = min(bounds) - 0.16 * span
            z_max = max(bounds) + 0.16 * span
            if np.isfinite(SENS_Z_ABS_MAX) and SENS_Z_ABS_MAX > 0:
                if min(bounds) >= -SENS_Z_ABS_MAX and max(bounds) <= SENS_Z_ABS_MAX:
                    z_min = max(z_min, -SENS_Z_ABS_MAX)
                    z_max = min(z_max, SENS_Z_ABS_MAX)
            if z_min >= z_max:
                z_min, z_max = (-SENS_Z_ABS_MAX, SENS_Z_ABS_MAX)
            z_grid = np.linspace(z_min, z_max, N_SENS_Z)
        else:
            z_grid = np.linspace(max(zl - 0.3 * (zr - zl), -SENS_Z_ABS_MAX), min(zr + 0.3 * (zr - zl), SENS_Z_ABS_MAX), N_SENS_Z)
        print(f'  {pname}: {len(pvals)} x {len(z_grid)} x {N_SENS_TRIALS} sensitivity grid')
        tasks_s = [(A, N, zv, pname, pv, b, c, Acoup, t_eval, discard_ratio, trial) for pv in pvals for zv in z_grid for trial in range(N_SENS_TRIALS)]
        chunksize = max(1, len(tasks_s) // max(1, N_WORKERS * 4))
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            results = list(pool.map(_worker_sensitivity, tasks_s, chunksize=chunksize))
        sim_counts = np.zeros((len(z_grid), len(pvals)), dtype=float)
        sim_n = np.zeros((len(z_grid), len(pvals)), dtype=float)
        invalid_n = np.zeros((len(z_grid), len(pvals)), dtype=float)
        delta_x = np.zeros((len(z_grid), len(pvals)), dtype=float)
        for pv, zv, trial, is_osc, dx, dy, is_valid in results:
            i = int(np.argmin(np.abs(z_grid - float(zv))))
            j = int(np.argmin(np.abs(pvals - float(pv))))
            if not is_valid:
                invalid_n[i, j] += 1.0
                continue
            sim_counts[i, j] += float(bool(is_osc))
            sim_n[i, j] += 1.0
            delta_x[i, j] += float(dx)
        valid_cells = sim_n > 0
        with np.errstate(invalid='ignore', divide='ignore'):
            sim_prob = np.divide(sim_counts, sim_n, out=np.full_like(sim_counts, np.nan), where=valid_cells)
            delta_x = np.divide(delta_x, sim_n, out=np.full_like(delta_x, np.nan), where=valid_cells)
        sim_osc = np.zeros_like(sim_counts, dtype=bool)
        sim_osc[valid_cells] = sim_prob[valid_cells] >= 0.5
        sigma = max(0.0, float(SENS_SMOOTH_SIGMA))
        if sigma > 0:
            smooth_num = gaussian_filter(np.where(valid_cells, sim_prob, 0.0), sigma=sigma, mode='nearest')
            smooth_den = gaussian_filter(valid_cells.astype(float), sigma=sigma, mode='nearest')
            sim_prob_plot = np.divide(smooth_num, smooth_den, out=np.full_like(sim_prob, np.nan), where=smooth_den > 1e-12)
            sim_prob_plot[~valid_cells] = np.nan
        else:
            sim_prob_plot = sim_prob.copy()
        zl_arr = np.array([iv[0] if iv is not None else np.nan for iv in intervals], dtype=float)
        zr_arr = np.array([iv[1] if iv is not None else np.nan for iv in intervals], dtype=float)
        pred_osc = np.zeros_like(sim_osc, dtype=bool)
        for j, iv in enumerate(intervals):
            if iv is not None:
                pred_osc[:, j] = (z_grid >= iv[0]) & (z_grid <= iv[1])
        pred_accuracy = float(np.mean(pred_osc[valid_cells] == sim_osc[valid_cells])) if np.any(valid_cells) else np.nan
        sens_summary.append({'parameter': pname, 'z_values': z_grid, 'p_values': pvals, 'n_trials': N_SENS_TRIALS, 'n_valid_grid': sim_n, 'n_invalid_grid': invalid_n, 'z_left': zl_arr, 'z_right': zr_arr, 'predicted_osc_grid': pred_osc, 'simulated_osc_probability_grid': sim_prob, 'smoothed_osc_probability_grid': sim_prob_plot, 'simulated_osc_grid': sim_osc, 'delta_x_eff_grid': delta_x, 'prediction_accuracy': pred_accuracy})
        fig, ax = mfig((6.4, 4.8))
        im = ax.contourf(pvals, z_grid, sim_prob_plot, levels=np.linspace(0, 1, 21), cmap=CMAP_OSC_PROB, vmin=0, vmax=1, alpha=0.88)
        if np.nanmin(sim_prob_plot) <= 0.5 <= np.nanmax(sim_prob_plot):
            ax.contour(pvals, z_grid, sim_prob_plot, levels=[0.5], colors='#272727', linewidths=0.8, alpha=0.78)
        ok = ~(np.isnan(zl_arr) | np.isnan(zr_arr))
        if np.any(ok):
            ax.plot(pvals[ok], zl_arr[ok], color=COLOR_THEORY_FP, lw=1.4, ls='--', label='Candidate boundary')
            ax.plot(pvals[ok], zr_arr[ok], color=COLOR_THEORY_FP, lw=1.4, ls='--')
        ax.axhline(0.0, color='#4D4D4D', ls=':', lw=1.0, label='$z_{\\mathrm{eff}}=0$')
        ax.axvline(pdef, color='#767676', ls='-.', lw=0.9)
        ax.set_xlabel(plbl)
        ax.set_ylabel('$z_{\\mathrm{eff}}$')
        ax.set_title(f'{plbl} sensitivity map (agreement={pred_accuracy:.2f}, n={N_SENS_TRIALS})', fontsize=8)
        handles = [Line2D([0], [0], color=COLOR_THEORY_FP, ls='--', lw=1.4, label='Candidate boundary'), Line2D([0], [0], color='#272727', lw=0.8, label='Sim. $P_{\\mathrm{osc}}=0.5$')]
        legend_clean(ax, handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('$P_{\\mathrm{osc}}$', fontsize=8)
        save_sub(fig, sd, fname)
    save_json(sens_summary, os.path.join(sd, 'fig3_parameter_sensitivity.json'))
    print('[Fig 3] Done.\n')

def rk4_net_collective_pinned(A, x0, y0, zv, tev, w, drivers, z_target, av=None, bv=None, cv=None):
    drivers = np.asarray(drivers, dtype=int)
    z_ctrl = np.asarray(zv, dtype=float).copy()
    if len(drivers) > 0:
        z_ctrl[drivers] = float(z_target)
    return rk4_net_collective(A, x0, y0, z_ctrl, tev, w, av, bv, cv)

def run_fig4(A_shared=None, A_ba_shared=None):
    print('\n[Fig 4] ER / BA / SW bounded-input control threshold...')
    sd = DIRS['fig4']
    cleanup_outputs(sd, ['fig4f'])
    NET_TYPES = ['ER', 'SW', 'BA']
    all_results = {}
    network_meta = {}
    dose_response = {}
    network_state = {}
    for net_type in NET_TYPES:
        print(f'\n  --- {net_type} network ---')
        if net_type == 'ER' and A_shared is not None:
            A_raw = np.asarray(A_shared, dtype=float).copy()
        elif net_type == 'BA' and A_ba_shared is not None:
            A_raw = np.asarray(A_ba_shared, dtype=float).copy()
        else:
            A_raw = make_fig4_network(net_type)
        A, net_info = benchmark_operating_point_info(A_raw, f'fig4_{net_type}')
        N = A.shape[0]
        w = degree_weights(A)
        Ae = effective_coupling(A)
        theory = derive_z_interval(Ae)
        z_eff_target = analytic_nonoscillatory_target(theory)
        print(f'  N={N}, edges={int(np.count_nonzero(A))}, Ae={Ae:.6f}, z_eff_target={z_eff_target:.4f}, |Δz_D|ref={REFERENCE_DRIVER_DOSE:.2f}')
        out_deg_order = np.argsort(-A.sum(axis=0))
        k_grid = np.arange(0, N + 1, FIG4_K_STEP, dtype=int)
        seed_base = FIG4_NET_SEEDS[net_type] * 1000
        baseline_tasks = [(A, w, seed_base, trial) for trial in range(N_CONTROL_PROB_TRIALS)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            baseline_rows = list(pool.map(_worker_baseline_probability, baseline_tasks, chunksize=1))
        baseline_by_trial = {r['trial']: r for r in baseline_rows}
        schedule = []
        for k in k_grid:
            ctrl = sorted(out_deg_order[:k].tolist()) if k > 0 else []
            info = solve_baseline_driver_z(w, ctrl, z_eff_target)
            schedule.append({'k': int(k), 'drivers': ctrl, 'W_ctrl': float(info['W_D']), 'z_driver': float(info['z_driver']) if info['valid'] else np.nan, 'z_driver_unbounded': float(info['z_driver_unbounded']) if info['valid'] else np.nan, 'z_eff_target': float(z_eff_target), 'z_eff_realized': float(info['z_eff_realized']), 'target_feasible': bool(info.get('target_feasible', False)), 'valid': bool(info['valid'])})
        tasks = []
        for info in schedule:
            for trial in range(N_CONTROL_PROB_TRIALS):
                base = baseline_by_trial[int(trial)]
                tasks.append((A, w, info['drivers'], info['z_driver'], seed_base, trial, base['baseline_osc'], base['baseline_tail_amp'], info['k']))
        chunksize = max(1, len(tasks) // max(1, N_WORKERS * 4))
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            results = list(pool.map(_worker_control_probability, tasks, chunksize=chunksize))
        schedule_by_k = {s['k']: s for s in schedule}
        summary = []
        for k in k_grid:
            k_int = int(k)
            rows = [r for r in results if r['k'] == k_int]
            sched = schedule_by_k[k_int]
            probability = summarize_control_probability(rows)
            summary.append({'k': k_int, 'k_frac': float(k_int / N), **probability, 'W_ctrl': float(sched['W_ctrl']), 'z_driver': float(sched['z_driver']) if np.isfinite(sched['z_driver']) else None, 'z_driver_unbounded': float(sched['z_driver_unbounded']) if np.isfinite(sched['z_driver_unbounded']) else None, 'z_eff_realized': float(sched['z_eff_realized']), 'target_feasible': bool(sched['target_feasible'])})
        all_results[net_type] = summary
        network_meta[net_type] = {'N': int(N), 'A_eff': float(Ae), 'z_eff_target': float(z_eff_target), 'weight_mode': net_info['weight_mode'], 'nominal_mean_degree': float(net_info['nominal_mean_degree']), 'W_D_required_at_reference_dose': float(abs(z_eff_target - OSCILLATORY_Z) / REFERENCE_DRIVER_DOSE) if REFERENCE_DRIVER_DOSE > 0 else np.nan, 'reachability_W_D_required_at_reference_dose': float(abs(z_eff_target - OSCILLATORY_Z) / REFERENCE_DRIVER_DOSE) if REFERENCE_DRIVER_DOSE > 0 else np.nan, 'z_interval': [float(theory['z_left']), float(theory['z_right'])], 'topology_assumption_audit': topology_reduction_assumption_audit(A, w, Ae)}
        network_state[net_type] = {'A': A, 'w': w, 'out_order': out_deg_order, 'z_eff_target': float(z_eff_target), 'seed_base': int(seed_base), 'N': int(N)}
        crit = find_k_eff_success(summary, CONTROL_PROB_THRESHOLD)
        threshold_reached = crit is not None
        k_eff = int(crit['k']) if threshold_reached else None
        k_eff_frac = float(k_eff / N) if threshold_reached else np.nan
        network_meta[net_type]['empirical_success_W_D_at_k_eff'] = float(crit['W_ctrl']) if threshold_reached else np.nan
        network_meta[net_type]['empirical_success_threshold_reached'] = bool(threshold_reached)
        network_meta[net_type]['empirical_success_k_eff'] = k_eff
        network_meta[net_type]['empirical_success_k_eff_frac'] = float(k_eff_frac)
        if threshold_reached:
            print(f'  K_eff={k_eff} ({k_eff_frac:.2%} of N)')
        else:
            print('  K_eff undefined: the tested node counts did not reach the success threshold')
        dose_rows = []
        dose_grid = sorted(set((float(v) for v in FIG4_LANDSCAPE_DOSE_GRID)))
        for dose in dose_grid:
            if abs(float(dose) - MAX_DRIVER_Z) < 1e-12:
                dose_summary = summary
            else:
                dose_tasks = []
                schedule_dose = {}
                for k in k_grid:
                    ctrl = sorted(out_deg_order[:int(k)].tolist()) if k > 0 else []
                    W_ctrl = driver_weight(w, ctrl)
                    schedule_dose[int(k)] = {'drivers': ctrl, 'W_ctrl': float(W_ctrl), 'z_driver': float(driver_z_from_dose(z_eff_target, dose)) if k > 0 else np.nan, 'z_eff_realized': realized_z_eff_from_driver(W_ctrl, driver_z_from_dose(z_eff_target, dose))}
                    for trial in range(N_CONTROL_PROB_TRIALS):
                        base = baseline_by_trial[int(trial)]
                        dose_tasks.append((A, w, ctrl, schedule_dose[int(k)]['z_driver'], seed_base, trial, base['baseline_osc'], base['baseline_tail_amp'], int(k)))
                chunksize = max(1, len(dose_tasks) // max(1, N_WORKERS * 4))
                with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                    dose_results = list(pool.map(_worker_control_probability, dose_tasks, chunksize=chunksize))
                dose_summary = []
                for k in k_grid:
                    k_int = int(k)
                    rows = [r for r in dose_results if r['k'] == k_int]
                    sched = schedule_dose[k_int]
                    probability = summarize_control_probability(rows)
                    dose_summary.append({'k': k_int, 'k_frac': float(k_int / N), **probability, 'W_ctrl': float(sched['W_ctrl']), 'z_driver': float(sched['z_driver']) if np.isfinite(sched['z_driver']) else None, 'z_eff_realized': float(sched['z_eff_realized'])})
            crit_dose = find_k_eff_success(dose_summary, CONTROL_PROB_THRESHOLD)
            dose_rows.append({'dose': float(dose), 'k_eff_frac': float(crit_dose['k_frac']) if crit_dose else np.nan, 'summary': dose_summary})
        dose_response[net_type] = dose_rows
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.2), sharey=True)
    k_effs = {}
    im = None
    for ax, net_type in zip(axes, NET_TYPES):
        summary = all_results[net_type]
        crit = find_k_eff_success(summary, CONTROL_PROB_THRESHOLD)
        n_net = max(1, int(network_state[net_type]['N']))
        k_effs[net_type] = crit['k'] / n_net if crit else np.nan
        dose_rows = sorted(dose_response[net_type], key=lambda r: r['dose'])
        doses = np.array([r['dose'] for r in dose_rows], dtype=float)
        kf = np.array([r['k_frac'] for r in dose_rows[0]['summary']], dtype=float)
        P = np.array([[rr['success_probability'] for rr in row['summary']] for row in dose_rows], dtype=float)
        im = ax.contourf(kf, doses, P, levels=SUCCESS_LEVELS, vmin=0, vmax=1, cmap=CMAP_SUCCESS)
        success_mask = (P >= CONTROL_PROB_THRESHOLD - 1e-12).astype(float)
        if np.nanmin(success_mask) < 0.5 < np.nanmax(success_mask):
            ax.contour(kf, doses, success_mask, levels=[0.5], colors='#272727', linewidths=1.1)
        if doses.min() - 1e-12 <= MAX_DRIVER_Z <= doses.max() + 1e-12:
            ax.axhline(MAX_DRIVER_Z, color='#272727', ls=':', lw=0.9)
            if np.isfinite(k_effs[net_type]):
                ax.axvline(k_effs[net_type], color=COLOR_NET[net_type], ls='-.', lw=1.0)
        display = 'WS' if net_type == 'SW' else net_type
        ax.text(0.03, 0.94, display, transform=ax.transAxes, fontsize=8, fontweight='bold', color=COLOR_NET[net_type], ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.5))
        ax.set_xlabel('Controlled fraction $k/N$')
        ax.set_xlim(0, 1)
        ax.set_ylim(doses.min(), doses.max())
    axes[0].set_ylabel('Driver dose $|\\Delta z_D|$')
    fig.subplots_adjust(wspace=0.16, right=0.9, left=0.08, bottom=0.16, top=0.92)
    cax = fig.add_axes([0.92, 0.24, 0.018, 0.56])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('$P_{\\mathrm{success}}$', fontsize=8)
    fig._skip_tight_layout = True
    save_sub(fig, sd, 'fig4a')
    fig4_dynamic_summary = {}
    panel_names = {'ER': 'fig4b', 'SW': 'fig4c', 'BA': 'fig4d'}
    for net_type in NET_TYPES:
        state = network_state[net_type]
        A = state['A']
        w = state['w']
        N = state['N']
        out_order = state['out_order']
        crit = find_k_eff_success(all_results[net_type], CONTROL_PROB_THRESHOLD)
        threshold_reached = crit is not None
        k_eff = int(crit['k']) if threshold_reached else None
        k_representative = int(k_eff) if threshold_reached else int(N)
        ctrl_top = sorted(out_order[:k_representative].tolist()) if k_representative > 0 else []
        rg = np.random.default_rng(state['seed_base'] + 909)
        ctrl_rand = sorted(rg.choice(N, size=k_representative, replace=False).tolist()) if k_representative > 0 else []
        x0, y0 = make_ic(N, state['seed_base'] + 321)
        zv = make_oscillatory_z_vec(N)
        av_, bv_, cv_ = make_abc(N, w, state['seed_base'] + 323)
        sn_none = rk4_net_collective(A, x0.copy(), y0.copy(), zv.copy(), t_eval, w, av_, bv_, cv_)
        sn_top = rk4_net_collective_pinned(A, x0.copy(), y0.copy(), zv.copy(), t_eval, w, ctrl_top, driver_z_from_dose(state['z_eff_target'], MAX_DRIVER_Z), av_, bv_, cv_)
        sn_rand = rk4_net_collective_pinned(A, x0.copy(), y0.copy(), zv.copy(), t_eval, w, ctrl_rand, driver_z_from_dose(state['z_eff_target'], MAX_DRIVER_Z), av_, bv_, cv_)
        metrics = {'none': collective_state_metrics(sn_none['xw'], t_eval), 'top_k': collective_state_metrics(sn_top['xw'], t_eval), 'random_k': collective_state_metrics(sn_rand['xw'], t_eval)}
        fig4_dynamic_summary[net_type] = {'threshold_reached': bool(threshold_reached), 'k_eff': k_eff, 'k_eff_frac': float(k_eff / N) if threshold_reached else np.nan, 'representative_k': int(k_representative), 'representative_k_frac': float(k_representative / N), 'top_W_ctrl': float(driver_weight(w, ctrl_top)), 'random_W_ctrl': float(driver_weight(w, ctrl_rand)), 'metrics': metrics}
        fig, ax = mfig((7.2, 3.8))
        display = 'WS' if net_type == 'SW' else net_type
        set_label = f'$k_{{eff}}$ ({k_representative})' if threshold_reached else f'max tested $k$ ({k_representative})'
        ax.plot(t_eval, sn_none['xw'], color=COLOR_NO_CONTROL, lw=1.2, label='No control')
        ax.plot(t_eval, sn_top['xw'], color=COLOR_NET[net_type], lw=1.9, label=f'Top-out {set_label}')
        ax.plot(t_eval, sn_rand['xw'], color='#6FBA4F', lw=1.45, ls='--', label=f'Random {set_label}')
        ax.set_xlabel('Time $t$')
        ax.set_ylabel('$\\bar{x}(t)$')
        ax.set_title(f'{display}: matched node-count control dynamics', fontsize=8)
        ax.set_xlim(t_eval[0], t_eval[-1])
        legend_clean(ax, loc='upper right', fontsize=6.4)
        save_sub(fig, sd, panel_names[net_type])
    fig, ax = mfig((6.4, 4.1))
    reachability_vals = [float(network_meta[net_type]['reachability_W_D_required_at_reference_dose']) for net_type in NET_TYPES if np.isfinite(network_meta[net_type]['reachability_W_D_required_at_reference_dose'])]
    if reachability_vals:
        y_lo, y_hi = (min(reachability_vals), max(reachability_vals))
        ax.axhspan(y_lo, y_hi, color=COLOR_NO_CONTROL, alpha=0.1, zorder=0)
        ax.axhline(float(np.mean(reachability_vals)), color=COLOR_NO_CONTROL, ls=':', lw=0.9, alpha=0.72, zorder=1)
        ax.text(0.02, min(0.16, y_hi + 0.035), 'Reduced-coordinate\nreachability bound', color=COLOR_NO_CONTROL, fontsize=6.2, ha='left', va='bottom')
    for net_type in NET_TYPES:
        summary = all_results[net_type]
        kf = np.array([r['k_frac'] for r in summary], dtype=float)
        wd = np.array([r['W_ctrl'] for r in summary], dtype=float)
        display = 'WS' if net_type == 'SW' else net_type
        ax.plot(kf, wd, color=COLOR_NET[net_type], lw=1.4, marker='o', ms=2.7, label=display)
        crit = find_k_eff_success(summary, CONTROL_PROB_THRESHOLD)
        if crit:
            ax.plot([0, crit['k_frac']], [crit['W_ctrl'], crit['W_ctrl']], color=COLOR_NET[net_type], ls=':', lw=1.0, alpha=0.45)
            annotate_threshold_point(ax, crit['k_frac'], crit['W_ctrl'], f'{display} $k_{{eff}}$', COLOR_NET[net_type])
        label_line_end(ax, kf, wd, display, COLOR_NET[net_type])
    ax.set_xlabel('Fraction of controlled nodes $k/N$')
    ax.set_ylabel('Cumulative output weight $W_D$')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    save_sub(fig, sd, 'fig4e')
    save_json({'k_eff_fractions': k_effs, 'threshold': CONTROL_PROB_THRESHOLD, 'strict_threshold': CONTROL_PROB_STRICT_THRESHOLD, 'reference_driver_dose': REFERENCE_DRIVER_DOSE, 'control_input_policy': CONTROL_INPUT_POLICY, 'dose_response': dose_response, 'representative_dynamics': fig4_dynamic_summary, 'network_meta': network_meta, 'threshold_sensitivity': {net_type: success_threshold_table(all_results[net_type]) for net_type in NET_TYPES}, 'summary': all_results}, os.path.join(sd, 'fig4_summary.json'))
    print('[Fig 4] Done.\n')

def _worker_ordering_scan(args):
    A, w, node_order, k, z_eff_target, seed_base, trial_idx, baseline_osc, baseline_amp, fixed_dose = args
    N = A.shape[0]
    baseline_valid = bool(np.isfinite(baseline_amp))
    ctrl = sorted(node_order[:k].tolist()) if k > 0 else []
    if len(ctrl) == 0:
        nonoscillatory = bool(baseline_valid and (not baseline_osc))
        return {'trial': int(trial_idx), 'k': int(k), 'success': False, 'nonoscillatory': nonoscillatory, 'controlled_valid': baseline_valid, 'baseline_valid': baseline_valid, 'W_ctrl': 0.0, 'baseline_osc': bool(baseline_osc)}
    info = solve_baseline_driver_z(w, ctrl, z_eff_target, reference_dose=fixed_dose)
    if not info['valid'] or not np.isfinite(info['z_driver']):
        return {'trial': int(trial_idx), 'k': int(k), 'success': False, 'nonoscillatory': False, 'controlled_valid': False, 'baseline_valid': baseline_valid, 'W_ctrl': float(info['W_D']), 'baseline_osc': bool(baseline_osc)}
    x0, y0 = make_ic(N, seed_base + trial_idx * 13 + 1)
    zv = make_oscillatory_z_vec(N)
    av_, bv_, cv_ = make_abc(N, w, seed_base + trial_idx * 13 + 3)
    sn = rk4_net_collective_pinned(A, x0, y0, zv, t_eval, w, ctrl, info['z_driver'], av_, bv_, cv_)
    mx = collective_state_metrics(sn['xw'], t_eval)
    nonoscillatory = bool(mx['valid'] and (not mx['is_osc']))
    return {'trial': int(trial_idx), 'k': int(k), 'success': bool(baseline_valid and baseline_osc and nonoscillatory), 'nonoscillatory': nonoscillatory, 'controlled_valid': bool(mx['valid']), 'baseline_valid': baseline_valid, 'W_ctrl': float(info['W_D']), 'baseline_osc': bool(baseline_osc)}

def _run_ordering_scan(A, w, node_order, k_grid, z_eff_target, seed_base, label='', fixed_dose=None):
    N = A.shape[0]
    baseline_tasks = [(A, w, seed_base, trial) for trial in range(N_CONTROL_PROB_TRIALS)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        baseline_rows = list(pool.map(_worker_baseline_probability, baseline_tasks, chunksize=1))
    baseline_by_trial = {r['trial']: r for r in baseline_rows}
    tasks = []
    for k in k_grid:
        for trial in range(N_CONTROL_PROB_TRIALS):
            base = baseline_by_trial[int(trial)]
            tasks.append((A, w, node_order, int(k), z_eff_target, seed_base, trial, base['baseline_osc'], base['baseline_tail_amp'], fixed_dose))
    chunksize = max(1, len(tasks) // max(1, N_WORKERS * 4))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_worker_ordering_scan, tasks, chunksize=chunksize))
    summary = []
    for k in k_grid:
        k_int = int(k)
        rows = [r for r in results if r['k'] == k_int]
        ctrl = sorted(node_order[:k_int].tolist()) if k_int > 0 else []
        info = solve_baseline_driver_z(w, ctrl, z_eff_target, reference_dose=fixed_dose)
        probability = summarize_control_probability(rows)
        summary.append({'k': k_int, 'k_frac': float(k_int / N), **probability, 'W_ctrl': float(np.mean([r['W_ctrl'] for r in rows])) if rows else 0.0, 'z_driver': float(info['z_driver']) if info['valid'] else None, 'z_driver_unbounded': float(info['z_driver_unbounded']) if info['valid'] else None, 'z_eff_realized': float(info['z_eff_realized']), 'target_feasible': bool(info.get('target_feasible', False))})
    return summary

def _run_grouped_control_sets(A, w, group_sets, z_driver, seed_base, n_trials=None):
    n_trials = int(n_trials or N_CONTROL_PROB_TRIALS)
    baseline_tasks = [(A, w, seed_base, trial) for trial in range(n_trials)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        baseline_rows = list(pool.map(_worker_baseline_probability, baseline_tasks, chunksize=1))
    baseline_by_trial = {r['trial']: r for r in baseline_rows}
    idx_meta = {}
    tasks = []
    idx = 0
    for label, ctrl_sets in group_sets.items():
        for rep, ctrl in enumerate(ctrl_sets):
            ctrl = sorted((int(i) for i in ctrl))
            idx_meta[idx] = {'label': label, 'rep': int(rep), 'ctrl': ctrl}
            for trial in range(n_trials):
                base = baseline_by_trial[int(trial)]
                tasks.append((A, w, ctrl, z_driver, seed_base, trial, base['baseline_osc'], base['baseline_tail_amp'], idx))
            idx += 1
    chunksize = max(1, len(tasks) // max(1, N_WORKERS * 4))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_worker_control_probability, tasks, chunksize=chunksize))
    summary = []
    for label in group_sets.keys():
        idxs = [i for i, meta in idx_meta.items() if meta['label'] == label]
        rows = [r for r in results if int(r['k']) in idxs]
        ctrls = [idx_meta[i]['ctrl'] for i in idxs]
        w_vals = [driver_weight(w, ctrl) for ctrl in ctrls]
        k_vals = [len(ctrl) for ctrl in ctrls]
        probs_by_set = []
        for i in idxs:
            rows_i = [r for r in results if int(r['k']) == i]
            probs_by_set.append(summarize_control_probability(rows_i)['success_probability'])
        probability = summarize_control_probability(rows)
        summary.append({'label': label, 'n_sets': int(len(ctrls)), 'k': int(round(float(np.mean(k_vals)))) if k_vals else 0, 'k_frac': float(np.mean(k_vals) / len(w)) if k_vals else 0.0, **probability, 'success_probability_std': float(np.nanstd(probs_by_set)) if probs_by_set else 0.0, 'W_ctrl': float(np.mean(w_vals)) if w_vals else 0.0, 'W_ctrl_std': float(np.std(w_vals)) if w_vals else 0.0, 'z_driver': float(z_driver) if np.isfinite(z_driver) else None, 'z_eff_realized': realized_z_eff_from_driver(float(np.mean(w_vals)), z_driver) if w_vals else float(OSCILLATORY_Z), 'paired_baseline_control': True})
    return summary

def _empirical_interval_from_zscan(rows, threshold=0.5):
    rows = sorted((row for row in rows if np.isfinite(float(row.get('oscillation_probability', np.nan)))), key=lambda r: float(r['z']))
    if not rows:
        return {'left': np.nan, 'right': np.nan, 'width': np.nan, 'threshold': float(threshold), 'valid': False}
    zs = np.array([float(r['z']) for r in rows], dtype=float)
    ps = np.array([float(r['oscillation_probability']) for r in rows], dtype=float)
    inside = ps >= float(threshold)
    if not np.any(inside):
        return {'left': np.nan, 'right': np.nan, 'width': np.nan, 'threshold': float(threshold), 'valid': False}

    def crossing(i0, i1):
        z0, z1 = (zs[i0], zs[i1])
        p0, p1 = (ps[i0], ps[i1])
        if abs(p1 - p0) < 1e-12:
            return float(z1 if p1 >= threshold else z0)
        alpha = (float(threshold) - p0) / (p1 - p0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(z0 + alpha * (z1 - z0))
    idx = np.flatnonzero(inside)
    i_left = int(idx[0])
    i_right = int(idx[-1])
    left = float(zs[i_left])
    right = float(zs[i_right])
    if i_left > 0 and ps[i_left - 1] < threshold <= ps[i_left]:
        left = crossing(i_left - 1, i_left)
    if i_right < len(zs) - 1 and ps[i_right] >= threshold > ps[i_right + 1]:
        right = crossing(i_right, i_right + 1)
    return {'left': float(left), 'right': float(right), 'width': float(right - left), 'threshold': float(threshold), 'valid': True}

def _run_empirical_interval_scan(A, theory, seed_base):
    zl = float(theory['z_left'])
    zr = float(theory['z_right'])
    width = max(float(theory['width']), 1e-06)
    pad = 0.22 * width
    z_grid = np.linspace(zl - pad, zr + pad, max(7, int(FIG6_INTERVAL_SCAN_POINTS)))
    z_grid = np.unique(np.concatenate([z_grid, [zl, zr, 0.0]])).astype(float)
    tasks = [(A, float(zi), int(seed_base + 50000 + trial * 997)) for zi in z_grid for trial in range(max(1, int(FIG6_INTERVAL_SCAN_TRIALS)))]
    chunksize = max(1, len(tasks) // max(1, N_WORKERS * 4))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_worker_zscan, tasks, chunksize=chunksize))
    rows = []
    for zi in z_grid:
        vals = [r for r in results if abs(float(r[0]) - float(zi)) < 1e-12]
        if not vals:
            continue
        valid_vals = [v for v in vals if bool(v[7])]
        osc_probs = [float(v[1]) for v in valid_vals]
        amps = [float(v[2]) for v in valid_vals if np.isfinite(v[2])]
        rows.append({'z': float(zi), 'oscillation_probability': float(np.mean(osc_probs)) if osc_probs else np.nan, 'mean_tail_amplitude': float(np.mean(amps)) if amps else np.nan, 'n_total': int(len(vals)), 'n_valid': int(len(valid_vals)), 'n_invalid': int(len(vals) - len(valid_vals))})
    interval = _empirical_interval_from_zscan(rows, threshold=0.5)
    return {'n_trials': int(max(1, int(FIG6_INTERVAL_SCAN_TRIALS))), 'n_z': int(len(z_grid)), 'rows': rows, 'interval': interval}

def _round_k_to_grid(frac, N, k_step):
    k = int(round(float(frac) * int(N)))
    k = int(round(k / max(1, int(k_step))) * max(1, int(k_step)))
    return int(np.clip(k, max(1, int(k_step)), int(N)))

def _fig5_k_values(N, k_step, fracs, crit_row=None):
    vals = [_round_k_to_grid(frac, N, k_step) for frac in fracs]
    if crit_row is not None:
        vals.append(_round_k_to_grid(float(crit_row['k_frac']), N, k_step))
    return sorted(set((int(v) for v in vals if 0 < int(v) <= int(N))))

def _fig5_group_key(strategy, k):
    return f'{strategy} | k={int(k)}'

def _parse_fig5_group_key(label):
    if ' | k=' not in str(label):
        return (str(label), None)
    strategy, k_txt = str(label).split(' | k=', 1)
    try:
        return (strategy, int(k_txt))
    except ValueError:
        return (strategy, None)

def _fig5_control_groups_for_k(N, order_out, order_low, k, random_repeats, seed_base):
    k = int(np.clip(k, 1, N))
    mid_start = max(0, (N - k) // 2)
    rg = np.random.default_rng(int(seed_base) + int(k) * 37)
    return {_fig5_group_key('Top out', k): [sorted(order_out[:k].tolist())], _fig5_group_key('Middle out', k): [sorted(order_out[mid_start:mid_start + k].tolist())], _fig5_group_key('Low out', k): [sorted(order_low[:k].tolist())], _fig5_group_key('Random', k): [sorted(rg.choice(N, size=k, replace=False).tolist()) for _ in range(int(random_repeats))]}

def run_fig5(A_shared=None):
    print('\n[Fig 5] Node-selection factors: k, out-strength, and z_D...')
    sd = DIRS['fig5']
    cleanup_outputs(sd, ['fig5f'])
    A_raw = make_directed_ba(FIG5_BA_N, FIG5_BA_M, FIG5_BA_SEED) if A_shared is None else np.asarray(A_shared, dtype=float)
    A, net_info = benchmark_operating_point_info(A_raw, 'fig5_ba')
    N = A.shape[0]
    w = degree_weights(A)
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    z_eff_target = analytic_nonoscillatory_target(theory)
    out_deg = A.sum(axis=0)
    order_out = np.argsort(-out_deg)
    order_low = np.argsort(out_deg)
    k_step = max(1, int(FIG5_K_STEP))
    k_grid = np.unique(np.concatenate([np.arange(0, N + 1, k_step, dtype=int), [N]]))
    seed_base = 55000
    dose_main = float(driver_z_from_dose(z_eff_target, FIG5_REFERENCE_DOSE))
    group_colors = {'Top out': COLOR_STRATEGY['out'], 'Middle out': COLOR_THEORY_FP, 'Low out': COLOR_STRATEGY['in'], 'Random': COLOR_STRATEGY['random']}
    print('  fig5a: output-strength rank and W_D concentration...')
    fig, ax = mfig((8, 3.6))
    sorted_outdeg = out_deg[order_out]
    colors_bar = []
    for val in np.linspace(1.0, 0.0, N):
        rgba = CMAP_OUTSTRENGTH(float(val))
        colors_bar.append((rgba[0], rgba[1], rgba[2], 0.92))
    ax.bar(range(N), sorted_outdeg, color=colors_bar, edgecolor='none', width=1.0)
    ax2 = ax.twinx()
    cum_w = np.cumsum(w[order_out])
    n_wd_pts = min(24, N)
    wd_rank_pts = np.unique(np.round(np.linspace(0, N - 1, n_wd_pts)).astype(int))
    ax2.plot(wd_rank_pts, cum_w[wd_rank_pts], color=COLOR_XW, lw=1.35, marker='o', ms=3.0, mfc=COLOR_XW, mec='white', mew=0.35, alpha=0.94)
    ax.set_xlabel('Node rank (by out-strength, descending)')
    ax.set_ylabel('Out-strength')
    ax2.set_ylabel('Cumulative $W_D$', color=COLOR_XW)
    ax2.tick_params(axis='y', colors=COLOR_XW)
    ax.set_xlim(-1, N)
    ax2.set_ylim(0, 1.03)
    save_sub(fig, sd, 'fig5a')
    print('  fig5b: node-count scan at fixed Δz_D...')
    summary_top = _run_ordering_scan(A, w, order_out, k_grid, z_eff_target, seed_base, 'top-out', fixed_dose=FIG5_REFERENCE_DOSE)
    summary_low = _run_ordering_scan(A, w, order_low, k_grid, z_eff_target, seed_base, 'low-out', fixed_dose=FIG5_REFERENCE_DOSE)
    random_summaries = []
    for rep in range(FIG5_N_RANDOM_REPEATS):
        rg_r = np.random.default_rng(seed_base + 20000 + rep)
        order_rand = rg_r.permutation(N)
        random_summaries.append(_run_ordering_scan(A, w, order_rand, k_grid, z_eff_target, seed_base, f'random_{rep}', fixed_dose=FIG5_REFERENCE_DOSE))
    summary_rand = []
    for ki, k in enumerate(k_grid):
        probs = [random_summaries[rep][ki]['success_probability'] for rep in range(FIG5_N_RANDOM_REPEATS)]
        wds = [random_summaries[rep][ki]['W_ctrl'] for rep in range(FIG5_N_RANDOM_REPEATS)]
        summary_rand.append({'k': int(k), 'k_frac': float(k / N), 'success_probability': float(np.mean(probs)), 'success_probability_std': float(np.std(probs)), 'W_ctrl': float(np.mean(wds)), 'W_ctrl_std': float(np.std(wds))})
    crit_top = find_k_eff_success(summary_top, CONTROL_PROB_THRESHOLD)
    crit_low = find_k_eff_success(summary_low, CONTROL_PROB_THRESHOLD)
    crit_rand = find_k_eff_success(summary_rand, CONTROL_PROB_THRESHOLD)
    fig, ax = mfig((7, 4.5))
    ax.plot([r['k_frac'] for r in summary_top], [r['success_probability'] for r in summary_top], color=COLOR_STRATEGY['out'], lw=2.0, marker='o', ms=3.0, label='Top out-strength', zorder=3)
    ax.plot([r['k_frac'] for r in summary_low], [r['success_probability'] for r in summary_low], color=COLOR_STRATEGY['in'], lw=1.45, ls='--', marker='s', ms=2.8, label='Low out-strength')
    p_rand = [r['success_probability'] for r in summary_rand]
    p_rand_std = [r['success_probability_std'] for r in summary_rand]
    kf_rand = [r['k_frac'] for r in summary_rand]
    ax.plot(kf_rand, p_rand, color=COLOR_STRATEGY['random'], lw=1.25, ls=':', marker='^', ms=2.8, label='Random')
    ax.fill_between(kf_rand, [max(0, p - s) for p, s in zip(p_rand, p_rand_std)], [min(1, p + s) for p, s in zip(p_rand, p_rand_std)], color=COLOR_STRATEGY['random'], alpha=0.16)
    ax.axhline(CONTROL_PROB_THRESHOLD, color='#767676', ls='--', lw=0.9)
    for crit_row, color, label in [(crit_top, COLOR_STRATEGY['out'], 'top'), (crit_low, COLOR_STRATEGY['in'], 'low'), (crit_rand, COLOR_STRATEGY['random'], 'random')]:
        if crit_row is not None:
            ax.axvline(crit_row['k_frac'], color=color, ls='-.', lw=0.9, alpha=0.75)
            ax.text(crit_row['k_frac'], 0.05, f'$k_{{eff}}$ {label}', color=color, fontsize=6.5, rotation=90, va='bottom', ha='right')
    ax.set_xlim(0, 1.02)
    ax.set_xlabel('Fraction of controlled nodes $k/N$')
    ax.set_ylabel(f'Transition probability at $|\\Delta z_D|={FIG5_REFERENCE_DOSE:g}$')
    ax.set_ylim(-0.05, 1.05)
    legend_clean(ax, loc='lower right', fontsize=7, handlelength=2.6, borderpad=0.25, labelspacing=0.35)
    save_sub(fig, sd, 'fig5b')
    print('  fig5c: output-strength effect across transition-range k...')
    k_probe_values = _fig5_k_values(N, k_step, FIG5_GROUP_K_FRACS, crit_top)
    fixed_group_sets = {}
    for k_probe in k_probe_values:
        fixed_group_sets.update(_fig5_control_groups_for_k(N, order_out, order_low, k_probe, FIG5_N_RANDOM_REPEATS, seed_base + 30000))
    fixed_strength_summary = _run_grouped_control_sets(A, w, fixed_group_sets, dose_main, seed_base + 30000)
    for row in fixed_strength_summary:
        strategy, k_val = _parse_fig5_group_key(row['label'])
        row['strategy'] = strategy
        row['k_probe'] = int(k_val or row['k'])
        row['label_full'] = row['label']
        row['label'] = strategy
    fig, ax = mfig((7.1, 4.7))
    strategy_order = ['Top out', 'Middle out', 'Low out', 'Random']
    for strategy in strategy_order:
        rows_s = sorted([r for r in fixed_strength_summary if r['strategy'] == strategy], key=lambda r: r['k_probe'])
        if not rows_s:
            continue
        xs_ = [r['W_ctrl'] for r in rows_s]
        ys_ = [r['success_probability'] for r in rows_s]
        ax.plot(xs_, ys_, color=group_colors[strategy], lw=1.1, alpha=0.45)
        for row in rows_s:
            ax.errorbar(row['W_ctrl'], row['success_probability'], xerr=row['W_ctrl_std'], yerr=row['success_probability_std'], fmt='o', ms=4.8 + 3.2 * row['k_frac'], color=group_colors[strategy], mec='k', mew=0.32, capsize=2.2, alpha=0.92)
        if rows_s:
            ax.text(rows_s[-1]['W_ctrl'] + 0.015, rows_s[-1]['success_probability'], strategy, ha='left', va='center', fontsize=6.4, color=group_colors[strategy])
    ax.axhline(CONTROL_PROB_THRESHOLD, color='#767676', ls='--', lw=0.9)
    for k_probe in k_probe_values:
        ax.text(0.015, min(1.04, 0.11 + 0.055 * k_probe_values.index(k_probe)), f'$k/N={k_probe / N:.2f}$', fontsize=5.9, color='#555555', ha='left', va='center')
    ax.set_xlabel('Cumulative output weight $W_D$ at fixed $|\\Delta z_D|$')
    ax.set_ylabel('Transition success probability')
    ax.set_title(f'Output-strength groups near transition, $|\\Delta z_D|={FIG5_REFERENCE_DOSE:g}$', fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.08)
    save_sub(fig, sd, 'fig5c')
    print('  fig5d: top-out driver-dose response at multiple k...')
    dose_k_values = _fig5_k_values(N, k_step, FIG5_DOSE_K_FRACS, crit_top)
    dose_response = []
    for dose in sorted(set((float(v) for v in FIG5_DOSE_GRID))):
        z_driver = float(driver_z_from_dose(z_eff_target, dose))
        dose_group_sets = {_fig5_group_key('Top out', k_val): [sorted(order_out[:int(k_val)].tolist())] for k_val in dose_k_values}
        rows = _run_grouped_control_sets(A, w, dose_group_sets, z_driver, seed_base + 40000)
        for row in rows:
            strategy, k_val = _parse_fig5_group_key(row['label'])
            row['strategy'] = strategy
            row['k_probe'] = int(k_val or row['k'])
            row['label_full'] = row['label']
            row['label'] = strategy
            row['dose'] = float(abs(dose))
            dose_response.append(row)
    fig, ax = mfig((6.9, 4.5))
    dose_colors = {k_val: CMAP_OUTSTRENGTH(0.18 + 0.72 * i / max(1, len(dose_k_values) - 1)) for i, k_val in enumerate(dose_k_values)}
    for k_val in dose_k_values:
        rows_label = sorted([r for r in dose_response if r['k_probe'] == k_val], key=lambda r: r['dose'])
        xs_ = [r['dose'] for r in rows_label]
        ys_ = [r['success_probability'] for r in rows_label]
        yerr = [r['success_probability_std'] for r in rows_label]
        color = dose_colors[k_val]
        ax.plot(xs_, ys_, color=color, lw=1.45, marker='o', ms=3.2, label=f'$k/N={k_val / N:.2f}$')
        if any((v > 0 for v in yerr)):
            ax.fill_between(xs_, [max(0, y - e) for y, e in zip(ys_, yerr)], [min(1, y + e) for y, e in zip(ys_, yerr)], color=color, alpha=0.13)
    ax.axhline(CONTROL_PROB_THRESHOLD, color='#767676', ls='--', lw=0.9)
    ax.set_xlabel('Driver dose $|\\Delta z_D|$')
    ax.set_ylabel('Transition success probability')
    ax.set_ylim(-0.05, 1.05)
    legend_clean(ax, loc='lower right', ncol=2, fontsize=6.5)
    save_sub(fig, sd, 'fig5d')
    print('  fig5e: top-out k/N x driver-dose control landscape...')
    landscape_doses = sorted(set((float(v) for v in FIG5_DOSE_GRID)))
    landscape_seed = seed_base + 50000
    baseline_tasks = [(A, w, landscape_seed, trial) for trial in range(N_CONTROL_PROB_TRIALS)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        baseline_rows = list(pool.map(_worker_baseline_probability, baseline_tasks, chunksize=1))
    baseline_by_trial = {r['trial']: r for r in baseline_rows}
    fig5_landscape = []
    for dose in landscape_doses:
        z_driver = float(driver_z_from_dose(z_eff_target, dose))
        schedule_dose = {}
        dose_tasks = []
        for k in k_grid:
            k_int = int(k)
            ctrl = sorted(order_out[:k_int].tolist()) if k_int > 0 else []
            W_ctrl = driver_weight(w, ctrl)
            schedule_dose[k_int] = {'drivers': ctrl, 'W_ctrl': float(W_ctrl), 'z_driver': float(z_driver) if k_int > 0 else np.nan, 'z_eff_realized': realized_z_eff_from_driver(W_ctrl, z_driver)}
            for trial in range(N_CONTROL_PROB_TRIALS):
                base = baseline_by_trial[int(trial)]
                dose_tasks.append((A, w, ctrl, schedule_dose[k_int]['z_driver'], landscape_seed, trial, base['baseline_osc'], base['baseline_tail_amp'], k_int))
        chunksize = max(1, len(dose_tasks) // max(1, N_WORKERS * 4))
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            dose_results = list(pool.map(_worker_control_probability, dose_tasks, chunksize=chunksize))
        dose_summary = []
        for k in k_grid:
            k_int = int(k)
            rows = [r for r in dose_results if r['k'] == k_int]
            sched = schedule_dose[k_int]
            probability = summarize_control_probability(rows)
            dose_summary.append({'k': k_int, 'k_frac': float(k_int / N), **probability, 'W_ctrl': float(sched['W_ctrl']), 'z_driver': float(sched['z_driver']) if np.isfinite(sched['z_driver']) else None, 'z_eff_realized': float(sched['z_eff_realized'])})
        fig5_landscape.append({'dose': float(dose), 'summary': dose_summary})
    doses_e = np.array([row['dose'] for row in fig5_landscape], dtype=float)
    kf_e = np.array([row['k_frac'] for row in fig5_landscape[0]['summary']], dtype=float)
    P_e = np.array([[rr['success_probability'] for rr in row['summary']] for row in fig5_landscape], dtype=float)
    fig, ax = mfig((7.0, 4.8))
    im = ax.contourf(kf_e, doses_e, P_e, levels=SUCCESS_LEVELS, vmin=0, vmax=1, cmap=CMAP_SUCCESS)
    success_mask_e = (P_e >= CONTROL_PROB_THRESHOLD - 1e-12).astype(float)
    if np.nanmin(success_mask_e) < 0.5 < np.nanmax(success_mask_e):
        ax.contour(kf_e, doses_e, success_mask_e, levels=[0.5], colors='#272727', linewidths=1.05)
    if doses_e.min() - 1e-12 <= FIG5_REFERENCE_DOSE <= doses_e.max() + 1e-12:
        ax.axhline(FIG5_REFERENCE_DOSE, color='#6E6E6E', ls=':', lw=0.9)
        if crit_top is not None:
            ax.scatter([crit_top['k_frac']], [FIG5_REFERENCE_DOSE], s=34, facecolors='white', edgecolors=COLOR_STRATEGY['out'], linewidths=1.0, zorder=5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(float(doses_e.min()), float(doses_e.max()))
    ax.set_xlabel('Controlled fraction $k/N$')
    ax.set_ylabel('Driver dose $|\\Delta z_D|$')
    ax.set_title('Top out-strength control landscape', fontsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    cbar.set_label('$P_{\\mathrm{success}}$', fontsize=7)
    save_sub(fig, sd, 'fig5e')
    effective_input_collapse = []
    for dose_row in fig5_landscape:
        dose = float(dose_row['dose'])
        for row in dose_row['summary']:
            effective_input_collapse.append({'strategy': 'Top out', 'source': 'fig5e_landscape', 'k': int(row['k']), 'k_frac': float(row['k_frac']), 'dose': dose, 'W_ctrl': float(row['W_ctrl']), 'effective_input': float(abs(row['W_ctrl'] * dose)), 'success_probability': float(row['success_probability'])})
    save_json({'N': int(N), 'A_eff': float(Ae), 'z_eff_target': float(z_eff_target), 'k_probe_values': [int(k) for k in k_probe_values], 'k_probe_fracs': [float(k / N) for k in k_probe_values], 'dose_k_values': [int(k) for k in dose_k_values], 'dose_k_fracs': [float(k / N) for k in dose_k_values], 'reference_driver_dose': FIG5_REFERENCE_DOSE, 'fig5_dose_grid': list(FIG5_DOSE_GRID), 'fig5_group_k_fracs': list(FIG5_GROUP_K_FRACS), 'fig5_dose_k_fracs': list(FIG5_DOSE_K_FRACS), 'control_input_policy': CONTROL_INPUT_POLICY, 'threshold': CONTROL_PROB_THRESHOLD, 'strict_threshold': CONTROL_PROB_STRICT_THRESHOLD, 'strategy_thresholds': {'top_out': crit_top, 'low_out': crit_low, 'random': crit_rand}, 'node_count_scan': {'top_out': summary_top, 'low_out': summary_low, 'random_mean': summary_rand, 'random_replicates': random_summaries}, 'fixed_k_strength_groups': fixed_strength_summary, 'top_out_dose_response': dose_response, 'dose_response': dose_response, 'joint_control_space': fig5_landscape, 'effective_input_collapse': effective_input_collapse, 'control_space_interpretation': {'x_axis': 'controlled node fraction k/N', 'y_axis': 'driver dose |Δz_D|', 'color': 'transition success probability', 'strategy': 'top out-strength', 'success_definition': f'P_success >= {CONTROL_PROB_THRESHOLD:g}', 'interpretation': 'successful control is a joint high-coverage/high-effective-input region rather than a single-axis collapse'}, 'threshold_sensitivity': {'top_out': success_threshold_table(summary_top), 'low_out': success_threshold_table(summary_low), 'random': success_threshold_table(summary_rand)}, 'network_operating_point': net_info}, os.path.join(sd, 'fig5_summary.json'))
    print('[Fig 5] Done.\n')

def _clean_empirical_connectome(A):
    A = np.asarray(A, dtype=float).copy()
    A[~np.isfinite(A)] = 0.0
    A[A < 0.0] = 0.0
    np.fill_diagonal(A, 0.0)
    return A

def load_full_drosophila_connectome():
    import zipfile, csv, io
    zip_path = os.path.join(os.path.dirname(__file__), 'data', 'Supplementary-Data-S1.zip')
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f'Missing Drosophila data: {zip_path}')
    z = zipfile.ZipFile(zip_path)
    f_ann = z.read('Supplementary-Data-S1/annotations.csv').decode('utf-8')
    reader = csv.reader(io.StringIO(f_ann))
    ann_rows = [r for r in reader]
    id_to_celltype = {}
    id_to_annotation = {}
    for r in ann_rows[1:]:
        ct, ann = (r[2], r[3])
        if r[0] != 'no pair':
            id_to_celltype[r[0]] = ct
            id_to_annotation[r[0]] = ann
        if r[1] != 'no pair':
            id_to_celltype[r[1]] = ct
            id_to_annotation[r[1]] = ann
    f_mat = z.read('Supplementary-Data-S1/all-all_connectivity_matrix.csv').decode('utf-8')
    reader2 = csv.reader(io.StringIO(f_mat))
    mat_rows = [r for r in reader2]
    mat_ids = mat_rows[0][1:]
    n = len(mat_ids)
    source_by_target = np.zeros((n, n), dtype=float)
    for i, row in enumerate(mat_rows[1:]):
        for j, val in enumerate(row[1:]):
            source_by_target[i, j] = float(val)
    A = source_by_target.T
    A = _clean_empirical_connectome(A)
    if not USE_RAW_EMPIRICAL_WEIGHTS:
        wsum = float(A.sum())
        if wsum > 0:
            A = A / wsum
    region_groups = {'motor': [], 'sensory_all': [], 'sensory_visual': [], 'sensory_olfactory': [], 'sensory_gustatory': [], 'mushroom_body': [], 'interneuron': [], 'projection': [], 'descending_sez': []}
    motor_types = {'DN-VNC', 'pre-DN-VNC'}
    mb_types = {'KC', 'MBON', 'MBIN', 'MB-FBN', 'MB-FFN'}
    inter_types = {'LHN', 'LN', 'CN', 'RGN'}
    proj_types = {'PN', 'PN-somato'}
    sez_types = {'DN-SEZ', 'pre-DN-SEZ'}
    for idx, mid in enumerate(mat_ids):
        ct = id_to_celltype.get(mid, 'unknown')
        ann = id_to_annotation.get(mid, '').lower()
        if ct in motor_types:
            region_groups['motor'].append(idx)
        elif ct == 'sensory':
            region_groups['sensory_all'].append(idx)
            if 'visual' in ann:
                region_groups['sensory_visual'].append(idx)
            elif 'olfactory' in ann:
                region_groups['sensory_olfactory'].append(idx)
            elif 'gustatory' in ann:
                region_groups['sensory_gustatory'].append(idx)
        elif ct in mb_types:
            region_groups['mushroom_body'].append(idx)
        elif ct in inter_types:
            region_groups['interneuron'].append(idx)
        elif ct in proj_types:
            region_groups['projection'].append(idx)
        elif ct in sez_types:
            region_groups['descending_sez'].append(idx)
    z.close()
    return (A, mat_ids, id_to_celltype, id_to_annotation, region_groups)

def _normalise_empirical_weights(A):
    A = _clean_empirical_connectome(A)
    if USE_RAW_EMPIRICAL_WEIGHTS:
        return A
    wsum = float(np.sum(A)) if A.size else 0.0
    if wsum > 0:
        A = A / wsum
    return A

def _np_scalar_to_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)

def _npz_first(data, keys):
    for key in keys:
        if key in data.files:
            return (key, data[key])
    return (None, None)

def _normalise_group_name(name):
    return str(name).strip().lower().replace(' ', '_').replace('-', '_')

def _coerce_region_groups(value, n_nodes):
    obj = value
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            obj = obj.item()
        else:
            obj = obj.tolist()
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    if isinstance(obj, dict):
        iterator = obj.items()
    elif isinstance(obj, (list, tuple)):
        iterator = [(item[0], item[1]) for item in obj if isinstance(item, (list, tuple)) and len(item) == 2]
    else:
        return None
    groups = {}
    for key, vals in iterator:
        arr = np.asarray(vals, dtype=int).ravel()
        arr = arr[(arr >= 0) & (arr < int(n_nodes))]
        if arr.size:
            groups[_normalise_group_name(key)] = sorted(set(arr.astype(int).tolist()))
    return groups if groups else None

def _groups_from_node_regions(node_regions):
    labels = [str(v) for v in np.asarray(node_regions).ravel().tolist()]
    groups = {}
    for idx, label in enumerate(labels):
        key = _normalise_group_name(label)
        if key and key not in {'none', 'nan', 'unknown'}:
            groups.setdefault(key, []).append(int(idx))
    return groups if groups else None

def _orient_external_directed_matrix(A, orientation):
    orientation = _normalise_group_name(orientation or 'code_target_by_source')
    source_by_target = {'source_by_target', 'source_target', 'source_rows_target_cols', 'rows_source_cols_target', 'src_by_dst', 'source_to_target'}
    target_by_source = {'target_by_source', 'target_source', 'target_rows_source_cols', 'rows_target_cols_source', 'code_target_by_source', 'dst_by_src'}
    if orientation in source_by_target:
        return (np.asarray(A, dtype=float).T, 'source_by_target_transposed')
    if orientation in target_by_source:
        return (np.asarray(A, dtype=float), 'code_target_by_source')
    raise ValueError(f"Unsupported directed matrix orientation '{orientation}'. Use orientation='source_by_target' for rows=source, cols=target, or orientation='code_target_by_source' for A[target, source].")

def _directed_npz_expected_message(dataset):
    filename = f'data/{dataset}_connectome.npz'
    return f"Missing directed Fig. 6 connectome file: {filename}. Expected a preprocessed npz with a square matrix key such as 'A', 'Cmat', 'projection_matrix', or 'connectivity_matrix'. Include orientation='source_by_target' if rows are source regions and columns are targets, or orientation='code_target_by_source' if A[target, source] already follows this codebase. Optional fields: node_labels, node_regions, region_groups, source_dataset."

def _load_directed_region_connectome_npz(dataset, label, analysis_level, default_source):
    path = os.path.join(os.path.dirname(__file__), 'data', f'{dataset}_connectome.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(_directed_npz_expected_message(dataset))
    data = np.load(path, allow_pickle=True)
    matrix_key, A_raw = _npz_first(data, ('A', 'Cmat', 'projection_matrix', 'connectivity_matrix', 'connectome', 'matrix', 'W'))
    if A_raw is None:
        raise KeyError(_directed_npz_expected_message(dataset))
    A_raw = np.asarray(A_raw, dtype=float)
    if A_raw.ndim != 2 or A_raw.shape[0] != A_raw.shape[1]:
        raise ValueError(f"{path} field '{matrix_key}' must be a square matrix.")
    orientation = _np_scalar_to_str(data['orientation']) if 'orientation' in data.files else 'code_target_by_source'
    A, orientation_used = _orient_external_directed_matrix(A_raw, orientation)
    A = _normalise_empirical_weights(A)
    n = A.shape[0]
    _, node_labels = _npz_first(data, ('node_labels', 'labels', 'region_labels', 'names'))
    if node_labels is not None:
        node_labels = [str(v) for v in np.asarray(node_labels).ravel().tolist()]
    _, node_names = _npz_first(data, ('node_names', 'long_names', 'structure_names'))
    if node_names is not None:
        node_names = [str(v) for v in np.asarray(node_names).ravel().tolist()]
    _, node_structure_ids = _npz_first(data, ('node_structure_ids', 'structure_ids'))
    if node_structure_ids is not None:
        node_structure_ids = [int(v) for v in np.asarray(node_structure_ids).ravel().tolist()]
    _, node_regions = _npz_first(data, ('node_regions', 'system_labels', 'coarse_regions', 'region_systems'))
    region_groups = _coerce_region_groups(data['region_groups'], n) if 'region_groups' in data.files else None
    if region_groups is None and node_regions is not None and (len(np.asarray(node_regions).ravel()) == n):
        region_groups = _groups_from_node_regions(node_regions)
    source_dataset = _np_scalar_to_str(data['source_dataset']) if 'source_dataset' in data.files else _np_scalar_to_str(data['dataset']) if 'dataset' in data.files else default_source
    return {'key': dataset, 'label': label, 'A': A, 'analysis_level': analysis_level, 'comparison_role': 'directed empirical brain transfer', 'source_dataset': source_dataset, 'region_groups': region_groups, 'node_labels': node_labels, 'node_names': node_names, 'node_structure_ids': node_structure_ids, 'matrix_key': matrix_key, 'matrix_orientation': orientation_used, 'data_path': os.path.relpath(path, os.path.dirname(__file__))}

def _make_drosophila_full_entry():
    A, mat_ids, id_to_celltype, id_to_annotation, region_groups = load_full_drosophila_connectome()
    return {'key': 'drosophila_full', 'label': 'Drosophila full', 'A': A, 'analysis_level': 'Drosophila cell-scale directed synaptic connectome', 'comparison_role': 'directed empirical brain transfer', 'source_dataset': 'Winding et al. larval connectome full graph', 'matrix_key': 'all-all_connectivity_matrix.csv', 'matrix_orientation': 'source_by_target_transposed_to_target_by_source', 'region_groups': region_groups, 'node_labels': [str(v) for v in mat_ids], 'mat_ids': mat_ids, 'id_to_celltype': id_to_celltype, 'id_to_annotation': id_to_annotation}

def load_empirical_brain_registry():
    allen = _load_directed_region_connectome_npz('allen_mouse', 'Allen mouse', 'mouse mesoscale directed projection connectome', 'Allen Mouse Brain Connectivity Atlas')
    marmoset = _load_directed_region_connectome_npz('marmoset', 'Marmoset', 'marmoset directed tracer projection connectome', 'Marmoset Brain Connectivity Atlas')
    if marmoset.get('region_groups') is None:
        raise ValueError("data/marmoset_connectome.npz must include 'region_groups' (dict-like region -> node indices) or 'node_regions' (one coarse brain-region label per node) for Fig. 6e-f.")
    return [allen, marmoset, _make_drosophila_full_entry()]

def _run_empirical_topk_analysis(entry, seed_base):
    key = entry['key']
    label = entry['label']
    A, net_info = network_operating_point_info(entry['A'], f'fig6_{key}')
    N = A.shape[0]
    w = degree_weights(A)
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    z_eff_target = choose_nonoscillatory_target(theory)
    empirical_interval_scan = _run_empirical_interval_scan(A, theory, seed_base)
    n_baseline = max(FIG6_TOPK_TRIALS, FIG6_LANDSCAPE_TRIALS)
    baseline_tasks = [(A, w, seed_base, trial) for trial in range(n_baseline)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        baseline_rows = list(pool.map(_worker_baseline_probability, baseline_tasks, chunksize=1))
    baseline_by_trial = {r['trial']: r for r in baseline_rows}
    valid_baseline_rows = [r for r in baseline_rows if r.get('baseline_valid', True)]
    baseline_osc_rate = float(np.mean([r['baseline_osc'] for r in valid_baseline_rows])) if valid_baseline_rows else np.nan
    baseline_invalid_count = int(len(baseline_rows) - len(valid_baseline_rows))
    out_deg = A.sum(axis=0)
    out_order = np.argsort(-out_deg)
    topk_fracs = sorted(set([0.0, 1.0] + [float(np.clip(frac, 0.0, 1.0)) for frac in FIG6_TOPK_FRACS]))
    k_grid = np.array(sorted(set((int(round(frac * N)) for frac in topk_fracs))), dtype=int)
    schedule = []
    for k in k_grid:
        ctrl = sorted(out_order[:int(k)].tolist()) if k > 0 else []
        WD = driver_weight(w, ctrl)
        schedule.append({'k': int(k), 'k_frac': float(k / N), 'drivers': ctrl, 'W_ctrl': float(WD), 'z_driver': float(driver_z_from_dose(z_eff_target, MAX_DRIVER_Z)) if k > 0 else np.nan, 'z_eff_realized': realized_z_eff_from_driver(WD, driver_z_from_dose(z_eff_target, MAX_DRIVER_Z))})
    tasks = []
    for info in schedule:
        for trial in range(FIG6_TOPK_TRIALS):
            base = baseline_by_trial[int(trial)]
            tasks.append((A, w, info['drivers'], info['z_driver'], seed_base, trial, base['baseline_osc'], base['baseline_tail_amp'], info['k']))
    chunksize = max(1, len(tasks) // max(1, N_WORKERS * 4))
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_worker_control_probability, tasks, chunksize=chunksize))
    schedule_by_k = {s['k']: s for s in schedule}
    topk_summary = []
    for k in k_grid:
        rows = [r for r in results if r['k'] == int(k)]
        sched = schedule_by_k[int(k)]
        amps = [r['x_tail_amp'] for r in rows if np.isfinite(r['x_tail_amp'])]
        probability = summarize_control_probability(rows)
        topk_summary.append({'k': int(k), 'k_frac': float(k / N), **probability, 'mean_tail_amp': float(np.mean(amps)) if amps else np.nan, 'W_ctrl': float(sched['W_ctrl']), 'z_driver': float(sched['z_driver']) if np.isfinite(sched['z_driver']) else None, 'z_eff_realized': float(sched['z_eff_realized'])})
    topk_landscape = []
    dose_grid = sorted(set([float(v) for v in FIG6_LANDSCAPE_DOSE_GRID] + [float(MAX_DRIVER_Z)]))
    for dose in dose_grid:
        if abs(dose - MAX_DRIVER_Z) < 1e-12:
            dose_summary = topk_summary
        else:
            dose_tasks = []
            for info in schedule:
                z_driver = float(driver_z_from_dose(z_eff_target, dose)) if info['k'] > 0 else np.nan
                for trial in range(FIG6_LANDSCAPE_TRIALS):
                    base = baseline_by_trial[int(trial)]
                    dose_tasks.append((A, w, info['drivers'], z_driver, seed_base, trial, base['baseline_osc'], base['baseline_tail_amp'], info['k'], dose))
            chunksize = max(1, len(dose_tasks) // max(1, N_WORKERS * 4))
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                dose_results = list(pool.map(_worker_control_probability_dose, dose_tasks, chunksize=chunksize))
            dose_summary = []
            for info in schedule:
                rows = [r for r in dose_results if r['k'] == int(info['k'])]
                amps = [r['x_tail_amp'] for r in rows if np.isfinite(r['x_tail_amp'])]
                probability = summarize_control_probability(rows)
                dose_summary.append({'k': int(info['k']), 'k_frac': float(info['k_frac']), **probability, 'mean_tail_amp': float(np.mean(amps)) if amps else np.nan, 'W_ctrl': float(info['W_ctrl']), 'z_driver': float(driver_z_from_dose(z_eff_target, dose)) if info['k'] > 0 else None, 'z_eff_realized': realized_z_eff_from_driver(info['W_ctrl'], driver_z_from_dose(z_eff_target, dose))})
        crit_dose = find_k_eff_success(dose_summary, CONTROL_PROB_THRESHOLD)
        topk_landscape.append({'dose': float(dose), 'k_eff_frac': float(crit_dose['k_frac']) if crit_dose else np.nan, 'summary': dose_summary})
    topk_crit = find_k_eff_success(topk_summary, CONTROL_PROB_THRESHOLD)
    topk_threshold_reached = topk_crit is not None
    topk_k_eff = int(topk_crit['k']) if topk_threshold_reached else None
    info = {'key': key, 'label': label, 'analysis_level': entry.get('analysis_level', 'empirical network'), 'comparison_role': entry.get('comparison_role', 'horizontal transfer dynamics'), 'source_dataset': entry.get('source_dataset', key), 'N': int(N), 'edges': int(np.count_nonzero(A)), 'density': float(np.count_nonzero(A) / max(N * (N - 1), 1)), 'A_eff': float(Ae), 'z_interval': {'left': float(theory['z_left']), 'right': float(theory['z_right']), 'width': float(theory['width'])}, 'empirical_z_interval': empirical_interval_scan['interval'], 'empirical_zscan': empirical_interval_scan, 'zero_in_interval': bool(theory['z_left'] <= 0.0 <= theory['z_right']), 'z_eff_target': float(z_eff_target), 'W_D_required_at_reference_dose': float(abs(z_eff_target - OSCILLATORY_Z) / MAX_DRIVER_Z) if MAX_DRIVER_Z > 0 else np.nan, 'baseline_osc_rate': baseline_osc_rate, 'baseline_invalid_count': baseline_invalid_count, 'topk_summary': topk_summary, 'topk_dose_landscape': topk_landscape, 'topk_threshold_reached': bool(topk_threshold_reached), 'topk_k_eff': topk_k_eff, 'topk_k_eff_fraction': float(topk_k_eff / N) if topk_threshold_reached else np.nan, 'network_operating_point': net_info, 'matrix_key': entry.get('matrix_key'), 'matrix_orientation': entry.get('matrix_orientation', 'code_target_by_source'), 'node_labels': entry.get('node_labels'), 'node_names': entry.get('node_names'), 'node_structure_ids': entry.get('node_structure_ids')}
    runtime = {'A': A, 'w': w, 'out_order': out_order, 'z_eff_target': float(z_eff_target), 'region_groups': entry.get('region_groups'), 'node_labels': entry.get('node_labels'), 'node_names': entry.get('node_names'), 'node_structure_ids': entry.get('node_structure_ids'), 'seed_base': int(seed_base)}
    return (info, runtime)

def _plot_sorted_connectivity_matrix(ax, A, label, color, meta=None):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    out_strength = A.sum(axis=0)
    order = np.argsort(-out_strength)
    M = A[np.ix_(order, order)] if n else A
    vals = M[M > 0]
    vmax = float(np.percentile(vals, 97)) if vals.size else 1.0
    cmap = LinearSegmentedColormap.from_list(f'{safe_label(label)}_matrix', ['#FFFFFF', '#E8EEF5', color])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=max(vmax, 1e-12), interpolation='nearest', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=7.5, color=color, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color('#D0D0D0')
    if meta:
        ax.text(0.02, 0.98, meta, transform=ax.transAxes, ha='left', va='top', fontsize=5.8, color='#333333', bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.78))

def _fig6_region_label(name):
    return FIG6_REGION_LABEL.get(name, safe_label(name).title())

def _fig6_ordered_regions(region_groups, region_results=None):
    regions = [r for r, nodes in region_groups.items() if len(nodes) > 0]
    if region_results is not None:
        regions = [r for r in regions if r in region_results]
        return sorted(regions, key=lambda r: region_results[r]['W_D'], reverse=True)
    return sorted(regions, key=lambda r: _fig6_region_label(r))

def _compute_region_reachability(A, w, region_groups, z_eff_target):
    N = A.shape[0]
    region_results = {}
    for rname in _fig6_ordered_regions(region_groups):
        ctrl = sorted(set((int(i) for i in region_groups.get(rname, []) if 0 <= int(i) < N)))
        info = solve_baseline_driver_z(w, ctrl, z_eff_target)
        region_results[rname] = {'n_nodes': len(ctrl), 'n_frac': float(len(ctrl) / max(N, 1)), 'W_D': float(info['W_D']), 'z_driver': float(info['z_driver']) if info['valid'] else np.nan, 'z_driver_unbounded': float(info['z_driver_unbounded']) if info['valid'] else np.nan, 'z_eff_realized': float(info['z_eff_realized']), 'target_feasible': bool(info.get('target_feasible', False)), 'reduced_coordinate_reachable': bool(info.get('target_feasible', False)), 'nodes': ctrl}
    return region_results

def _fig6_structural_top_drivers(w, out_order, z_eff_target):
    w_required = float(abs(z_eff_target - OSCILLATORY_Z) / MAX_DRIVER_Z) if MAX_DRIVER_Z > 0 else np.inf
    cum_w = np.cumsum(w[out_order]) if len(out_order) else np.array([], dtype=float)
    if len(cum_w) and np.any(cum_w >= w_required):
        k_eff = int(np.flatnonzero(cum_w >= w_required)[0] + 1)
    else:
        k_eff = int(len(out_order))
    drivers = sorted(out_order[:k_eff].astype(int).tolist()) if k_eff > 0 else []
    return (k_eff, drivers, w_required)

def _nodes_from_labels(node_labels, wanted):
    wanted = {str(v).lower() for v in wanted}
    return [int(i) for i, label in enumerate(node_labels or []) if str(label).lower() in wanted]

def _fig6_allen_epilepsy_targets(region_groups, node_labels):
    targets = {}
    ant_nodes = _nodes_from_labels(node_labels, {'AV', 'AM', 'AD', 'IAD'})
    if ant_nodes:
        targets['anterior_thalamic'] = {'label': 'ANT proxy', 'literature_link': 'anterior thalamic DBS', 'nodes': ant_nodes}
    hippocampal = list(region_groups.get('hippocampal', []))
    if hippocampal:
        targets['hippocampal'] = {'label': 'Hippocampal formation', 'literature_link': 'temporal-lobe/RNS or hippocampal seizure focus', 'nodes': sorted((int(v) for v in hippocampal))}
    lha_nodes = _nodes_from_labels(node_labels, {'LHA'})
    if lha_nodes:
        targets['lateral_hypothalamus'] = {'label': 'Lateral hypothalamus', 'literature_link': 'hypothalamic orexin/DBS seizure modulation', 'nodes': lha_nodes}
    cereb_nuclei = _nodes_from_labels(node_labels, {'FN', 'IP', 'DN'})
    if cereb_nuclei:
        targets['cerebellar_nuclei'] = {'label': 'Cerebellar nuclei', 'literature_link': 'cerebellar seizure modulation', 'nodes': cereb_nuclei}
    thalamus = list(region_groups.get('thalamus', []))
    if thalamus:
        targets['thalamus'] = {'label': 'Whole thalamus', 'literature_link': 'thalamic seizure-control network', 'nodes': sorted((int(v) for v in thalamus)), 'summary_only': True}
    return targets

def _unique_nodes(*groups):
    nodes = []
    for group in groups:
        nodes.extend((int(v) for v in group or []))
    return sorted(set(nodes))

def _fig6_allen_epilepsy_combo_targets(region_groups, single_targets):
    combos = {}
    thal = single_targets.get('thalamus', {}).get('nodes', [])
    hip = single_targets.get('hippocampal', {}).get('nodes', [])
    lha = single_targets.get('lateral_hypothalamus', {}).get('nodes', [])
    cereb = single_targets.get('cerebellar_nuclei', {}).get('nodes', [])
    iso = sorted((int(v) for v in region_groups.get('isocortex', [])))
    hypo = sorted((int(v) for v in region_groups.get('hypothalamus', [])))
    if thal and hip:
        combos['thalamo_hippocampal'] = {'label': 'Thalamus + hippocampal', 'literature_link': 'thalamic neuromodulation plus hippocampal/temporal-lobe targets', 'nodes': _unique_nodes(thal, hip)}
    if thal and iso:
        combos['thalamo_cortical'] = {'label': 'Thalamocortical', 'literature_link': 'thalamocortical seizure-network modulation', 'nodes': _unique_nodes(thal, iso)}
    if thal and (hypo or lha):
        combos['thalamo_hypothalamic'] = {'label': 'Thalamus + hypothalamus', 'literature_link': 'thalamic neuromodulation plus hypothalamic/orexin seizure modulation', 'nodes': _unique_nodes(thal, hypo, lha)}
    if thal and cereb:
        combos['thalamo_cerebellar'] = {'label': 'Thalamus + cerebellar nuclei', 'literature_link': 'thalamic neuromodulation plus cerebellar seizure-network modulation', 'nodes': _unique_nodes(thal, cereb)}
    if thal and iso and (hypo or lha):
        combos['thalamo_cortico_hypothalamic'] = {'label': 'Thalamo-cortico-hypothalamic', 'literature_link': 'distributed thalamocortical/hypothalamic seizure-control network', 'nodes': _unique_nodes(thal, iso, hypo, lha)}
    return combos

def run_fig6():
    print('\n[Fig 6] Directed empirical brain networks and region control...')
    sd = DIRS['fig6']
    entries = load_empirical_brain_registry()
    expected_keys = ['allen_mouse', 'marmoset', 'drosophila_full']
    entry_keys = [e['key'] for e in entries]
    if entry_keys != expected_keys:
        raise ValueError(f'Fig6 expects directed empirical networks in this order: {expected_keys}; got {entry_keys}.')
    analyses = []
    runtimes = {}
    for i, entry in enumerate(entries):
        print(f"  empirical network: {entry['label']}...")
        info, runtime = _run_empirical_topk_analysis(entry, 90000 + i * 20000)
        analyses.append(info)
        runtimes[entry['key']] = runtime
    print('  fig6a: empirical directed connectivity matrices...')
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.35))
    for ax, info in zip(axes, analyses):
        key = info['key']
        color = COLOR_REAL_BRAIN.get(key, '#888888')
        meta = f"N={info['N']}\nE={info['edges']}\nd={info['density']:.3f}\n{info.get('source_dataset', '')}"
        _plot_sorted_connectivity_matrix(ax, runtimes[key]['A'], info['label'], color, meta=meta)
    fig.suptitle('Directed empirical brain connectomes', fontsize=9.2, y=0.985)
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.1, top=0.84, wspace=0.28)
    fig._skip_tight_layout = True
    save_sub(fig, sd, 'fig6a')
    print('  fig6b: reduced-coordinate oscillation intervals...')
    fig, ax = mfig((7.4, 4.3))
    interval_values = []
    for info in analyses:
        zi = info['z_interval']
        interval_values.extend([zi['left'], zi['right'], info['z_eff_target'], 0.0])
        ezi = info.get('empirical_z_interval', {})
        if ezi.get('valid', False):
            interval_values.extend([ezi['left'], ezi['right']])
    interval_values = np.array(interval_values, dtype=float)
    interval_values = interval_values[np.isfinite(interval_values)]
    if interval_values.size:
        pad = 0.12 * max(float(np.ptp(interval_values)), 1.0)
        xmin = float(np.min(interval_values) - pad)
        xmax = float(np.max(interval_values) + pad)
    else:
        xmin, xmax = (-1.0, 1.0)
    y = np.arange(len(analyses))
    ax.axvline(0, color='#5F5F5F', ls=':', lw=0.9, zorder=1)
    for yi, info in zip(y, analyses):
        key = info['key']
        color = COLOR_REAL_BRAIN.get(key, '#888888')
        zi = info['z_interval']
        zl = float(zi['left'])
        zr = float(zi['right'])
        ezi = info.get('empirical_z_interval', {})
        zt = float(info['z_eff_target'])
        ax.hlines(yi, xmin, xmax, color='#D9D9D9', lw=8.0, alpha=0.42, zorder=1)
        ax.hlines(yi, zl, zr, color=color, lw=8.0, alpha=0.64, zorder=2)
        if ezi.get('valid', False):
            el = float(ezi['left'])
            er = float(ezi['right'])
            y_emp = yi + 0.15
            ax.hlines(y_emp, el, er, color='#303030', lw=2.0, alpha=0.88, zorder=4)
            ax.scatter([el, er], [y_emp, y_emp], marker='|', s=105, color='#303030', linewidths=1.0, zorder=5)
            ax.text((el + er) / 2.0, y_emp + 0.12, 'empirical', ha='center', va='bottom', fontsize=5.6, color='#303030')
        ax.scatter([0], [yi], marker='|', s=120, color='#303030', linewidths=1.2, zorder=4)
        ax.scatter([zt], [yi], marker='D', s=44, facecolors=color, edgecolors='#202020', linewidths=0.55, zorder=5)
        ax.text(zl, yi + 0.22, f'{zl:.2f}', ha='center', va='bottom', fontsize=6.0, color=color)
        ax.text(zr, yi + 0.22, f'{zr:.2f}', ha='center', va='bottom', fontsize=6.0, color=color)
        ax.text(zt, yi - 0.28, 'target', ha='center', va='top', fontsize=5.8, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([a['label'] for a in analyses])
    ax.invert_yaxis()
    ax.set_xlabel('Reduced coordinate $z_{\\mathrm{eff}}$')
    ax.set_title('Candidate oscillatory envelopes on directed empirical connectomes', fontsize=8.4)
    ax.set_xlim(xmin, xmax)
    interval_handle = Line2D([0], [0], color='#6E8DB8', lw=5.0, label='Candidate envelope $[z_l,z_r]$')
    empirical_handle = Line2D([0], [0], color='#303030', lw=2.0, marker='|', markersize=7, label='Empirical interval ($P_{\\mathrm{osc}}\\geq0.5$)')
    baseline_handle = Line2D([0], [0], marker='|', color='#303030', linestyle='None', markersize=9, label='Baseline $z_{\\mathrm{eff}}=0$')
    target_handle = Line2D([0], [0], marker='D', color='#303030', markerfacecolor='#FFFFFF', linestyle='None', markersize=5.2, label='Non-oscillatory target $z_{\\mathrm{eff}}^*$')
    legend_clean(ax, handles=[interval_handle, empirical_handle, baseline_handle, target_handle], loc='lower right', fontsize=6.4)
    save_sub(fig, sd, 'fig6b')
    print('  fig6c: empirical top-out dose landscapes...')
    fig, axes = plt.subplots(1, len(analyses), figsize=(3.35 * len(analyses), 3.35), sharey=True)
    if len(analyses) == 1:
        axes = [axes]
    im = None
    for ax, info in zip(axes, analyses):
        dose_rows = sorted(info['topk_dose_landscape'], key=lambda r: r['dose'])
        doses = np.array([r['dose'] for r in dose_rows], dtype=float)
        kf = np.array([r['k_frac'] for r in dose_rows[0]['summary']], dtype=float)
        P = np.array([[rr['success_probability'] for rr in row['summary']] for row in dose_rows], dtype=float)
        if P.shape[0] >= 2 and P.shape[1] >= 2:
            im = ax.contourf(kf, doses, P, levels=SUCCESS_LEVELS, vmin=0, vmax=1, cmap=CMAP_SUCCESS)
        else:
            ypad = max(0.05, 0.08 * max(float(np.nanmax(doses)), 1.0))
            im = ax.imshow(P, origin='lower', aspect='auto', cmap=CMAP_SUCCESS, vmin=0, vmax=1, extent=[float(np.nanmin(kf)), float(np.nanmax(kf)), float(np.nanmin(doses) - ypad), float(np.nanmax(doses) + ypad)])
        if P.shape[0] >= 2 and P.shape[1] >= 2 and (np.nanmin(P) <= CONTROL_PROB_THRESHOLD <= np.nanmax(P)):
            ax.contour(kf, doses, P, levels=[CONTROL_PROB_THRESHOLD], colors='#272727', linewidths=1.0)
        ax.axhline(MAX_DRIVER_Z, color='#272727', ls=':', lw=0.8)
        if np.isfinite(info['topk_k_eff_fraction']):
            ax.axvline(info['topk_k_eff_fraction'], color=COLOR_REAL_BRAIN.get(info['key'], '#555555'), ls='-.', lw=0.95)
        ax.set_title(info['label'], fontsize=8, color=COLOR_REAL_BRAIN.get(info['key'], '#555555'))
        ax.set_xlabel('Top out-strength fraction $k/N$')
        ax.set_xlim(0, 1)
    axes[0].set_ylabel('Driver dose $|\\Delta z_D|$')
    fig.subplots_adjust(wspace=0.16, right=0.9, left=0.08, bottom=0.18, top=0.88)
    cax = fig.add_axes([0.92, 0.26, 0.018, 0.52])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('$P_{\\mathrm{success}}$', fontsize=8)
    fig._skip_tight_layout = True
    save_sub(fig, sd, 'fig6c')
    print('  fig6d: empirical W_D concentration...')
    fig, ax = mfig((6.6, 4.1))
    for info in analyses:
        rows = info['topk_summary']
        xs_ = [r['k_frac'] for r in rows]
        ys_ = [r['W_ctrl'] for r in rows]
        color = COLOR_REAL_BRAIN.get(info['key'], '#888888')
        ax.plot(xs_, ys_, color=color, lw=1.5, marker='o', ms=2.7, label=info['label'])
        crit = find_k_eff_success(rows, threshold=CONTROL_PROB_THRESHOLD)
        if crit:
            ax.axvline(crit['k_frac'], color=color, ls=':', lw=0.85, alpha=0.58)
            ax.scatter([crit['k_frac']], [crit['W_ctrl']], s=34, facecolors=color, edgecolors='#202020', linewidths=0.45, zorder=5)
    ax.set_xlabel('Top out-strength fraction $k/N$')
    ax.set_ylabel('Cumulative output weight $W_D$')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    legend_clean(ax, loc='lower right', fontsize=6.3)
    save_sub(fig, sd, 'fig6d')
    print('  fig6e-f: Allen mouse brain-region and epilepsy-target control...')
    allen_runtime = runtimes.get('allen_mouse')
    if not allen_runtime or not allen_runtime.get('region_groups'):
        raise ValueError('Fig6e/f require data/allen_mouse_connectome.npz to provide `region_groups` or `node_regions`; no fallback dataset is used.')
    A = allen_runtime['A']
    w = allen_runtime['w']
    out_order = allen_runtime['out_order']
    region_groups = allen_runtime['region_groups']
    node_labels = allen_runtime.get('node_labels') or []
    z_eff_target = float(allen_runtime['z_eff_target'])
    N = A.shape[0]
    region_results = _compute_region_reachability(A, w, region_groups, z_eff_target)
    bar_order = _fig6_ordered_regions(region_groups, region_results)
    if not bar_order:
        raise ValueError('Fig6e/f found no nonempty Allen mouse brain regions in `region_groups` or `node_regions`.')
    epilepsy_targets = _fig6_allen_epilepsy_targets(region_groups, node_labels)
    epilepsy_combo_targets = _fig6_allen_epilepsy_combo_targets(region_groups, epilepsy_targets)
    all_epilepsy_targets = dict(epilepsy_targets)
    all_epilepsy_targets.update(epilepsy_combo_targets)
    epilepsy_target_results = {}
    for target_key, meta in all_epilepsy_targets.items():
        ctrl = sorted(set((int(i) for i in meta['nodes'] if 0 <= int(i) < N)))
        info = solve_baseline_driver_z(w, ctrl, z_eff_target)
        epilepsy_target_results[target_key] = {'label': meta['label'], 'literature_link': meta['literature_link'], 'target_type': 'network_combo' if target_key in epilepsy_combo_targets else 'single_region', 'summary_only': bool(meta.get('summary_only', False)), 'n_nodes': int(len(ctrl)), 'n_frac': float(len(ctrl) / max(N, 1)), 'W_D': float(info['W_D']), 'z_driver': float(info['z_driver']) if info['valid'] else np.nan, 'z_driver_unbounded': float(info['z_driver_unbounded']) if info['valid'] else np.nan, 'z_eff_realized': float(info['z_eff_realized']), 'target_feasible': bool(info.get('target_feasible', False)), 'reduced_coordinate_reachable': bool(info.get('target_feasible', False)), 'nodes': ctrl}
    w_req = abs(z_eff_target - OSCILLATORY_Z) / max(MAX_DRIVER_Z, 1e-12)
    fig_height = max(4.8, min(10.5, 0.33 * len(bar_order) + 1.6))
    fig, ax = mfig((7.6, fig_height))
    y = np.arange(len(bar_order))
    vals = [region_results[r]['W_D'] for r in bar_order]
    xmax = min(1.05, max(0.25, max(vals) * 1.25 if vals else 0.25, w_req * 1.6))
    ax.axvspan(w_req, xmax, color='#CAEBE7', alpha=0.18, zorder=0)
    for yi, rname in zip(y, bar_order):
        res = region_results[rname]
        v = float(res['W_D'])
        ok = bool(res['reduced_coordinate_reachable'])
        nf = float(res['n_frac'])
        col = COLOR_REGION.get(rname, COLOR_REAL_BRAIN.get('allen_mouse', '#999999'))
        ax.hlines(yi, 0, v, color=col, lw=2.0, alpha=0.58, zorder=1)
        size = 45 + 190 * min(nf, 0.35) / 0.35
        ax.scatter(v, yi, s=size, facecolors=col if ok else 'white', edgecolors='#272727' if ok else col, lw=0.55, alpha=0.94, zorder=3)
        if rname in {'thalamus', 'hippocampal', 'hypothalamus', 'cerebellum'}:
            ax.scatter(min(v + 0.045, xmax * 0.985), yi, s=50, marker='*', facecolors='#272727', edgecolors='white', linewidths=0.35, zorder=4)
        ax.text(v + 0.012, yi, f'{v:.2f}', va='center', fontsize=6.0)
    ax.set_yticks(y)
    ax.set_yticklabels([_fig6_region_label(r) for r in bar_order], fontsize=6.2)
    ax.invert_yaxis()
    ax.axvline(w_req, color='#767676', ls='--', lw=0.9, label=f'Reachability bound $|z^*|/|\\Delta z_D|_{{ref}}={w_req:.2f}$')
    ax.text(w_req + 0.012, -0.55, 'reachable at reference dose', color='#767676', fontsize=6.0, ha='left', va='center')
    ax.set_xlabel('Cumulative output weight $W_D$ (reduced-coordinate reachability)')
    ax.set_xlim(0, xmax)
    clinical_handle = Line2D([0], [0], marker='*', color='w', markerfacecolor='#272727', markeredgecolor='white', markersize=7, label='epilepsy-linked region class')
    bound_handle = Line2D([0], [0], color='#767676', ls='--', lw=0.9, label=f'$|z^*|/|\\Delta z_D|_{{ref}}={w_req:.2f}$')
    legend_clean(ax, handles=[bound_handle, clinical_handle], loc='lower right', fontsize=6.2)
    save_sub(fig, sd, 'fig6e')
    region_time_series = []
    k_eff_struct, ctrl_struct, w_req_struct = _fig6_structural_top_drivers(w, out_order, z_eff_target)
    allen_analysis = next((a for a in analyses if a['key'] == 'allen_mouse'), None)
    allen_topk_crit = find_k_eff_success(allen_analysis['topk_summary'], CONTROL_PROB_THRESHOLD) if allen_analysis is not None else None
    k_eff_empirical = int(allen_topk_crit['k']) if allen_topk_crit else int(k_eff_struct)
    ctrl_top = sorted(out_order[:k_eff_empirical].astype(int).tolist()) if k_eff_empirical > 0 else []
    x0, y0 = make_ic(N, allen_runtime['seed_base'] + 1)
    zv = make_oscillatory_z_vec(N)
    av_, bv_, cv_ = make_abc(N, w, allen_runtime['seed_base'] + 3)
    fig, ax = mfig((10.2, 4.8))
    sn_none = rk4_net_collective(A, x0.copy(), y0.copy(), zv.copy(), t_eval, w, av_, bv_, cv_)
    mx_none = collective_state_metrics(sn_none['xw'], t_eval)
    ax.plot(t_eval, sn_none['xw'], color=COLOR_NO_CONTROL, lw=1.25, label=f"No control ($A_{{tail}}={mx_none['tail_amp']:.2f}$)")
    region_time_series.append({'label': 'No control', 'region': None, 'n_nodes': 0, 'W_D': 0.0, 'metrics': mx_none})
    plot_target_keys = ['anterior_thalamic', 'hippocampal', 'lateral_hypothalamus', 'cerebellar_nuclei', 'thalamus', 'thalamo_hippocampal', 'thalamo_cortical', 'thalamo_hypothalamic', 'thalamo_cortico_hypothalamic']
    target_styles = {'anterior_thalamic': (1.0, '-', COLOR_REGION.get('anterior_thalamic', '#6FBA4F'), 0.62), 'hippocampal': (1.0, '--', COLOR_REGION.get('hippocampal', '#38C08F'), 0.62), 'lateral_hypothalamus': (0.95, '-.', COLOR_REGION.get('lateral_hypothalamus', '#DD7597'), 0.58), 'cerebellar_nuclei': (0.95, ':', COLOR_REGION.get('cerebellar_nuclei', '#9AA8B4'), 0.58), 'thalamus': (1.35, '-', COLOR_REGION.get('thalamus', '#A9D9BB'), 0.88), 'thalamo_hippocampal': (1.45, '--', COLOR_REGION.get('thalamo_hippocampal', '#0C8F7A'), 0.92), 'thalamo_cortical': (1.45, '-.', COLOR_REGION.get('thalamo_cortical', '#2C6BB3'), 0.92), 'thalamo_hypothalamic': (1.45, (0, (5, 1.8)), COLOR_REGION.get('thalamo_hypothalamic', '#C45A8A'), 0.92), 'thalamo_cortico_hypothalamic': (1.55, (0, (3, 1, 1, 1)), COLOR_REGION.get('thalamo_cortico_hypothalamic', '#413A97'), 0.95)}
    for target_key in plot_target_keys:
        if target_key not in epilepsy_target_results:
            continue
        res = epilepsy_target_results[target_key]
        ctrl = res['nodes']
        info = solve_baseline_driver_z(w, ctrl, z_eff_target)
        if not info['valid'] or not np.isfinite(info['z_driver']):
            continue
        sn = rk4_net_collective_pinned(A, x0.copy(), y0.copy(), zv.copy(), t_eval, w, ctrl, info['z_driver'], av_, bv_, cv_)
        mx = collective_state_metrics(sn['xw'], t_eval)
        lw, ls, color, alpha = target_styles.get(target_key, (1.2, '-', '#777777', 0.8))
        label = f"{res['label']} ({len(ctrl)}; $A_{{tail}}={mx['tail_amp']:.2f}$)"
        ax.plot(t_eval, sn['xw'], color=color, lw=lw, ls=ls, alpha=alpha, label=label)
        region_time_series.append({'label': res['label'], 'region': target_key, 'literature_link': res['literature_link'], 'target_type': res.get('target_type', 'single_region'), 'n_nodes': len(ctrl), 'W_D': float(info['W_D']), 'z_driver': float(info['z_driver']), 'target_feasible': bool(info.get('target_feasible', False)), 'reduced_coordinate_reachable': bool(info.get('target_feasible', False)), 'metrics': mx})
    if ctrl_top:
        z_driver_top = float(driver_z_from_dose(z_eff_target, MAX_DRIVER_Z))
        sn_top = rk4_net_collective_pinned(A, x0.copy(), y0.copy(), zv.copy(), t_eval, w, ctrl_top, z_driver_top, av_, bv_, cv_)
        mx_top = collective_state_metrics(sn_top['xw'], t_eval)
        ax.plot(t_eval, sn_top['xw'], color='#2A2A2A', lw=1.25, label=f"Top-$W_D$ empirical set ({k_eff_empirical}; $A_{{tail}}={mx_top['tail_amp']:.2f}$)")
        region_time_series.append({'label': 'Top-W_D empirical set', 'region': 'top_out_strength', 'n_nodes': int(k_eff_empirical), 'W_D': float(driver_weight(w, ctrl_top)), 'z_driver': z_driver_top, 'target_feasible': bool(driver_weight(w, ctrl_top) >= w_req_struct), 'reduced_coordinate_reachable': bool(driver_weight(w, ctrl_top) >= w_req_struct), 'metrics': mx_top})
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$\\bar{x}(t)$')
    ax.set_title('Allen mouse epilepsy-linked target suppression dynamics', fontsize=8)
    ax.set_xlim(t_eval[0], t_eval[-1])
    legend_clean(ax, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=5.2, handlelength=1.65, labelspacing=0.28, borderpad=0.25)
    fig.subplots_adjust(right=0.7, left=0.09, bottom=0.16, top=0.9)
    fig._skip_tight_layout = True
    save_sub(fig, sd, 'fig6f')
    print('  fig6f: Allen epilepsy-target probability check...')
    epilepsy_group_sets = {meta['label']: [epilepsy_target_results[key]['nodes']] for key, meta in all_epilepsy_targets.items() if key in epilepsy_target_results and epilepsy_target_results[key]['nodes']}
    epilepsy_probability_summary = _run_grouped_control_sets(A, w, epilepsy_group_sets, float(driver_z_from_dose(z_eff_target, MAX_DRIVER_Z)), int(allen_runtime['seed_base'] + 70000), n_trials=FIG6_TOPK_TRIALS) if epilepsy_group_sets else []
    save_json({'threshold': CONTROL_PROB_THRESHOLD, 'strict_threshold': CONTROL_PROB_STRICT_THRESHOLD, 'reference_driver_dose': REFERENCE_DRIVER_DOSE, 'control_input_policy': CONTROL_INPUT_POLICY, 'empirical_networks': analyses, 'horizontal_transfer_networks': [a['key'] for a in analyses], 'reduced_interval_application': [{'key': a['key'], 'label': a['label'], 'A_eff': a['A_eff'], 'z_interval': a['z_interval'], 'empirical_z_interval': a.get('empirical_z_interval'), 'empirical_zscan_n_trials': a.get('empirical_zscan', {}).get('n_trials'), 'empirical_zscan_n_z': a.get('empirical_zscan', {}).get('n_z'), 'zero_in_interval': a['zero_in_interval'], 'z_eff_target': a['z_eff_target'], 'W_D_required_at_reference_dose': a['W_D_required_at_reference_dose']} for a in analyses], 'empirical_threshold_sensitivity': {a['key']: success_threshold_table(a['topk_summary']) for a in analyses}, 'vertical_application_network': 'allen_mouse', 'allen_region_reachability_bound': float(w_req), 'allen_region_results': region_results, 'allen_epilepsy_target_results': epilepsy_target_results, 'allen_epilepsy_target_probability': epilepsy_probability_summary, 'allen_structural_reachability': {'k_eff_structural_bound': int(k_eff_struct), 'k_eff_structural_bound_fraction': float(k_eff_struct / max(N, 1)), 'k_eff_empirical_success': int(k_eff_empirical), 'k_eff_empirical_success_fraction': float(k_eff_empirical / max(N, 1)), 'W_D_required_at_reference_dose': float(w_req_struct), 'W_D_structural_bound': float(driver_weight(w, ctrl_struct)), 'W_D_empirical_success': float(driver_weight(w, ctrl_top))}, 'allen_region_time_series': region_time_series}, os.path.join(sd, 'fig6_summary.json'))
    print('[Fig 6] Done.\n')

def _convergence_state_summary(series, times):
    state = collective_state_metrics(series, times)
    period = est_period(series, times)
    return {'state': state['state'], 'valid': bool(state['valid']), 'is_osc': bool(state['is_osc']), 'tail_amplitude': float(state['tail_amp']), 'tail_std': float(state['tail_std']), 'n_peaks': int(state['n_peaks']), 'period_valid': bool(period['valid']), 'period': float(period['T_mean']) if period['valid'] else np.nan}

def _relative_change(coarse, fine, floor=1e-12):
    if not np.isfinite(coarse) or not np.isfinite(fine):
        return np.nan
    return float(abs(coarse - fine) / max(abs(fine), float(floor)))

def _time_step_comparison(coarse_x, coarse_y, fine_x, fine_y, coarse_t, fine_t):
    fine_x_on_coarse = np.asarray(fine_x, dtype=float)[::2]
    fine_y_on_coarse = np.asarray(fine_y, dtype=float)[::2]
    coarse_x = np.asarray(coarse_x, dtype=float)
    coarse_y = np.asarray(coarse_y, dtype=float)
    if len(fine_x_on_coarse) != len(coarse_x):
        raise RuntimeError('The half-step convergence grid does not align with the main grid')
    n0 = int(discard_ratio * len(coarse_x))
    numerator = float(np.sqrt(np.mean((coarse_x[n0:] - fine_x_on_coarse[n0:]) ** 2 + (coarse_y[n0:] - fine_y_on_coarse[n0:]) ** 2)))
    denominator = max(float(np.sqrt(np.mean(fine_x_on_coarse[n0:] ** 2 + fine_y_on_coarse[n0:] ** 2))), 1e-12)
    coarse_summary = _convergence_state_summary(coarse_x, coarse_t)
    fine_summary = _convergence_state_summary(fine_x, fine_t)
    return {'coarse': coarse_summary, 'half_step': fine_summary, 'classification_agreement': bool(coarse_summary['valid'] == fine_summary['valid'] and coarse_summary['is_osc'] == fine_summary['is_osc']), 'tail_amplitude_relative_change': _relative_change(coarse_summary['tail_amplitude'], fine_summary['tail_amplitude']), 'period_relative_change': _relative_change(coarse_summary['period'], fine_summary['period']), 'post_transient_state_nrmse': float(numerator / denominator)}

def run_numerical_convergence_checks(A_raw):
    print('\n[Validation] RK4 and slow-branch quadrature convergence...')
    validation_dir = os.path.join(output_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    A, net_info = benchmark_operating_point_info(A_raw, 'convergence_er')
    w = degree_weights(A)
    Ae = effective_coupling(A)
    theory = derive_z_interval(Ae)
    stable_margin = stable_target_margin_info(theory)
    z_nonoscillatory = float(FIG2_STABLE_Z)
    if z_nonoscillatory < theory['z_right'] + stable_margin['absolute_buffer']:
        raise ValueError('FIG2_STABLE_Z is not outside the buffered oscillatory interval.')
    fine_t = np.linspace(tStart, tEnd, 2 * (len(t_eval) - 1) + 1)
    regimes = [('oscillatory', OSCILLATORY_Z, FIG2_SEED_SHIFT, make_weighted_mean_z_vec(OSCILLATORY_Z, w, ER_SEED + 40000)), ('nonoscillatory', z_nonoscillatory, FIG2_SEED_SHIFT + 100, make_weighted_mean_z_vec(z_nonoscillatory, w, ER_SEED + 50000))]
    step_rows = []
    for label, z_value, seed_shift, z_vector in regimes:
        x0, y0 = make_ic(A.shape[0], ER_SEED + 10000 + seed_shift)
        av, bv, cv = make_abc(A.shape[0], w, ER_SEED + 30000 + seed_shift)
        z_eff = z_eff_value(z_vector, w)
        coarse_net = rk4_net_collective(A, x0.copy(), y0.copy(), z_vector.copy(), t_eval, w, av, bv, cv)
        fine_net = rk4_net_collective(A, x0.copy(), y0.copy(), z_vector.copy(), fine_t, w, av, bv, cv)
        x_eff0 = float(np.dot(w, x0))
        y_eff0 = float(np.dot(w, y0))
        coarse_red = rk4_red(x_eff0, y_eff0, z_eff, Ae, t_eval)
        fine_red = rk4_red(x_eff0, y_eff0, z_eff, Ae, fine_t)
        step_rows.append({'regime': label, 'z_eff': float(z_eff), 'seed_shift': int(seed_shift), 'full_network': _time_step_comparison(coarse_net['xw'], coarse_net['yw'], fine_net['xw'], fine_net['yw'], t_eval, fine_t), 'reduced_model': _time_step_comparison(coarse_red['x'], coarse_red['y'], fine_red['x'], fine_red['y'], t_eval, fine_t)})
    zl, zr = (theory['z_left'], theory['z_right'])
    margin = 0.05 * (zr - zl)
    zlo = max(zl + margin, OSCILLATORY_Z - FIG2_DIAG_Z_HALF_WIDTH)
    zhi = min(zr - margin, OSCILLATORY_Z + FIG2_DIAG_Z_HALF_WIDTH)
    if zlo >= zhi:
        zlo, zhi = (zl + margin, zr - margin)
    quadrature_rows = []
    for z_value in np.linspace(zlo, zhi, N_Z_PERIOD):
        base = relaxation_period(Ae, float(z_value), nq=20000)
        refined = relaxation_period(Ae, float(z_value), nq=40000)
        quadrature_rows.append({'z_eff': float(z_value), 'base_valid': bool(base['valid']), 'refined_valid': bool(refined['valid']), 'period_20000': float(base['T']) if base['valid'] else np.nan, 'period_40000': float(refined['T']) if refined['valid'] else np.nan, 'relative_change': _relative_change(base['T'], refined['T'])})
    timestep_values = [comparison[key] for row in step_rows for comparison in (row['full_network'], row['reduced_model']) for key in ('tail_amplitude_relative_change', 'period_relative_change', 'post_transient_state_nrmse') if np.isfinite(comparison[key])]
    quadrature_values = [row['relative_change'] for row in quadrature_rows if np.isfinite(row['relative_change'])]
    classifications_agree = all((comparison['classification_agreement'] for row in step_rows for comparison in (row['full_network'], row['reduced_model'])))
    report = {'benchmark': 'shared edge-budget-matched directed ER network', 'network_operating_point': net_info, 'A_eff': float(Ae), 'main_dt': float(t_eval[1] - t_eval[0]), 'half_dt': float(fine_t[1] - fine_t[0]), 'step_halving': step_rows, 'quadrature_refinement': quadrature_rows, 'classification_agreement_all': bool(classifications_agree), 'maximum_finite_step_metric_change': float(max(timestep_values)) if timestep_values else np.nan, 'maximum_period_quadrature_relative_change': float(max(quadrature_values)) if quadrature_values else np.nan}
    save_json(report, os.path.join(validation_dir, 'numerical_convergence.json'))
    print(f"  classifications agree={report['classification_agreement_all']}, max quadrature relative change={report['maximum_period_quadrature_relative_change']:.3g}")
    return report

def main():
    reset_main_output_dirs()
    save_json({'N_CONTROL_PROB_TRIALS': N_CONTROL_PROB_TRIALS, 'BRAIN_CONTROL_PROB_TRIALS': BRAIN_CONTROL_PROB_TRIALS, 'FIG6_TOPK_TRIALS': FIG6_TOPK_TRIALS, 'FIG6_LANDSCAPE_TRIALS': FIG6_LANDSCAPE_TRIALS, 'FIG6_INTERVAL_SCAN_POINTS': FIG6_INTERVAL_SCAN_POINTS, 'FIG6_INTERVAL_SCAN_TRIALS': FIG6_INTERVAL_SCAN_TRIALS, 'CONTROL_PROB_THRESHOLD': CONTROL_PROB_THRESHOLD, 'CONTROL_PROB_STRICT_THRESHOLD': CONTROL_PROB_STRICT_THRESHOLD, 'REFERENCE_DRIVER_DOSE': REFERENCE_DRIVER_DOSE, 'CONTROL_INPUT_POLICY': CONTROL_INPUT_POLICY, 'ER_N': ER_N, 'ER_M': ER_M, 'ER_P': ER_P, 'ER_SEED': ER_SEED, 'FIG4_N': FIG4_N, 'FIG4_ER_M': FIG4_ER_M, 'FIG4_ER_P': FIG4_ER_P, 'FIG4_BA_M': FIG4_BA_M, 'FIG4_SW_K': FIG4_SW_K, 'FIG4_SW_P': FIG4_SW_P, 'FHN_a': a, 'FHN_b': b, 'FHN_c': c, 'A_OVER_B': a / b, 'CONTROL_COORDINATE': 'direct additive FHN input z', 'OSCILLATORY_Z': OSCILLATORY_Z, 'FIG2_STABLE_Z': FIG2_STABLE_Z, 'Z_STD': Z_STD, 'FIG2_Z_DISTRIBUTION': 'weighted-recentered Gaussian', 'ABC_REL_HALF_RANGE': ABC_REL_HALF_RANGE, 'STABLE_TARGET_MARGIN': STABLE_TARGET_MARGIN, 'FINAL_WINDOW_RATIO': FINAL_WINDOW_RATIO, 'INIT_DISTRIBUTION': f'Gaussian N(0,{INIT_VAR:g})', 'INIT_VARIANCE': INIT_VAR, 'SYNTH_EDGE_LOW': SYNTH_EDGE_LOW, 'SYNTH_EDGE_HIGH': SYNTH_EDGE_HIGH, 'BENCHMARK_WEIGHT_MODE': BENCHMARK_WEIGHT_MODE, 'BENCHMARK_NOMINAL_MEAN_DEGREE': BENCHMARK_NOMINAL_MEAN_DEGREE, 'SYNTH_EDGE_RULE': 'direct_uniform_nonzero_edges', 'EMPIRICAL_WEIGHT_NORMALIZATION': EMPIRICAL_WEIGHT_NORMALIZATION, 'FIG4_DOSE_GRID': list(FIG4_DOSE_GRID), 'FIG4_LANDSCAPE_DOSE_GRID': list(FIG4_LANDSCAPE_DOSE_GRID), 'FIG6_LANDSCAPE_DOSE_GRID': list(FIG6_LANDSCAPE_DOSE_GRID), 'FIG6_TOPK_FRACS': list(FIG6_TOPK_FRACS), 'FIG_EXTS': list(FIG_EXTS), 'FIG4_K_STEP': FIG4_K_STEP, 'FIG5_K_STEP': FIG5_K_STEP, 'FIG5_N_RANDOM_REPEATS': FIG5_N_RANDOM_REPEATS, 'FIG5_FIXED_K_FRAC': FIG5_FIXED_K_FRAC, 'FIG5_GROUP_K_FRACS': list(FIG5_GROUP_K_FRACS), 'FIG5_DOSE_K_FRACS': list(FIG5_DOSE_K_FRACS), 'FIG5_DOSE_GRID': list(FIG5_DOSE_GRID), 'FIG5_REFERENCE_DOSE': FIG5_REFERENCE_DOSE, 'FIG5_BA_N': FIG5_BA_N, 'FIG5_BA_M': FIG5_BA_M, 'FIG5_BA_SEED': FIG5_BA_SEED, 'USE_RAW_EMPIRICAL_WEIGHTS': USE_RAW_EMPIRICAL_WEIGHTS, 'N_SCAN': N_SCAN, 'N_ZSCAN_TRIALS': N_ZSCAN_TRIALS, 'N_SENS': N_SENS, 'N_SENS_Z': N_SENS_Z, 'N_SENS_TRIALS': N_SENS_TRIALS, 'SENS_SMOOTH_SIGMA': SENS_SMOOTH_SIGMA, 'SENS_Z_ABS_MAX': SENS_Z_ABS_MAX, 'N_Z_PERIOD': N_Z_PERIOD, 'FIG2_SEED_SHIFT': FIG2_SEED_SHIFT, 'RUN_CONVERGENCE_CHECKS': RUN_CONVERGENCE_CHECKS, 'N_TIME': N_TIME, 'tEnd': tEnd}, os.path.join(output_dir, 'run_parameters.json'))
    print('=' * 60)
    print('Nature Brain FHN - Dimensionality Reduction + Collective Control')
    print('=' * 60)
    print('\n[Network] Generating shared ER benchmark graph...')
    A_er = make_shared_fig234_network()
    print(f'  N={ER_N}, M={ER_M}, p={ER_P:.5f}, seed={ER_SEED}, edges={int(np.count_nonzero(A_er))}')
    print('[Network] Generating shared BA benchmark graph for Fig. 4-5...')
    A_ba = make_fig4_network('BA')
    print(f"  N={FIG4_N}, m={FIG4_BA_M}, seed={FIG4_NET_SEEDS['BA']}, edges={int(np.count_nonzero(A_ba))}")
    if RUN_FIG2:
        run_fig2(A_er)
    if RUN_FIG3:
        run_fig3(A_er)
    if RUN_FIG4:
        run_fig4(A_er, A_ba)
    if RUN_FIG5:
        run_fig5(A_ba)
    if RUN_FIG6:
        run_fig6()
    if RUN_CONVERGENCE_CHECKS:
        run_numerical_convergence_checks(A_er)
    print('\nAll selected figures finished.')
if __name__ == '__main__':
    main()
