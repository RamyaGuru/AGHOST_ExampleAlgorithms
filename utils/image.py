import numpy      as np
import tensorflow as tf
import awkward    as ak
import skimage
import vector

from utils.cells import get_layer, to_4momentum, remove_transition

def cells_to_towers(cells, Et_key='cell_et'):
    cells        = remove_transition(cells)
    cell_layer   = get_layer(cells.cell_sampling)
    cell_vectors = to_4momentum(cells, Et_key=Et_key)

    cell_eta     = cell_vectors.eta
    cell_eta     = (
        cell_eta
        + ak.where((cell_eta > 1.5 ) & (cell_layer == 2), -0.01, 0)
        + ak.where((cell_eta < -1.5) & (cell_layer == 2), +0.01, 0)
        + ak.where((cell_eta > 0.1 ) & (cell_eta < 1.4) & (cell_layer == 1), -0.005, 0)
    )

    cell_vectors = vector.zip(
        {
            "pt" : cell_vectors.pt,
            "m"  : cell_vectors.m,
            "eta": cell_eta,
            "phi": cell_vectors.phi,
        }
    )

    towers = np.concatenate(
        [vector_to_tower(cell_vectors[cell_layer == layer]) for layer in range(6)],
        axis = -1,
    )

    return towers

def sliding_window(x, size):
    assert x.ndim == 4
    px      = pad(x, (size - 1) // 2)
    windows = skimage.util.view_as_windows(
        px, window_shape = (1, size, size, px.shape[-1]), step = 1
    )

    return windows

def get_tower_eta(X):
    eta_idxs = np.indices(X[..., :1].shape)[1]

    return (eta_idxs - np.median(eta_idxs)) * 0.1

def vector_to_tower(
        vectors,
        eta_edges = np.linspace(-2.5  , 2.5  , 51),
        phi_edges = np.linspace(-np.pi, np.pi, 65),
):
    tower_edges   = (np.arange(1 + len(vectors)), eta_edges, phi_edges)
    event_indices = ak.flatten(get_index(vectors))
    flat_vectors  = ak.flatten(vectors)

    towers = np.histogramdd(
        (
            ak.to_numpy(event_indices),
            ak.to_numpy(flat_vectors.eta),
            ak.to_numpy(flat_vectors.phi),
        ),
        bins    = tower_edges,
        weights = ak.to_numpy(flat_vectors.pt),
    )[0]

    return np.expand_dims(towers, axis=-1)

def tower_to_vector(X):
    _, eta_idxs, phi_idxs, _ = np.indices(X.shape)

    eta = (eta_idxs - np.median(eta_idxs)) * 0.1
    phi = (phi_idxs - np.median(phi_idxs)) * np.pi / 32

    vectors = vector.arr(
        {
            'eta': eta,
            'phi': phi,
            'pt' : X,
            'm'  : np.zeros_like(X),
        }
    )

    return vectors

def pad(x, pad_size):
    assert x.ndim == 4
    y = np.pad(
        x,
        ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)),
        mode            = 'constant',
        constant_values = 0,
    )
    y = np.pad(y, ((0,0), (0,0), (pad_size, pad_size), (0,0)), mode='wrap')
    return y

def unpad(x, pad_size):
    assert x.ndim == 4, "Expected shape (N, H, W, C)"
    return x[:, pad_size:-pad_size or None, pad_size:-pad-size or None, :]

def augment_image(X):
    aug_X = tf.concat(
        [
            X,
            tf.reverse(X, axis=[1]),
            tf.reverse(X, axis=[2]),
            tf.reverse(X, axis=[1,2]),
        ],
        axis = 0,
    )

    return aug_X

def get_index(vectors):
    return ak.ones_like(vectors.eta, dtype=int) * np.arange(len(vectors))

def get_centers(edges):
    return (edges[:-1] + edges[1:]) / 2

def get_weight_variations(weights, N = 25):
    np.random.seed(42)
    poiss           = np.random.poisson(1.0, size = (len(weights), N))
    poisson_weights = np.asarray(weights).reshape(-1, 1) * poiss
    return poisson_weights

def get_hist_variations(x, bins=20, weights=None, density=False, N=25):
    if weights is None:
        weights = np.ones_like(x)

    poisson_weights = get_weight_variations(weights, N)

    bin_edges = np.histogram_bin_edges(x, bins, weights=weights)
    _, widx   = np.indices(poisson_weights.shape)

    x = np.broadcast_to(x.reshape(-1, 1), poisson_weights.shape)

    variations = np.histogram2d(
        x.flatten(),
        widx.flatten(),
        bins    = [bin_edges, N],
        weights = poisson_weights.flatten(),
    )[0]

    if density:
        norm = variations.mean(axis=1).sum() * np.diff(bin_edges)[:, None]
    else:
        norm = 1.0

    return variations / norm, bin_edges

def _clopper_pearson(k, n, alpha=0.03173):
    from scipy.stats import beta

    lower = beta.ppf(alpha / 2, k, n - k + 1)
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k)

    return lower, upper

def plot_turn_on(x, mask, bins = 20, ax = None, conf_level = 0.6828, **kwargs):
    if ax is None:
        from matplotlib import pyplot as plot
        _, ax = plt.subplots()

    x = np.asarray(x)
    m = np.asarray(mask, dtype=bool)

    if np.isscalar(bins):
        bin_edges = np.histogram_bin_edges(x, bins)
    else:
        bin_edges = np.asarray(bins)

    n, _ = np.histogram(x   , bins = bin_edges)
    k, _ = np.histogram(x[m], bins = bin_edges)

    eff = np.zeros_like(n, dtype=float)
    ok  = n > 0 

    eff[ok] = k[ok] / n[ok]

    alpha  = 1.0 - float(conf_level)
    lo, hi = _clopper_pearson(k, n, alpha)

    yerr = np.vstack([eff-lo, hi-eff])

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    ax.errorbar(
        centers,
        eff,
        yerr      = yerr,
        fmt       = 'o',
        capsize   = 3,
        linestyle = 'none',
        **kwargs,
    )

    ax.set_title(kwargs.get('label', ''))
    ax.set_xlabel('Truth Jet $p_T$ [GeV]')
    ax.set_ylabel('Efficiency')
    ax.grid()

    return centers, eff, lo, hi

