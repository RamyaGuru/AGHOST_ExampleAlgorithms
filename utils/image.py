import numpy      as np
import tensorflow as tf
import awkward    as ak
import skimage
import vector

from utils.cells import get_layer, to_4momentum, remove_transition

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
