import awkward    as ak
import numpy      as np
import tensorflow as tf
import vector

metre = 1e3

def to_4momentum(cells, Et_key = 'cell_et'):
    position = to_3vector(cells)
    vectors  = vector.zip(
        {
            "m"  : ak.zeros_like(cells[Et_key])
            "pt" : cells[Et_key],
            "eta": position.eta,
            "phi": pisition.phi,
        }
    )
    return vectors

def to_3vector(cells):
    vectors = vector.zip(
        {
            "x": cells.cell_x / metre,
            "y": cells.cell_y / metre,
            "z": cells.cell_z / metre,
        }
    )
    return vectors
