from modules.cnn import QuantisedCNN
from utils.image import cells_to_towers, pad, sliding_window, get_tower_eta
from utils.misc  import get_config, get_data

from tensorflow.keras import (
    losses,
    Input,
    Model,
    optimizers,
    callbacks,
    backend,
    layers,
)

from sklearn.model_selection import train_test_split

import keras_tuner as kt
import numpy       as np

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--sample" , type=str, default="ggF_SM_HH4b_delphes")
parser.add_argument("--max-trials"   , type=int, default=1)
parser.add_argument("--window-size"  , type=int, default=3)
parser.add_argument("--batch-size"   , type=int, default=128)
parser.add_argument("--epochs"       , type=int, default=100)
parser.add_argument("--lr-decay"     , action="store_true")
parser.add_argument("--min-Et"       , type=int, default=2)
parser.add_argument("--overwrite"    , action="store_true")
parser.add_argument("--abseta"       , action="store_true")
parser.add_argument("--loss-by-layer", action="store_true")
parser.add_argument("--add-dense"    , type=int, default=0)
args = parser.parse_args()


def build_regressor(hp):
    input_shape = (None, None, X.shape[-1])
    inputs      = Input(input_shape)

    if use_abseta:
        aux_input_shape = (None, None, 1)
        aux_inputs = Input(aux_input_shape)

    else:
        aux_input_shape = None

    weights = QuantisedCNN(
        size               = size,
        depth_multiplier   = 4,  # hp.Choice("depth_mult", [1, 2, 3, 4]),
        conv_bits          = 12,  # hp.Int("conv_bits", min_value=2, max_value=12, step=2),
        dense_bits         = 12,  # hp.Int("dense_bits", min_value=2, max_value=12, step=2),
        hidden_layer_sizes = [add_dense] if add_dense > 0 else [],
        input_shape        = input_shape,
        aux_input_shape    = aux_input_shape,
        output_layers      = [(6, layers.Activation("hard_sigmoid"))],
        symmetric          = True,
    )([inputs, aux_inputs] if use_abseta else inputs)

    if size == 1:
        Et = inputs
    elif size % 2 == 1:
        pad_size = size // 2
        Et       = inputs[:, pad_size:-pad_size, pad_size:-pad_size]
    else:
        raise ValueError("window size must be odd number")

    if not backend.all(Et.shape == weights.shape):
        raise ValueError(
            f"w and x shapes should match, got {weights.shape} and {Et.shape}"
        )

    outputs = Et * weights

    if not loss_by_layer:
        outputs = backend.sum(outputs, axis=-1, keepdims=True)

    model = Model([inputs, aux_inputs] if use_abseta else inputs, outputs)

    if lr_decay:
        learning_rate = optimizers.schedules.ExponentialDecay(
            1e-3,
            (epochs * len(train_idxs)) // (4 * batch_size),
            0.5,
            staircase=True,
        )
    else:
        learning_rate = 1e-3

    model.compile(
        loss             = losses.MSE,
        optimizer        = optimizers.Adam(learning_rate=learning_rate),
        weighted_metrics = [],
    )
    return model


sample        = args.sample
max_trials    = args.max_trials
size          = args.window_size
epochs        = args.epochs
batch_size    = args.batch_size
lr_decay      = args.lr_decay
min_Et        = args.min_Et
use_abseta    = args.abseta
overwrite     = args.overwrite
loss_by_layer = args.loss_by_layer
add_dense     = args.add_dense

print("Starting hp scan:")
for k, v in vars(args).items():
    print("-->", k, ":", v)

project = f"qcnn_{sample}_size{size}_minEt{args.min_Et}"
if use_abseta:
    project += "_abseta"
if loss_by_layer:
    project += "_lbl"
if add_dense > 0:
    project += f"_d{add_dense}"

if "delphes" in sample:
    path         = get_config(sample)['dir']
    name         = get_config(sample)['keyword']
    data         = ak.from_parquet(path+'/'+name)
    towers       = data['Towers']
    tower_labels = data['Towers_NoPU']
else:
    cells        = get_data(get_config(sample), filter_name="cell_*")
    towers       = cells_to_towers(cells)
    tower_labels = cells_to_towers(cells, Et_key="cell_et_mu0")

if not loss_by_layer:
    tower_labels = tower_labels.sum(axis=-1, keepdims=True)

if min_Et > 0:
    X = sliding_window(towers, size=size)
    y = sliding_window(tower_labels, size=1)

    centre = size // 2
    mask   = X[..., centre, centre, :].sum(axis=-1) >= args.min_Et
    X, y   = X[mask], y[mask]

else:
    X = towers if size == 1 else pad(towers, size // 2)
    y = tower_labels

if use_abseta:
    abseta = np.log(np.abs(get_tower_eta(towers)))
    if min_Et > 0:
        abseta = sliding_window(abseta, size=1)
        abseta = abseta[mask]

train_idxs, test_idxs = train_test_split(
    np.arange(len(y)),
    test_size    = 0.2,
    random_state = 101,
)

y_train, y_test = y[train_idxs], y[test_idxs]
if use_abseta:
    X_train = [X[train_idxs], abseta[train_idxs]]
    X_test  = [X[test_idxs] , abseta[test_idxs] ]

else:
    X_train = X[train_idxs]
    X_test  = X[test_idxs]

tuner = kt.RandomSearch(
    build_regressor,
    objective    = "val_loss",
    max_trials   = max_trials,
    overwrite    = overwrite,
    directory    = "models/",
    project_name = project,
    seed         = 0,
)

tuner.search(
    X_train,
    y_train,
    validation_data = (X_test, y_test),
    batch_size      = batch_size,
    epochs          = epochs,
    verbose         = 2,
    callbacks       = [
        callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = epochs // 5,
            restore_best_weights = True,
        ),
    ],
)

tuner.results_summary()

best_model = tuner.get_best_models(1)[0]
best_model.summary()

best_model.save(f"models/{project}/best_model.h5")
json.dump(vars(args), open(f"models/{project}/config.json", "w"))
