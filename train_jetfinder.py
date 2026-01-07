import argparse
import os
import random
import json
import numpy             as np
import awkward           as ak
import matplotlib.pyplot as plt
import matplotlib        as mpl
import wp21_train        as wp

from modules.cone     import CNNJetAlgo
from utils.image      import vector_to_tower, tower_to_vector, pad, augment_image
from utils.misc       import awkward_to_vector, sparse_to_awkward
from utils.cells      import to_4momentum
from tensorflow.keras import losses, optimizers, callbacks
from tensorflow.data  import Dataset, AUTOTUNE

class RandomCheckpoint(callbacks.Callback, wp.callbacks.base_callback):
    def __init__(self, out_dir="out/", p=0.1, model_name="", save_weights=False, project_name="", notes=""):
        callbacks.Callback.__init__(self)
        wp.callbacks.base_callback.__init__(self, project_name, notes)
        self._out_dir      = out_dir
        self._model_name   = model_name
        self._p            = p
        self._save_weights = save_weights
        os.makedirs(out_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if random.random() < self._p:
            path = os.path.join(self._out_dir, f"epoch_{self._model_name}_{epoch:04d}.keras")
            if self._save_weights:
                path = os.path.join(self._out_dir, f"epoch_{self._model_name}_{epoch:04d}.weights.h5")
                self.model.save_weights(path)
            else:
                self.model.save(path)
            print(f"\n[RandomCheckPoint] Saved checkpoint: {path}")

def load_data(file_name: str = '/workspace/samples/ggfhh/ggfhh_train.parquet'):
    eos_path = file_name
    data     = ak.from_parquet(eos_path)
    inp_img  = ak.to_numpy(data['Towers_NoPU'])
    tar_img  = data['GenJet_NoPU']
    tar_img  = vector_to_tower(awkward_to_vector(tar_img))
    return [inp_img, tar_img]

def plot_event(in_img, tar_img, pred_img = None, evt_num = 0):
    fig, ax = plt.subplots(1, 3, figsize=(12,5), sharey=True)

    phi_edges = np.linspace(-np.pi, np.pi, 65)
    eta_edges = np.linspace(-2.5  , 2.5  , 51)

    im0 = ax[0].pcolormesh(phi_edges, eta_edges, in_img, norm=mpl.colors.LogNorm(vmin=0.1, vmax=100))
    plt.colorbar(im0)
    ax[0].set_title('Input Event')

    im1 = ax[1].pcolormesh(phi_edges, eta_edges, tar_img, norm=mpl.colors.LogNorm(vmin=0.1, vmax=100))
    plt.colorbar(im1)
    ax[1].set_title('Target Event')

    if pred_img:
        im2 = ax[2].pcolormesh(phi_edges, eta_edges, pred_img, norm=mpl.colors.LogNorm(vmin=0.1, vmax=100))
        plt.colorbar(im2)
        ax[2].set_title('Predicted Event')

    fig.savefig(f'event_{evt_num}.png')

def augment_batch(x,y):
    return augment_image(x), augment_image(y)
    
def make_train_set(inp_imgs, tar_imgs, batch_size=128, conv_size=1, add_pad=False):
    if add_pad:
        inp_imgs = pad(inp_imgs, conv_size//2)
        tar_imgs = pad(tar_imgs, conv_size//2)

    return (Dataset.from_tensor_slices((inp_imgs, tar_imgs)).batch(128).map(lambda x,y: augment_batch(x,y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))            
    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample"  , type=str  , default="ggF_SM_HH4b_train")
    parser.add_argument("-w", "--window"  , type=int  , nargs="+", default=7)
    parser.add_argument("-b", "--batches" , type=int  , default=128)
    parser.add_argument("-f", "--freq"    , type=float, default=0.15)
    parser.add_argument("-e", "--epochs"  , type=int  , default=20)
    parser.add_argument("-l", "--layers"  , type=int  , default=1)
    parser.add_argument("-c", "--conv"    , type=int  , nargs="+", default=1)
    parser.add_argument("-m", "--mlp"     , type=int  , nargs="+", default=1)
    args = parser.parse_args()

    sample_name = args.sample
    conv_size   = args.window
    batches     = args.batches
    channels    = args.layers
    save_freq   = args.freq
    epochs      = args.epochs
    n_convs     = args.conv
    n_mlp       = args.mlp

    conv_tuples = []
    if len(conv_size) > 0 and len(n_convs) > 0:
        for i_conv in range(1,len(n_convs)):
            conv_tuples.append((n_convs[i_conv], conv_size[i_conv], 'relu'))

    mlp_tuples = []
    if len(n_mlp) > 0:
        for i_mlp in range(len(n_mlp)):
            mlp_tuples.append((n_mlp[i_mlp], 'relu'))

    model = CNNJetAlgo(conv_size[0], channels, n_convs[0], conv_tuples, mlp_tuples, fix_layer=True)

    model.summary()

    model.compile(loss=losses.MSE, optimizer=optimizers.Adam(learning_rate=10e-3))

    my_data = load_data()
    
    if channels < 2:
        my_data[0] = my_data[0].sum(axis=-1, keepdims=True)

    my_data[1] = ak.to_numpy(my_data[1])

    train_set  = make_train_set(my_data[0], my_data[1], batches, conv_size[0], True)

    model_name = f'model_s{sample_name}_w{conv_size}_b{batches}_l{channels}_nc{n_convs}_nd{n_mlp}'

    my_callback = RandomCheckpoint('out/',save_freq,model_name,False,'jet_finder')
    
    history = model.fit(
        train_set,
        epochs=epochs,
        batch_size=batches,
        verbose=1,
        callbacks=[my_callback],
        )

    model.save(f'out/trained_{model_name}.keras')
    with open(f'out/train_{model_name}_history.json','w') as f:
        json.dump(history.history, f, indent=2)

if __name__=="__main__":
    main()
