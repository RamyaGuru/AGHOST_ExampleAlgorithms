import awkward as ak
import numpy   as np
import vector
import json
import os
import uproot
import functools

print = functools.partial(print, flush=True)

def get_config(sample):
    return json.load(open('config/samples.json'))[sample]

def get_data(config, **kwargs):
    files = [
        config['dir'] + '/' + f
        for f in os.listdir(config['dir'])
        if config['keyword'] in f and '.root' in f
    ]
    events = uproot.concatenate(files, **kwargs)

    if 'EventWeight' in events.fields:
        weights = ak.to_numpy(events['EventWeight'][:, 0])
        events['weight'] = (
            weights
            * config['xsec']
            * config['hstp_filter_sf']
            * config['filter_eff']
            / weights.sum()
        )
        events = ak.without_field(events, where = 'EventWeight')

    for var in ['cell_et', 'cell_et_mu0']:
        if var in events.fields:
            events[var] = events[var] / 1000

    return events

def awkward_to_vector(obj):
    if "pt" in obj.fields:
        return vector.zip({"pt": obj.pt, "eta": obj.eta, "phi": obj.phi, "m": obj.m})

    if "rho" in obj.fields:
        return vector.zip({"pt": obj.rho, "eta": obj.eta, "phi": obj.phi, "m": obj.tau})

    elif "t" in obj.fields:
        return vector.zip({"px": obj.x, "py": obj.y, "pz": obj.z, "E": obj.t})

    elif "px" in obj.fields:
        return vector.zip({"px": obj.px, "py": obj.py, "pz": obj.pz, "E": obj.E})

    else:
        raise ValueError("Could not detect vector component keys.")

def sparse_to_awkward(arr, min_pt=1e-12):
    mask   = np.asarray(arr.pt > min_pt)
    counts = ak.from_numpy(np.sum(mask.reshape(mask.shape[0], -1), axis=1))
    m      = ak.unflatten(arr.m[mask]  , counts)
    pt     = ak.unflatten(arr.pt[mask] , counts)
    eta    = ak.unflatten(arr.eta[mask], counts)
    phi    = ak.unflatten(arr.phi[mask], counts)
    return vector.zip({"m": m, "pt": pt, "eta": eta, "phi": phi})
