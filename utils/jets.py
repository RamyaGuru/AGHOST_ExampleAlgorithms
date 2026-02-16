import fastjet
import vector
import json
import awkward as ak
import numpy   as np

from sklearn.model_selection import train_test_split

def truth_match(recon, truth, max_dR=0.4, bijective=False):
    def drop_duplicate_pairs(a, b):
        max_b = ak.max(b, axis=-1, keepdims=True)
        key   = a * (max_b + 1) + b

        order      = ak.argsort(key, axis=-1)
        a_sorted   = a[order]
        b_sorted   = b[order]
        key_sorted = key[order]

        keep = ak.run_lengths(key_sorted) == 1

        a_unique = a_sorted[keep]
        b_unique = b_sorted[keep]

        restore = ak.argsort(order[keep], axis=-1)
        return a_unique[restore], b_unique[restore]
    
    if len(truth) != len(recon):
        raise ValueError("Truth and reconstructed jet collections must have the same length.")
    
    recon_idxs, truth_idxs = ak.unzip(ak.argcartesian([recon, truth]))

    dR = recon[recon_idxs].deltaR(truth[truth_idxs])    

    is_close = dR < max_dR

    recon_idxs = recon_idxs[is_close]
    truth_idxs = truth_idxs[is_close]   

    if bijective:
        recon_idxs, truth_idxs = drop_duplicate_pairs(recon_idxs, truth_idxs)   

    matched_truth = truth[truth_idxs]
    matched_recon = recon[recon_idxs]
    return matched_recon, matched_truth

def antikt_jets(vectors, min_pt, r=0.4):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
    jets = fastjet.ClusterSequence(vectors, jetdef)
    return jets.inclusive_jets(min_pt)