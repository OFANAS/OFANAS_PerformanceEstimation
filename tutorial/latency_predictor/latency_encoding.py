import copy
import torch.nn as nn
import torch


# Helper for constructing the one-hot vectors.
def construct_maps(keys):
    d = dict()
    keys = list(set(keys))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d


ks_map = construct_maps(keys=(3, 5, 7))
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))


def spec2feats(ks_list, ex_list, d_list, r, additional_param=None):
    # This function converts a network config to a feature vector (128-D).
    start = 0
    end = 4
    for d in d_list:
        for j in range(start + d, end):
            ks_list[j] = 0
            ex_list[j] = 0
        start += 4
        end += 4

    # convert to onehot
    ks_onehot = [0 for _ in range(60)]
    ex_onehot = [0 for _ in range(60)]
    r_onehot = [0 for _ in range(8)]

    if additional_param != None:
        """
        core_onehot = [0 for _ in range(8)]
        ram_onehot =  [0 for _ in range(8)]
        bandwidth_onehot = [0 for _ in range(8)]
        """

    for i in range(20):
        start = i * 3
        if ks_list[i] != 0:
            ks_onehot[start + ks_map[ks_list[i]]] = 1
        if ex_list[i] != 0:
            ex_onehot[start + ex_map[ex_list[i]]] = 1

    r_onehot[(r - 112) // 16] = 1

    if additional_param != None:

        x, y, z = additional_param

        """
        core_onehot[(x - 112) // 16] = 1
        ram_onehot[(y - 112) // 16] = 1
        bandwidth_onehot[(z - 112) // 16] = 1
        """

        #There are not actually one hot vectors
        
        core_onehot = [x]
        ram_onehot = [y]
        bandwidth_onehot = [z]
    
    if additional_param == None:
        return torch.Tensor(ks_onehot + ex_onehot + r_onehot)
    else:
        return torch.Tensor(ks_onehot + ex_onehot + r_onehot + core_onehot + ram_onehot + bandwidth_onehot) 


def latency_encoding(child_arch, generalized=False):
    

    ks_list = copy.deepcopy(child_arch['ks'])
    ex_list = copy.deepcopy(child_arch['e'])
    d_list = copy.deepcopy(child_arch['d'])
    r = copy.deepcopy(child_arch['r'])[0]

    if generalized == True:
        core = copy.deepcopy(child_arch['cores'])
        ram = copy.deepcopy(child_arch['ram'])
        bandwidth = copy.deepcopy(child_arch['bandwidth'])
        
        feats = spec2feats(ks_list, ex_list, d_list, r, [core, ram, bandwidth] ).reshape(1, -1)
    else:
        feats = spec2feats(ks_list, ex_list, d_list, r).reshape(1, -1)


    return feats

def trial_encoding(child_arch):

    try:
        feats = child_arch['ks']
        #print(feats)
        return feats
    except:
        print(child_arch)
