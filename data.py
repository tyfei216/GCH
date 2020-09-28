import h5py 
import numpy as np 
import pandas as pd 
import scipy.sparse 
from sklearn.model_selection import train_test_split
import collections
import functools
import copy

def getdata(expr_data):
    labels = np.asarray(expr_data.obs['mapping']).astype(np.int8)
    train_X, test_X, train_Y, test_Y = train_test_split(expr_data.exprs, labels, test_size=0.2)
    return train_X, test_X, train_Y, test_Y

def split_data(X, Y, test_size=0.2):
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=test_size)
    return train_X, test_X, train_Y, test_Y

class ExprDataSet(object):
    def __init__(self, exprs, obs, var, uns):
        assert exprs.shape[0] == obs.shape[0] and exprs.shape[1] == var.shape[0]
        if scipy.sparse.issparse(exprs):
            self.exprs = exprs.tocsr()
        else:
            self.exprs = exprs
        self.obs = obs
        
        var['index'] = range(var.shape[0])
        
        self.var = var
        self.uns = uns
        self.shape = exprs.shape
    
    def generate_labels(self, mapping = None):
        #self.mat = np.asarray(self.mat.toarray())
        if mapping == None:
            mapping = {}
        self.obs['mapping'] = 0 
        for row in self.obs.iterrows():
            if row[1]['cell_type1'] not in mapping:
                mapping[row[1]['cell_type1']] = len(mapping)
            self.obs.loc[row[0],'mapping'] = mapping[row[1]['cell_type1']]
        

decode = np.vectorize(lambda _x: _x.decode("utf-8"))

def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
        mask = data == "__nAn_RePlAcEmEnT__"
        if np.any(mask):
            data = data.astype(object)
            data[mask] = np.nan
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    # d = utils.dotdict()
    d = {}
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def read_dataset(filename):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(
                dict_from_group(f["obs"]),
                index=decode(f["obs_names"][...])
            )
        var = pd.DataFrame(
                dict_from_group(f["var"]),
                index=decode(f["var_names"][...])
            )
        uns = dict_from_group(f["uns"])

        exprs_handle = f["exprs"]
        if isinstance(exprs_handle, h5py.Group):  # Sparse matrix
            mat = scipy.sparse.csr_matrix((
                exprs_handle['data'][...],
                exprs_handle['indices'][...],
                exprs_handle['indptr'][...]
            ), shape=exprs_handle['shape'][...])
        else:  # Dense matrix
            mat = exprs_handle[...].astype(np.float32)
    
    return ExprDataSet(mat, obs, var, uns)

def merge_datasets(dataset_dict):
    
    dataset_dict = collections.OrderedDict(dataset_dict)
    var_name_list = [dataset.var.index for dataset in dataset_dict.values()]
    var_intersect = functools.reduce(np.intersect1d, var_name_list)
    print("input_dim:", len(var_intersect))

    exprs = []
    labels = []

    mapping = {}
    for item in dataset_dict:
        var = dataset_dict[item].var 
        index = var['index'][var_intersect]
        data_all = dataset_dict[item].exprs.toarray()[:, index]
        print(item, " : ", data_all.shape)
        exprs.append(data_all)
        
        dataset_dict[item].generate_labels(mapping)
        labels.append(np.array(list(dataset_dict[item].obs['mapping'])))

    exprs = np.concatenate(exprs, axis=0)
    labels = np.concatenate(labels, axis=0)

    print("all labels : ", len(mapping))

    return exprs, labels


#not used
'''
def merge_datasets(dataset_dict):
    dataset_dict = collections.OrderedDict(dataset_dict)

    var_name_list = [dataset.var.index for dataset in dataset_dict.values()]
    
    var_union = functools.reduce(np.union1d, var_name_list)
    var_intersect = functools.reduce(np.intersect1d, var_name_list)

    for item in dataset_dict:
        dataset_dict[item] = copy.deepcopy(dataset_dict[item])  # Avoid contaminating original datasets

    merge_uns_slots = []
    merged_slot = {}
    for slot in merge_uns_slots:
        merged_slot[slot] = []
        for dataset in dataset_dict.values():
            merged_slot[slot].append(dataset.uns[slot])
        merged_slot[slot] = np.intersect1d(
            functools.reduce(np.union1d, merged_slot[slot]), var_intersect)

    merged_var = []
    for item in dataset_dict:
        var = dataset_dict[item].var.reindex(var_union)
        var.columns = ["_".join([c, item]) for c in var.columns]
        merged_var.append(var)
    merged_var = pd.concat(merged_var, axis=1)

    merged_obs = []
    for key in dataset_dict.keys():
        merged_obs.append(dataset_dict[key].obs)
    merged_obs = pd.concat(merged_obs, sort=True)

    if np.any([scipy.sparse.issparse(dataset.exprs)
            for dataset in dataset_dict.values()]):
        merged_exprs = scipy.sparse.vstack([scipy.sparse.csr_matrix(dataset.exprs)
            for dataset in dataset_dict.values()])
    else:
        merged_exprs = np.concatenate([dataset.exprs for dataset in dataset_dict.values()], axis=0)

    return ExprDataSet(merged_exprs, merged_obs, merged_var, merged_slot)

'''

if __name__ == '__main__':
    '''
    Baron_human = read_dataset("../data/Baron_human/data.h5")
    Muraro = read_dataset("../data/Muraro/data.h5")
    Enge = read_dataset("../data/Enge/data.h5")
    Segerstolpe = read_dataset("../data/Segerstolpe/data.h5")
    Xin_2016 = read_dataset("../data/Xin_2016/data.h5")
    Lawlor = read_dataset("../data/Lawlor/data.h5")
    merge = {'Baron_human':Baron_human, 'Muraro':Muraro, 'Enge':Enge, 'Segerstolpe':Segerstolpe, 
    'Xin_2016':Xin_2016, 'Lawlor':Lawlor}
    
    '''
    Quake_Smart_seq2 = read_dataset("../data/Quake_Smart-seq2/data.h5")
    Quake_10x = read_dataset("../data/Quake_10x/data.h5")
    merge = {"A":Quake_Smart_seq2, "B":Quake_10x}
    
    
    mergedexpr, mergedl = merge_datasets(merge)
    s = mergedexpr.sum(axis=1)
    x = (mergedexpr.T/s).T
    x = x.astype(np.float)
    x = x * 10000
    print(x.max(axis=1))
    print(x.max(axis=1).max())
    print(mergedexpr.shape, mergedl.shape)

    np.save("./Quake.npy", x)
    np.save("./QuakeL.npy", mergedl)
    
    #merged.generate_labels()
    '''
    var_name_list = [a.var.index, b.var.index]
    var_intersect = functools.reduce(np.intersect1d, var_name_list)
    print(len(var_intersect))
    #print(a.var)
    k = a.var['index'] 
    k = k[var_intersect]
    print(k)
    l = b.var['index']
    l = l[var_intersect]
    print(l)
    print(list(k))
    print(list(l))
    '''
    '''
    var_intersect = functools.reduce(np.intersect1d, var_name_list)
    
    a = merged
    print(a.var) 
    print(a.obs)

    a = np.asarray(a.obs['mapping'])
    print(a.astype(np.int8))
    '''
