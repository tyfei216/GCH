import data
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random

class KNNdataset(Dataset):
    def __init__(self, train_X, knn):
        super(KNNdataset, self).__init__()
        self.train_X = torch.tensor(train_X).float()
        self.knn = knn
        self.kx = len(knn[0])
    
    def __len__(self):
        return len(self.train_X)

    def __getitem__(self, index):

        a = random.randint(0, self.kx-1)
        b = random.randint(0, len(self)-1)
        while b in self.knn[index]:
            b = random.randint(0, len(self)-1)
        a = self.knn[index][a]

        return self.train_X[index], self.train_X[a], self.train_X[b]
        

class Single(Dataset):
    def __init__(self, train_X, train_Y):
        super(Single, self).__init__()

        self.train_X = torch.tensor(train_X).float()
        self.train_Y = train_Y
        #self.length = len(self.train_Y)

        self.record = {}
        for i in range(len(train_Y)):
            if train_Y[i] not in self.record:
                self.record[train_Y[i]] = []
            self.record[train_Y[i]].append(i)

        self.index = []
        self.st = {}
        for key, value in self.record.items():
            self.index.extend(value)
            lv = len(value)
            self.st[key] = (len(self.index)-lv, len(self.index))

    def save_dataset(self, path):
        np.save(path+"data.npy", self.train_X.numpy())
        np.save(path+"label.npy", self.train_Y)
    
    def __len__(self):
        return len(self.index)

    def count_types(self):
        return len(self.st)

    def set_subset(self, sublist):
        self.index = []
        self.st = {}
        for i in sublist:
            if i in self.record:
                self.index.extend(self.record[i])
                lv = len(self.record[i])
                self.st[i] = (len(self.index)-lv, len(self.index))

    def get_subset(self, sublist = None):
        index = []
        if sublist == None:
            sublist = range(self.count_types())
        for i in sublist:
            if i in self.record:
                index.extend(self.record[i])
        if len(index) == 0:
            return (None, None)
        return self.train_X[index], self.train_Y[index]

    def print_info(self):
        print("length of dataset:", len(self))
        print("num of types:", self.count_types())
        list_of_length = []
        for i in self.st.values():
            list_of_length.append(i[1]-i[0])
        print("length of all types:", list_of_length)

    def __getitem__(self, index):
        index = self.index[index]
        label = self.train_Y[index]
        r1 = random.randint(0, self.st[label][1]-self.st[label][0]-1) + self.st[label][0]
        r2 = random.randint(0, len(self) - self.st[label][1] + self.st[label][0]-1)
        r2 = (r2 + self.st[label][1]) % len(self) 
        r1 = self.index[r1]
        r2 = self.index[r2]
        assert self.train_Y[index] == self.train_Y[r1] 
        assert self.train_Y[index] != self.train_Y[r2]
        return self.train_X[index], self.train_X[r1], self.train_X[r2], self.train_Y[index]


if __name__ == '__main__':
    a = data.read_dataset("../data/Plasschaert/data.h5")
    #a.exprs = a.exprs[:,:1000]
    x,y,z,w = data.getdata(a)
    print(x.shape)
    dataset = Single(x, z)
    print(dataset.catagories)
    dl = DataLoader(dataset, batch_size=5)
    for i, j in enumerate(dl):
        pass
