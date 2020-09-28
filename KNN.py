import numpy as np 
import pickle 
import data

data = np.load('./data/train15720data.npy')
label = np.load('./data/train15720label.npy')

def buildKNN(data, label = None, k = 10):
    knn = []
    if label != None:
        gcnt = 0
    for i in range(data.shape[0]):
        #if i %100 == 0:
        #    print(i)
        dis = []
        for j in range(data.shape[0]):
            if i == j:
                dis.append(10000000000.0)
            else:
                dis.append(np.sqrt(((data[i]-data[j])**2).sum()))
        index = np.argsort(dis)
        knn.append(index[:k])
        if label != None:
            for j in range(k):
                if label[index[j]] == label[i]:
                    gcnt += 1

    if label != None:
        print(gcnt, gcnt/float(len(label)*10))

    return knn

if __name__ =='__main__':
    a = data.read_dataset("../data/Adam/data.h5")
    #a.exprs = a.exprs[:,:1000]
    x,y,z,w = data.getdata(a)

    knn = buildKNN(x, label = z, k = 10)

    with open("./data/Adamknn.pkl", 'wb') as f:
        pickle.dump(knn, f)

    np.save()