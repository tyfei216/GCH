import torch
import numpy as np
import configparser
import argparse

import os

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def count_map(test,data,test_lab,data_lab):
    qlen = len(test)
    dlen = len(data)

    dist = np.zeros(dlen)
    res = np.zeros(qlen)
    
    for i in range(qlen):
		#print i
        for j in range(dlen):
            # print(i)
			# pdb.set_trace()
            # print()
            dist[j] = sum(test[i]^data[j])
        idx = np.argsort(dist)
        ton = 0
        for k in range(dlen):
            if data_lab[idx[k]]==test_lab[i]:
                # dist[j] = sum(test[i]^data[j])
                # print(k)
                ton = ton+1
                res[i] += ton/(k+1.0)   
        res[i] = res[i]/ton
    return np.mean(res)

def sum_dis(test,data,cnt,dis):
    qlen = len(test)
    dlen = len(data)

    dist = np.zeros(dlen)
    res = np.zeros(qlen)
    
    for i in range(qlen):
		#print i
        for j in range(dlen):
            # print(i)
			# pdb.set_trace()
            # print()
            dist[j] = dis(test[i], data[j])
        idx = np.sort(dist)   
        res[i] = sum(idx[:cnt])/cnt
    return np.mean(res)


def test(encoder, test_x, test_y, train_x, train_y):

    encoder = encoder.eval().cpu()
    test = torch.tensor(test_x).float()
    _, test = encoder(test)

    database = torch.tensor(train_x).float()
    _, database = encoder(database)

    test = test.ge(0.5).int()
    database = database.ge(0.5).int()

    # test = torch.cat(test.values(), 0).numpy()
    res = count_map(test.numpy(), database.numpy(), 
    test_y.astype(np.int32), train_y.astype(np.int32))
        # print(count_map(database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32)))

    print(res)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
    return res
    #fp = open("./result.txt","a")
    #fp.write(label+str(res)+"\n")
    #fp.close()

def original_dis(x, y):
    return sum(x.ge(0.5).int() ^ y.ge(0.5).int())
    
def L1_dis(x, y):
    return sum(abs(x-y))

def L2_dis(x, y):
    return sum((x-y)*(x-y))

def count_distance(encoder, test_x, train_x, cnt = None, dis = None):
    if test_x == None:
        return None 

    if train_x == None:
        return None
    
    if dis == None:
        dis = [original_dis]
    
    if cnt == None:
        cnt = len(train_x)
    
    encoder = encoder.eval().cpu()
    test = torch.tensor(test_x).float()
    _, test = encoder(test)

    database = torch.tensor(train_x).float()
    _, database = encoder(database)

    # test = torch.cat(test.values(), 0).numpy()
    res = []
    for i in dis:
        res.append(sum_dis(test.detach(), database.detach(), cnt, i))
        # print(count_map(database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32)))

    #print(np.array(res))
    return res

def count_negative(encoder, train_X, test_X):
    if test_x == None:
        return None 

    if train_x == None:
        return None

    encoder = encoder.eval().cpu()
    test = torch.tensor(test_x).float()
    ifnew1, test = encoder(test)

    database = torch.tensor(train_x).float()
    ifnew2, database = encoder(database)

    ifnew1t = (ifnew1>0).int().sum()
    ifnew1f = (ifnew1<0).int().sum()
    ifnew2t = (ifnew2>0).int().sum()
    ifnew2f = (ifnew2<0).int().sum()

    return ifnew1f, ifnew2t, ifnew1t, ifnew2f




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='path to the required configure file')
    parser.add_argument('-gpuid', type=str, default=-1, help='given gpu to train on')
    parser.add_argument('-gpu', type=bool, default=False, help='whether to use a gpu')
    parser.add_argument('-save', type=str, default='./checkpoints/try/', help='place to save')
    parser.add_argument('-weights', type=str, required=True, help='the weights')
    args = parser.parse_args()
    
    if args.gpuid != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    config = configparser.ConfigParser()
    config.read(args.cfg)

    encoder = model.emcoder(config)
    # generator = model.Generator(config) 
    # discriminator = model.Discriminator(config)
    if args.gpu:
        encoder = encoder.cuda()
        # generator = generator.cuda()
        # discriminator = discriminator.cuda()

    encoder.load_state_dict(torch.load(args.weights), False)
    # generator = generator.load_state_dict(args.weights+'-generator.pth', args.gpu)
    # discriminator = discriminator.load_state_dict(args.weights+'-discriminator.pth', args.gpu)

    testset = dataset.xmedia_test(config)
    test = totensor(testset.test_feature)
    test = encoder(test)
    database = encoder(totensor(testset.database_feature))

    

    test = test.ge(0.5).int32()
    database = database.ge(0.5).int32()

    # test = torch.cat(test.values(), 0).numpy()
    database_label = torch.cat(tuple(totensor(testset.database_label).values()), 0).numpy()

    print(database_label.size)
    print(database_f.size)

    res = count_map(test.numpy(), database.numpy(), 
     np.array(testset.test_label).astype('int32'), database_label.astype(np.int32))
    # print(count_map(database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32),database_label.astype(np.int32)))

    print(res)
    
