import torch
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

import configparser
import inspect
import model, data, dataset, config, test
import pickle

def maskBig(x, target, threshold):
    y = x * (target-0.5)
    x[y>threshold] = 0.0
    return x

if __name__ == '__main__':
    #parser.add_argument('-save', type=str, default = './checkpoint/test/', help='place to save')
    _path = ''#'/content/drive/My Drive/Colab Notebooks/myblast/'
    
    config = configparser.ConfigParser()
    config.read(_path+'mixed_15720.ini')
    #gpu_tracker.track()
    encoder = model.get_encoder(config, "M")
    discriminator = model.get_discriminator(config)
    generator = model.get_generator(config)
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        discriminator = discriminator.cuda()
        generator = generator.cuda()
    #classifier = model.get_classifier(config).cuda()
    #gpu_tracker.track()
    #optimC = optim.Adam(classifier.parameters(), lr=config.getfloat('training', 'lr'))
    optimE = optim.Adam(encoder.parameters(), lr=config.getfloat('training', 'lr')*0.01) 
    optimG = optim.Adam(generator.parameters(), lr=config.getfloat('training', 'lr'))
    optimD = optim.Adam(discriminator.parameters(), lr=config.getfloat('training', 'lr'))

    '''
    Quake_Smart_seq2 = data.read_dataset(_path+"../data/Quake_Smart-seq2/data.h5")
    Quake_10x = data.read_dataset(_path+"../data/Quake_10x/data.h5")
    merge = {"A":Quake_Smart_seq2, "B":Quake_10x}
    mergedexpr, mergedl = data.merge_datasets(merge)
    s = mergedexpr.sum(axis=1)
    x = (mergedexpr.T/s).T
    x = x * 10000
    x,y,z,w = data.split_data(x, mergedl, test_size=0.01)
    '''
    
    Baron_human = data.read_dataset(_path+"../data/Baron_human/data.h5")
    Muraro = data.read_dataset(_path+"../data/Muraro/data.h5")
    Enge = data.read_dataset(_path+"../data/Enge/data.h5")
    Segerstolpe = data.read_dataset(_path+"../data/Segerstolpe/data.h5")
    Xin_2016 = data.read_dataset(_path+"../data/Xin_2016/data.h5")
    Lawlor = data.read_dataset(_path+"../data/Lawlor/data.h5")
    merge = {'Baron_human':Baron_human, 'Muraro':Muraro, 'Enge':Enge, 'Segerstolpe':Segerstolpe, 
    'Xin_2016':Xin_2016, 'Lawlor':Lawlor}
    mergedexpr, mergedl = data.merge_datasets(merge)

    s = mergedexpr.sum(axis=1)
    x = (mergedexpr.T/s).T
    x = x*10000
    #x = x[: ,:1000]
    whole_set = dataset.Single(x, mergedl)
    
    x,y,z,w = data.split_data(x, mergedl) 
    whole_set.print_info()

    for 
    
    exit()
    
    x = np.load("./data/train15720data.npy")
    z = np.load("./data/train15720label.npy")
    y = np.load("./data/test15720data.npy")
    w = np.load("./data/test15720label.npy")
    

    train_set = dataset.Single(x, z)
    test_set = dataset.Single(y, w)
    dl = DataLoader(train_set, batch_size=60, shuffle=True)

    #test_set = dataset.Single(y, w)
    
    #all_dis = [test.original_dis, test.L1_dis, test.L2_dis]
    #print(test.count_distance(encoder, whole_set.get_subset([20])[0], x, cnt=50, dis=all_dis))

    crit = model.CutBCELoss(30)
    crit2 = model.CutBCELoss(150)
    lam = 0.1
    mse = nn.MSELoss()
    cri = nn.TripletMarginLoss(margin=0.5, p=2)
    threshold = 50
    print('loading model')
    encoder.load_state_dict(torch.load("./1-encoder_1.pth"),False)
    checkpoint_path = os.path.join('./', '{epoch}-{net}.pth')
    
    
    #test.test(encoder, y,w,x,z)

    #train_set.save_dataset("./data/train15720")
    #test_set.save_dataset("./data/test15720")
    
    '''
    for i in range(40):
        for _, data_in in enumerate(dl):
            #print(data[0].shape)
            #print(data_in[0].shape)
            ran = torch.rand(data_in[0].shape)
            if torch.cuda.is_available():
                data_in[0] = data_in[0].cuda()
                data_in[1] = data_in[1].cuda()
                data_in[2] = data_in[2].cuda()
                ran = ran.cuda()
 
            _, ori = encoder(data_in[0] + ran)
            #classout = classifier(data_in[0].cuda())
            _, pos = encoder(data_in[1])
            _, neg = encoder(data_in[2])
            #print(data[3])
            #print(data_in[3].shape)
            
            
            #data_in[3] = data_in[3].long().cuda()
            #for k in range(data_in[3].shape[0]):
            #  goal[k] = target[data_in[3][k]]

            #lossE = (ori-pos)*(ori-pos)-(ori-neg)*(ori-neg)+0.5
            lossE = cri(ori, pos, neg) 
            #lossE[lossE<0] = 0
            #lossE = lossE.mean()
            #lossC = crit(classout, data_in[3].long().cuda())
            optimE.zero_grad()
            lossE.backward()
            optimE.step()
            #optimC.zero_grad()
            #lossC.backward()
            #optimC.step()
        print("training ", i, " lossE ", lossE.item())#, " LossC ", lossC.item())


    torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder_1', epoch=1))
    test.test(encoder, y,w,x,z)
    '''    
    #all_dis = [test.original_dis, test.L1_dis, test.L2_dis]

    #dis = np.zeros((3))

    #num = 20.0
    '''
    for i in range(20):
        print("producing ", i)
        res = test.count_distance(encoder, test_set.get_subset([i])[0], train_set.get_subset([i])[0], dis=all_dis)
        if res == None:
            num -= 1.0
        else:
            dis += res

    print(dis/num)

    print(test.count_distance(encoder, whole_set.get_subset([20])[0], x, cnt=50, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([21])[0], x, cnt=50, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([22])[0], x, cnt=50, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([23])[0], x, cnt=50, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([24])[0], x, cnt=50, dis=all_dis))

    print(test.count_distance(encoder, whole_set.get_subset([20])[0], x, cnt=100, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([21])[0], x, cnt=100, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([22])[0], x, cnt=100, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([23])[0], x, cnt=100, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([24])[0], x, cnt=100, dis=all_dis))

    print(test.count_distance(encoder, whole_set.get_subset([20])[0], x, cnt=500, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([21])[0], x, cnt=500, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([22])[0], x, cnt=500, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([23])[0], x, cnt=500, dis=all_dis))
    print(test.count_distance(encoder, whole_set.get_subset([24])[0], x, cnt=500, dis=all_dis))
    '''
    
    '''
    data_in = (torch.randn(20, 15720),torch.randn(20, 15720),torch.randn(20, 15720))
    crit = nn.MSELoss()
    lam = 0.1
    mse = nn.MSELoss()
    cri = nn.TripletMarginLoss(margin=0.4, p=2)

    print('start running')
    
    for _, data_in in enumerate(dl):
    
        ori = encoder(data_in[0].cuda())
        pos = encoder(data_in[1].cuda())
        neg = encoder(data_in[2].cuda())
        break
    
    #ori = encoder(data_in[0].cuda())
    #pos = encoder(data_in[1].cuda())
    #neg = encoder(data_in[2].cuda())
    #ori_d = ori.detach()
    #ori_d = ori_d.ge(0.5).float()

    #rand = torch.rand(ori_d.shape)
    #rand = rand.ge(0.5).float().cuda()

    #gent = generator(ori_d)
    #genf = generator(rand)

    #oridw = discriminator(data_in[0].detach())
    #tw, tc = discriminator(data_in[1].detach())
    #fw, fc = discriminator(genf.detach())

    #print("only one mse333")
    #lossD = crit(oridc, tc)#+ \
    #crit(tw, torch.zeros_like(tw).cuda())+ \
    #crit(fw, torch.zeros_like(fw).cuda())
    #lossD = oridw

    #print(oridw.shape, oridc.shape)
    #print(tw.shape, tc.shape)

    #lossE = cri(ori, pos, neg)# + mse(en, rand)
    #optimE.zero_grad()
    #lossE.backward()
    #optimE.step()
    #lossD = lossD + lam*(crit(oridc, ori_d) + crit(tc, ori_d) + crit(fc, rand))
    lossD = lossD.mean()    
    optimD.zero_grad()
    #gpu_tracker.track()
    #time.sleep(10)
    #lossD = lossD.mean()

    
    lossD.backward()

    optimD.step()
    '''
    
    
    
    mseloss = 0

    iters = 0
    for i in range(500):
        print('epoch', i)
        for j, data_in in enumerate(dl):
            if torch.cuda.is_available():
                data_in[0] = data_in[0].cuda()
                data_in[1] = data_in[1].cuda()
                data_in[2] = data_in[2].cuda()

            #print(data_in[0].shape)
            ifnew, ori = encoder(data_in[0])
            #print(ori.shape)
            _, pos = encoder(data_in[1])
            _, neg = encoder(data_in[2])

            ori_d = ori.detach()
            ori_d = ori_d.ge(0.5).float()

            rand = torch.rand(ori_d.shape)
            rand = rand.ge(0.5).float()
            if torch.cuda.is_available():
                rand = rand.cuda()

            gent = generator(ori_d)
            #print(gent.shape)
            genf = generator(rand)
            
            if True:
                oridw, oridc = discriminator(data_in[0], noise=True)
                tw, tc = discriminator(gent.detach())
                fw, fc = discriminator(genf.detach())

                target1 = torch.ones_like(oridw)
                target2 = torch.zeros_like(tw)
                if torch.cuda.is_available():
                    target1 = target1.cuda()
                    target2 = target2.cuda() 
                    #lossDb = oridw.mean() - tw.mean()
                    #lossDb = crit(oridw, torch.ones_like(oridw).cuda())+ \
                    #crit(tw, torch.zeros_like(tw).cuda())# + crit(fw, torch.zeros_like(fw).cuda())#+ \
                #else:
                #    lossDb = crit(oridw, torch.ones_like(oridw))+ \
                #    crit(tw, torch.zeros_like(tw))# + crit(fw, torch.zeros_like(fw))
                
                lossDb = crit(oridw, target1) + \
                crit(tw, target2) + crit(fw, target2)
                
                lossD = lossDb + 10 * crit2(oridc, ori_d) + crit2(fc, rand)
                optimD.zero_grad()
                lossD.backward()
                optimD.step()

            tw, tc = discriminator(gent)
            fw, fc = discriminator(genf)
            target3 = torch.ones_like(tw) 
            target4 = torch.ones_like(fw)
            if torch.cuda.is_available():
                target3 = target3.cuda()
                target4 = target4.cuda()
                #lossGb = mse(tw, data_in[0])#tw.mean() + fw.mean()
                #lossGb = crit(fw, torch.ones_like(fw).cuda()) + crit(tw, torch.ones_like(tw).cuda())
            #else:
            #    lossGb = crit(tw, torch.ones_like(tw)) + crit(fw, torch.ones_like(fw)) 
            
            if (iters/50) % 2 == 0:
                lossGb = crit(tw, target3) + \
                     crit(fw, target4) 
            else:
                lossGb = 0

            iters += 1

            lossG = lossGb + (crit2(tc, ori_d) + crit2(fc, rand))
            
            
            #print(torch.max(gent, axis = 1)) 
            #print(torch.max(data_in[0], axis = 1))
            optimG.zero_grad()
            lossG.backward()
            optimG.step()
            
            
            if i>100:
                lossE = cri(ori, pos, neg)
                #print(ori_d.shape)
                gent = generator(ori_d)
                #print(gent.shape)
                _, ent = encoder(gent)
                #print(ent.shape)
                genf = generator(rand)
                fakelabel, enf = encoder(genf)
                #print(gent.sum(axis=1), gent.shape)
                #print(genf.sum(axis=1), genf.shape)
                mseloss = mse(enf, rand) + mse(ent, ori_d) 
                lossE = lossE + mseloss

                target5 = torch.ones_like(ifnew)
                target6 = torch.zeros_like(fakelabel)
                if torch.cuda.is_available():
                    target5 = target5.cuda()
                    target6 = target6.cuda()

                lossE += crit(ifnew, target5) + crit(fakelabel, target6)
                optimE.zero_grad()
                lossE.backward()
                optimE.step()
            
        
        if (i+1) % 40 == 0:
            print(i)
            test.test(encoder, y,w,x,z)
        print(i, "lossG", lossGb, lossG, "lossD", lossDb, lossD)
        #print(i, "lossE", lossE.item(), "mseloss", mseloss)
        print(gent.shape, ori_d.shape)
        print(genf.shape, rand.shape)
        print(data_in[0])
        print(torch.max(data_in[0], axis=1))
        print(gent[0])
        print(torch.max(gent, axis=1))
        print(fw)
        print(tw)
        #print(torch.min(gent, axis=1))
        print(gent.sum(axis=1))
    print(test.test(encoder, y,w,x,z))
    
    torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder_10', epoch=1))
    torch.save(generator.state_dict(), checkpoint_path.format(net='generator_10', epoch=1))
    torch.save(discriminator.state_dict(), checkpoint_path.format(net='discriminator_10',epoch=1))
