import torch
import torch.optim as optim
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

import configparser
import inspect
import model, data, dataset, config, test

def maskBig(x, target, threshold):
    y = x * (target-0.5)
    x[y>threshold] = 0.0
    return x


if __name__ == '__main__':
    #parser.add_argument('-save', type=str, default = './checkpoint/test/', help='place to save')
    _path = ''#'/content/drive/My Drive/Colab Notebooks/myblast/'
    config = configparser.ConfigParser()
    config.read(_path+'mixed_23341.ini')
    #gpu_tracker.track()
    encoder = model.get_encoder(config, "M").cuda()
    discriminator = model.get_discriminator(config).cuda()
    generator = model.get_generator(config).cuda()
    encoder = encoder.cpu()
    encoder = encoder.cuda()
    #classifier = model.get_classifier(config).cuda()
    #gpu_tracker.track()
    #optimC = optim.Adam(classifier.parameters(), lr=config.getfloat('training', 'lr'))
    optimE = optim.Adam(encoder.parameters(), lr=config.getfloat('training', 'lr')*0.01) 
    optimG = optim.Adam(generator.parameters(), lr=config.getfloat('training', 'lr'))
    optimD = optim.Adam(discriminator.parameters(), lr=config.getfloat('training', 'lr'))

    
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
    x,y,z,w = data.split_data(x, mergedl)

    '''
    train_set = dataset.Single(x, z)
    dl = DataLoader(train_set, batch_size=60, shuffle=True)

    test_set = dataset.Single(y, w)

    crit = nn.BCEWithLogitsLoss()
    lam = 0.1
    mse = nn.MSELoss()
    cri = nn.TripletMarginLoss(margin=0.5, p=2)

    checkpoint_path = os.path.join('./', '{epoch}-{net}.pth')
    
    for i in range(320):
        for _, data_in in enumerate(dl):
            #print(data[0].shape)
            _, ori = encoder(data_in[0].cuda())
            #classout = classifier(data_in[0].cuda())
            _, pos = encoder(data_in[1].cuda())
            _, neg = encoder(data_in[2].cuda())
            #print(data[3])
            #print(data_in[3].shape)
            #goal = torch.randn(data_in[3].shape[0], 32).cuda()
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
        if (i+1) % 40 == 0:
            test.test(encoder, y,w,x,z)
    
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
    '''
    print('finish')
    '''

    
    mseloss = 0
    threshold = 30.0

    for i in range(160):
        for j, data_in in enumerate(dl):
            data_in[0] = data_in[0].cuda()
            data_in[1] = data_in[1].cuda()
            data_in[2] = data_in[2].cuda()

            #print(data_in[0].shape)
            ifnew, ori = encoder(data_in[0])
            pos = encoder(data_in[1])
            neg = encoder(data_in[2])

            ori_d = ori.detach()
            ori_d = ori_d.ge(0.5).float()

            rand = torch.rand(ori_d.shape)
            rand = rand.ge(0.5).float().cuda()

            gent = generator(ori_d)
            genf = generator(rand)
            
            if j % 10 == 0:
                oridw, oridc = discriminator(data_in[0])
                tw, tc = discriminator(gent.detach()*10000)
                fw, fc = discriminator(genf.detach()*10000)

                lossD = crit(maskBig(oridw, torch.ones_like(oridw).cuda(), threshold), torch.ones_like(oridw).cuda())+ \
                crit(maskBig(tw, torch.zeros_like(tw).cuda, threshold), torch.zeros_like(tw).cuda())#+ \
                #crit(fw, torch.zeros_like(fw).cuda())
                lossD += crit(maskBig(oridc, ori_d, threshold*5), ori_d) + crit(maskBig(fc, rand, threshold * 5), rand)
                optimD.zero_grad()
                lossD.backward()
                optimD.step()

            tw, tc = discriminator(gent*10000)
            fw, fc = discriminator(genf*10000)

            if (j / 500) % 3 == 0:
                lossGb = crit(maskBig(tw, torch.ones_like(tw).cuda(), threshold), torch.ones_like(tw).cuda()) +\
                     crit(maskBig(fw, torch.ones_like(fw).cuda(), threshold), torch.ones_like(fw).cuda()) 
            else:
                lossGb = 0
            lossG = lossGb + (crit(maskBig(tc, ori_d, threshold*5), ori_d) +\
                 crit(maskBig(fc, rand, threshold*5), rand))
            
            optimG.zero_grad()
            lossG.backward()
            optimG.step()

            gent = generator(ori_d)
            ent = encoder(gent*10000)
            genf = generator(rand)
            enf = encoder(genf*10000)

            lossE = cri(ori, pos, neg)
            if i>800:
                mseloss = mse(enf, rand) + mse(ent, ori_d) 
            lossE = lossE + mseloss
            optimE.zero_grad()
            lossE.backward()
            optimE.step()
        
        if (i+1) % 40 == 0:
            print(i, test.test(encoder, y,w,x,z))
        print(i, "lossE", lossE.item(), "lossG", lossGb.item(), lossG.item(), "lossD", lossD.item(), "mseloss", mseloss)

    print(test.test(encoder, y,w,x,z))
    
    torch.save(encoder.state_dict(), checkpoint_path.format(net='encoder222', epoch=1))
    #torch.save(generator.state_dict(), checkpoint_path.format(net='generator', epoch=1))
    #torch.save(discriminator.state_dict(), checkpoint_path.format(net='discriminator',epoch=1))
