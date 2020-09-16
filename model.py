import torch
import torch.nn as nn
import configparser
import torch.nn.functional as F 

class TripletLoss(nn.Module):

    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

    def set_margin(self, margin):
        self.margin = margin

class Classifier(nn.Module):
    def __init__(self, dim_In, dim_Hid, dim_Out, 
        batch_Norm=False, hid_Layers=1):
        
        super(Classifier, self).__init__()
        
        if batch_Norm:
            layers = [nn.Linear(dim_In, dim_Hid), nn.BatchNorm1d(dim_Hid)]
            for _ in range(hid_Layers-1):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Hid), nn.BatchNorm1d(dim_Hid)])
            layers.extend([nn.Tanh(), nn.Linear(dim_Hid, dim_Out), nn.Sigmoid()])
        else:
            layers = [nn.Linear(dim_In, dim_Hid)]
            for _ in range(hid_Layers-1):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Hid)])
            layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Out)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_classifier(cfg):

    dim_In = cfg.getint('classifier', 'dim_In')
    dim_Hid = cfg.getint('classifier', 'dim_Hid')
    dim_Out = cfg.getint('classifier', 'dim_Out')
    batch_Norm = cfg.getboolean('classifier', 'batch_Norm')
    hid_Layers = cfg.getint('classifier', 'hid_Layers')
    
    return Classifier(dim_In, dim_Hid, dim_Out, 
        batch_Norm = batch_Norm, hid_Layers = hid_Layers)

class EncoderM(nn.Module):
    def __init__(self, dim_In, dim_Hid, dim_Out, 
        batch_Norm=False, hid_Layers=1):
        super(EncoderM, self).__init__()

        if batch_Norm:
            layers = [nn.Linear(dim_In, dim_Hid), nn.BatchNorm1d(dim_Hid)]
            for _ in range(hid_Layers-1):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Hid), nn.BatchNorm1d(dim_Hid)])
            #layers.extend([nn.Tanh(), nn.Linear(dim_Hid, dim_Out), nn.Sigmoid()])
        else:
            layers = [nn.Linear(dim_In, dim_Hid)]
            for _ in range(hid_Layers-1):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Hid)])
            #layers.extend([nn.Tanh(), nn.Linear(dim_Hid, dim_Out), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

        self.ifnew = nn.Sequential(nn.Tanh(), nn.Linear(dim_Hid, 1))

        self.hashing = nn.Sequential(nn.Tanh(), nn.Linear(dim_Hid, dim_Out), nn.Sigmoid())

    def forward(self, x):
        first = self.net(x)
        ifnew = self.ifnew(first)
        hashing = self.hashing(first)
        return ifnew, hashing

class Encoder(nn.Module):
    def __init__(self, dim_In, dim_Hid, dim_Out, 
        batch_Norm=False, hid_Layers=1):
        
        super(Encoder, self).__init__()
        
        if batch_Norm:
            layers = [nn.Linear(dim_In, dim_Hid), nn.BatchNorm1d(dim_Hid)]
            for _ in range(hid_Layers-1):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Hid), nn.BatchNorm1d(dim_Hid)])
            layers.extend([nn.Tanh(), nn.Linear(dim_Hid, dim_Out), nn.Sigmoid()])
        else:
            layers = [nn.Linear(dim_In, dim_Hid)]
            for _ in range(hid_Layers-1):
                layers.extend([nn.ReLU(inplace=True), nn.Linear(dim_Hid, dim_Hid)])
            layers.extend([nn.Tanh(), nn.Linear(dim_Hid, dim_Out), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def get_encoder(cfg, et = "M"):

    dim_In = cfg.getint('encoder', 'dim_In')
    dim_Hid = cfg.getint('encoder', 'dim_Hid')
    dim_Out = cfg.getint('encoder', 'dim_Out')
    batch_Norm = cfg.getboolean('encoder', 'batch_Norm')
    hid_Layers = cfg.getint('encoder', 'hid_Layers')

    if et == "M":
        return EncoderM(dim_In, dim_Hid, dim_Out, 
            batch_Norm = batch_Norm, hid_Layers = hid_Layers)
    else:
        return Encoder(dim_In, dim_Hid, dim_Out, 
            batch_Norm = batch_Norm, hid_Layers = hid_Layers)

class CutBCELoss(nn.Module):
    def __init__(self, threshold):
        super(CutBCELoss, self).__init__()
        self.threshold = threshold 
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        y = x * (target-0.5)
        x[y>self.threshold] = 0.0
        return self.crit(x, target)

class Generator(nn.Module):
    def __init__(self, dim_In, dim_Ran, dim_Hid, dim_Out, act = "ReLU"):   
        super(Generator, self).__init__()
        
        layers = [
            nn.Linear(dim_In+dim_Ran, dim_In*2), 
            nn.ReLU(inplace=True),
            nn.Linear(dim_In*2, dim_Hid), 
            nn.ReLU(inplace=True),
            nn.Linear(dim_Hid, dim_Out),
        ]
        if act == 'ReLU':
            layers.append(nn.ReLU())
            self.mul = 1
            self.softmax = None
        elif act == "SoftMax":
            #layers.append(nn.Softmax())
            self.softmax = nn.Softmax(dim = 1)
            self.mul = 10000
        elif act == "Sigmoid":
            #layers.append(nn.Sigmoid())
            self.softmax = nn.Sigmoid()
            self.mul = 10000
        else:
            assert False
        self.net =  nn.Sequential(*layers)

        self.dim_Ran = dim_Ran


    def forward(self, x):
        batch_size = x.shape[0]
        if torch.cuda.is_available():
            rand = torch.rand(batch_size, self.dim_Ran).cuda()
        else:
            rand = torch.rand(batch_size, self.dim_Ran)
        #print("x", x)
        #print("rand", rand)
        x = torch.cat((x, rand), 1)
        x = self.net(x)
        #s = x.sum(axis=1)
        #s = s.detach() + 1
        #x = (x.T/s).T
        if self.softmax != None:
            #print(x.shape)
            y = self.softmax(x)*self.mul
            return y
        return x

def get_generator(cfg:configparser.ConfigParser):
    dim_In = cfg.getint('generator', 'dim_In')
    dim_Hid = cfg.getint('generator', 'dim_Hid')
    dim_Out = cfg.getint('generator', 'dim_Out')
    dim_Ran = cfg.getint('generator', 'dim_Ran')

    return Generator(dim_In, dim_Ran, dim_Hid, dim_Out, act="SoftMax")

class Discriminator(nn.Module):
    def __init__(self, dim_In, dim_Hid, dim_Out, noise=0.0):
        super(Discriminator, self).__init__()

        self.first_Layer = nn.Sequential(
            nn.Linear(dim_In, dim_Hid),
            nn.Dropout(0.25), 
            nn.ReLU(), 
            nn.Linear(dim_Hid, dim_Hid), 
            nn.Dropout(0.25),
            nn.ReLU(),
        )

        self.TF = nn.Linear(dim_Hid, 1)

        self.AC = nn.Linear(dim_Hid, dim_Out)

        self.noise = noise

    def forward(self, x, noise = False):
        if noise:
            if torch.cuda.is_available():
                x = x + self.noise*torch.rand(x.shape).cuda()
            else:
                x = x + self.noise*torch.rand(x.shape)
        o = self.first_Layer(x)
        tf = self.TF(o)
        ac = self.AC(o)
        return tf, ac

def get_discriminator(cfg:configparser.ConfigParser):
    dim_In = cfg.getint('discriminator', 'dim_In')
    dim_Hid = cfg.getint('discriminator', 'dim_Hid')
    dim_Out = cfg.getint('discriminator', 'dim_Out')

    return Discriminator(dim_In, dim_Hid, dim_Out, noise=0.1)

if __name__ == '__main__':
    cfg = configparser.ConfigParser()
    cfg.read('test.ini')
    en = get_encoder(cfg)
    a = torch.randn(5, 23797)
    b = en(a)
    cri = nn.MSELoss()
    print(b)
    print(cri(b, torch.zeros_like(b)))
    print(cri(b,b))
    print(b.shape)
  
