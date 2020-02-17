import torch
import torchvision
import torchvision.datasets as dataset
mnist_trainset = dataset.MNIST(root='./data', train=True, download=False)
from matplotlib import pyplot as plt



# Dataset / Dataloader

class MonDataset(torch.utils.data.Dataset):
    def __init__(self,set):
        self.datos = set

    def __getitem__(self,index):
        aux = self.datos.data[index]
        aux = aux.float() / 255
        aux = aux.view(aux.shape[0]*aux.shape[1])
        return(aux,self.datos.targets[index].view(1))

    def __len__(self):
        return(len(self.datos))


mon_dataset = MonDataset(mnist_trainset)
print(mon_dataset[1][0].shape)

Batch_size = 50

data = torch.utils.data.DataLoader(MonDataset(mnist_trainset), shuffle = True, batch_size = Batch_size)
i = 0
for x,y in data:
    i += 1

print("Il y a", i, "batch de taille",Batch_size)

# Contrôle

import numpy as np

index_random = np.random.randint(0,60000,8)

test_random = torch.stack([mon_dataset[i][0] for i in index_random])

# Fonctions

d_hidden = 10

norm_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(d_hidden),torch.eye(d_hidden))

def KLloss(mu,sigm):
    res = -0.5*torch.sum(1 + sigm - mu*mu - torch.exp(sigm))
    return(res)

def mult_norm(mu,sigm,sample):
    sig = torch.sqrt(torch.exp(sigm))
    return(mu + sig*sample)

def tracage(writer,img_batch_r):

    y_inter = encoder_layer(test_random)
    mu = mu_layer(y_inter)
    sigm = sigm_layer(y_inter)

    x_sample = mult_norm(mu,sigm,norm_distribution.sample())
    y_pred = decoder_layer(x_sample)

    img_batch_r += [torch.stack([y_pred[i].reshape(28,28) for i in range(8)])]


# Encoder | Decoder

encoder_layer = torch.nn.Sequential(
    torch.nn.Linear(28*28,50),
    torch.nn.ReLU()
)

mu_layer = torch.nn.Sequential(
    torch.nn.Linear(50,d_hidden)
)

sigm_layer = torch.nn.Sequential(
    torch.nn.Linear(50,d_hidden)
)

decoder_layer = torch.nn.Sequential(
    torch.nn.Linear(d_hidden,70),
    torch.nn.ReLU(),
    torch.nn.Linear(70,28*28),
    torch.nn.Sigmoid()
)

# Learning

from torch.utils.tensorboard import SummaryWriter

nb_epoch = 32

loss_fn = torch.nn.BCELoss(reduction = "sum")
l_r = 1e-4

params = list(encoder_layer.parameters()) + list(mu_layer.parameters()) + list(sigm_layer.parameters()) + list(decoder_layer.parameters())

optim = torch.optim.Adam(params,lr=l_r)

writer = SummaryWriter("runs/TME10AMAL")

with torch.no_grad():
    img_batch_r = []
    img_batch_r += [torch.stack([test_random[i].reshape(28,28) for i in range(8)])]

i = 0

for epoch in range(nb_epoch):
    data = torch.utils.data.DataLoader(MonDataset(mnist_trainset), shuffle = True, batch_size = Batch_size)
    for x,_ in data:

        y_inter = encoder_layer(x)
        mu = mu_layer(y_inter)
        sigm = sigm_layer(y_inter)

        loss_1 = KLloss(mu,sigm)

        x_sample = mult_norm(mu,sigm,norm_distribution.sample())
        y_pred = decoder_layer(x_sample)

        loss_2 = loss_fn(y_pred,x)

        loss = loss_1 + loss_2

        optim.zero_grad()
        loss.backward()
        optim.step()
        writer.add_scalar("VAE/1-4-2050-8", loss_1, i)
        writer.add_scalar("VAE/2-4-2050-8", loss_2, i)
        i += 1
    if epoch%2 == 0 or epoch == (nb_epoch-1):
        with torch.no_grad():
            tracage(writer,img_batch_r)
    print(epoch)

img_batch_r = torch.stack(img_batch_r).reshape(8*((nb_epoch//2)+2),1,28,28)
writer.add_images('3-4-2050-8', img_batch_r,0)
writer.close()
