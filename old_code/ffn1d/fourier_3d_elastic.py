#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !/usr/bin/env python
# coding: utf-8

# In[1]:


"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities3 import *
from Adam import Adam
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import scipy.io
import pickle

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)


# Complex multiplication
def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 3, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, 2,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.irfft(out_ft, 3, normalized=True, onesided=True, signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        #        self.fc0 = nn.Linear(5, self.width)
        self.fc0 = nn.Linear(4, self.width)  # input channel is 4 = 1+T_in. 1 is for gridt. T_in=3 is for Vp, Vs, source

        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.selu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.selu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.selu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.selu(x)
        x = self.fc2(x)

        return x


class Net3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(Net3d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock3d(modes1, modes2, modes3, width)

    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


# In[ ]:


ntrain = 100
ntest = 20

modes1 = 12
modes2 = 12
modes3 = 12
width = 32

batch_size = 8
batch_size2 = batch_size

epochs = 500
learning_rate = 1e-3
scheduler_step = 50
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

path = 'Elastic_N' + str(ntrain) + '_ep' + str(epochs) + '_m' + str(
    modes1) + '_' + str(modes2) + '_' + str(modes3) + '_w' + str(width) + '_lr' + str(learning_rate) + '_sched' + str(
    scheduler_step) + '_batch' + str(batch_size) + '_selu_loss2_Flayer4' + 'vK'

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 3
T = 134  # data is 5 sec with 0.01s, downsample to 4 sec with dt=0.03s (to be changes)

################################################################
# load data

datapath = '/net/arius/scratch/yyang7/Elastic_ncorr10_sigma10_vK/'
Vsset = np.load(datapath + 'model/Vs.npy')
Vpset = np.load(datapath + 'model/Vp.npy')
srcxy = np.load(datapath + 'model/srcxy.npy')

#####
nx, ny = 64, 64
x = np.linspace(-5e3, 5e3, nx)
y = np.linspace(0, 1e4, ny)

xx, yy = np.meshgrid(x, y, indexing="ij")

vp = 3000.0 * np.ones_like(xx)

src_Frequency = 8
src_Velocity = 3000
src_width = (1 / src_Frequency * src_Velocity)

train_a = np.zeros((ntrain, S, S, T_in))
train_u = np.zeros((ntrain, S, S, T, 2))

test_a = np.zeros((ntest, S, S, T_in))
test_u = np.zeros((ntest, S, S, T, 2))

for i in range(ntrain):
    train_a[i, :, :, 0] = Vpset[i, :, :]
    train_a[i, :, :, 1] = Vsset[i, :, :]
    train_a[i, :, :, 2] = src_Velocity * np.exp(-(xx - srcxy[i, 0]) ** 2 / src_width ** 2
                                                ) * np.exp(-(yy - srcxy[i, 1]) ** 2 / src_width ** 2)

    train_u[i, :, :, :, :] = np.load(datapath + 'waveform/No' + str(i) + '.npy')[:, :, :400:3, :]

offset_test = 4800
for i in range(ntest):
    test_a[i, :, :, 0] = Vpset[offset_test + i, :, :]
    test_a[i, :, :, 1] = Vsset[offset_test + i, :, :]
    test_a[i, :, :, 2] = src_Velocity * np.exp(-(xx - srcxy[offset_test + i, 0]) ** 2 / src_width ** 2
                                               ) * np.exp(-(yy - srcxy[offset_test + i, 1]) ** 2 / src_width ** 2)

    test_u[i, :, :, :, :] = np.load(datapath + 'waveform/No' + str(offset_test + i) + '.npy')[:, :, :400:3, :]

train_a = torch.from_numpy(train_a.astype(np.float32))
train_u = torch.from_numpy(train_u.astype(np.float32))
test_a = torch.from_numpy(test_a.astype(np.float32))
test_u = torch.from_numpy(test_u.astype(np.float32))
print(train_a.shape)
print(test_a.shape)
print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-3])
assert (T == train_u.shape[-2])

# In[ ]:


a_normalizer = UnitGaussianNormalizer(train_a)
y_normalizer = UnitGaussianNormalizer(train_u)

# ----------------------------------------------
# # Saving pickles:
pickle_out = open('/net/arius/scratch/yyang7/fourier_neural_operator/pickle/' + path + '_a_normalizer.pickle', "wb")
pickle.dump(a_normalizer, pickle_out)
pickle_out.close()
pickle_out = open('/net/arius/scratch/yyang7/fourier_neural_operator/pickle/' + path + '_y_normalizer.pickle', "wb")
pickle.dump(y_normalizer, pickle_out)
pickle_out.close()

# In[6]:


# ---------------------------------------------

train_a = a_normalizer.encode(train_a)
# train_u = y_normalizer.encode(train_u)
test_a = a_normalizer.encode(test_a)
# test_u = y_normalizer.encode(test_u)

train_a = train_a.reshape(ntrain, S, S, 1, T_in).repeat([1, 1, 1, T, 1])
test_a = test_a.reshape(ntest, S, S, 1, T_in).repeat([1, 1, 1, T, 1])

# pad locations (x,y,t)
# gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
# gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
# gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
# gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T + 1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

# train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
#                      gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)

# test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
#                     gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)
train_a = torch.cat((gridt.repeat([ntrain, 1, 1, 1, 1]), train_a), dim=-1)
test_a = torch.cat((gridt.repeat([ntest, 1, 1, 1, 1]), test_a), dim=-1)
# -----------------------------------------------------------------
# mask
train_mask = torch.ones_like(train_u)
test_mask = torch.ones_like(test_u)

# Now zero padding to make it periodic for fft
S_pad = S // 16  #
T_pad = T // 16  #

print(train_a.shape)
train_a = F.pad(train_a, (0, 0, T_pad, T_pad, S_pad, S_pad, S_pad, S_pad, 0, 0))
print(train_a.shape)

print(test_a.shape)
test_a = F.pad(test_a, (0, 0, T_pad, T_pad, S_pad, S_pad, S_pad, S_pad, 0, 0))
print(test_a.shape)

print(train_u.shape)
train_u = F.pad(train_u, (0, 0, T_pad, T_pad, S_pad, S_pad, S_pad, S_pad, 0, 0))
print(train_u.shape)

print(test_u.shape)
test_u = F.pad(test_u, (0, 0, T_pad, T_pad, S_pad, S_pad, S_pad, S_pad, 0, 0))
print(test_u.shape)

print(train_mask.shape)
train_mask = F.pad(train_mask, (0, 0, T_pad, T_pad, S_pad, S_pad, S_pad, S_pad, 0, 0))
print(train_mask.shape)

print(test_mask.shape)
test_mask = F.pad(test_mask, (0, 0, T_pad, T_pad, S_pad, S_pad, S_pad, S_pad, 0, 0))
print(test_u.shape)

# In[8]:


# -------------------------------------------------------------------------------
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u, train_mask),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u, test_mask),
                                          batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2 - t1)

# In[ ]:


################################################################
# training and evaluation
################################################################

device = torch.device('cuda:0')
model = Net3d(modes1, modes2, modes3, width).to(device)
# model = torch.load(path_model,map_location=device)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# print(model.count_params())
print(model.module.count_params())
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = Adam(model.module.parameters(), lr=learning_rate, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False, p=2)

y_normalizer.cuda()

losstrain = np.zeros(epochs)
losstest = np.zeros(epochs)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    flag = 0
    for x, y, z in train_loader:
        x, y, z = x.to(device), y.to(device), z.to(device)

        optimizer.zero_grad()
        out = model(x)

        # y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        # ---------------------

        out = out.reshape(z.shape) * z
        # ---------------------

        l2 = myloss(out.view(batch_size, -1),
                    y.view(batch_size, -1))  # * 0.8 + mylossl1(out.view(batch_size, -1), y.view(batch_size, -1)) * 0.2
        l2.backward()

        optimizer.step()
        train_l2 += l2.item()

        flag += 1

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y, z in test_loader:
            x, y, z = x.to(device), y.to(device), z.to(device)
            out = model(x)

            # y = y_normalizer.decode(y)
            out = y_normalizer.decode(out)
            # ---------------------
            out = out.reshape(z.shape) * z
            # ---------------------
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size,
                                                               -1)).item()  # * 0.8 + mylossl1(out.view(batch_size, -1), y.view(batch_size, -1)).item() * 0.2

    print(default_timer() - t1)
    train_l2 /= ntrain
    train_l2_0 /= ntrain

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2, flush=True)
    losstrain[ep] = train_l2 * 1
    losstest[ep] = test_l2 * 1

torch.save(model, path + '_model')
# evaluation of train dataset
obs = train_u[0:10, :, :, :, :]
pred = torch.zeros_like(obs)
for i in range(10):
    x = train_a[i:i + 1, :, :, :, :].to(device)
    out = model(x)
    pred[i, :, :, :] = y_normalizer.decode(out).cpu().detach()

# evaluation of test dataset
obs_test = test_u[0:10, :, :, :, :]
pred_test = torch.zeros_like(obs_test)
for i in range(10):
    x = test_a[i:i + 1, :, :, :, :].to(device)
    out = model(x)
    pred_test[i, :, :, :] = y_normalizer.decode(out).cpu().detach()

np.savez(path + '.npz', pred=pred.numpy()[:, S_pad:S_pad + S, S_pad:S_pad + S, :, :],
         obs=obs.numpy()[:, S_pad:S_pad + S, S_pad:S_pad + S, :, :],
         pred_test=pred_test.numpy()[:, S_pad:S_pad + S, S_pad:S_pad + S, :, :],
         obs_test=obs_test.numpy()[:, S_pad:S_pad + S, S_pad:S_pad + S, :, :],
         losstrain=losstrain, losstest=losstest, losstrainl2=losstrainl2, losstestl2=losstestl2)
