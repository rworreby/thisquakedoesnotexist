import torch.nn as nn
import torch.nn.functional as F
import torch

def embed(in_chan, out_chan):
    """
        Creates embeding with 4 dense layers
        Progressively grows the number of output nodess
    """
    layers = nn.Sequential(
        nn.Linear(in_chan, 32), torch.nn.ReLU(),
        nn.Linear(32, 64), torch.nn.ReLU(),
        nn.Linear(64, 256), torch.nn.ReLU(),
        nn.Linear(256, 512), torch.nn.ReLU(),
        nn.Linear(512, out_chan), torch.nn.ReLU(),
    )

    return layers


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        #  --- init function ---

        #  get embeddings for the conditional variables
        self.embed1 = embed(1, 1000)
        self.embed2 = embed(1, 1000)
        self.embed3 = embed(1, 1000)

        # first layer
        # concatenate conditional variables | (6, 4000, 1) out
        # (Chan,H,W)
        # (2,1000,1) input 2-D tensor input
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), )
        # (16,500,1) out | apply F.leaky_relu
        self.conv1b = nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # (16,500,1) out | apply F.leaky_relu

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), )
        # (32,250,1) out | apply F.leaky_relu
        self.conv2b = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # (32,250,1) out | apply F.leaky_relu

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), )
        # (64,125,1) out | apply F.leaky_relus
        self.conv3b = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # (64,125,1) out | apply F.leaky_relu

        self.fc0 = nn.Linear(125, 110)
        # (64x110) out | apply F.leaky_relu
        self.fc1 = nn.Linear(110, 128)
        # (64x128) out | apply F.leaky_relu
        self.fc1b = nn.Linear(128, 100)
        # (64x100) out | apply F.leaky_relu | flatten op | (64*100)
        self.fc2 = nn.Linear(64 * 100, 1)
        # (1) out

    def forward(self, x, vc, ):
        # print('---------------- Discriminator -------------')

        # conv2D + leaky relu activation
        # print('shape x init:', x.shape)

        # reshape for one channel convolutions
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)
        v3 = self.embed3(v3)

        # # reshape
        v1 = v1.view(-1, 1, 1000, 1)
        v2 = v2.view(-1, 1, 1000, 1)
        v3 = v3.view(-1, 1, 1000, 1)

        # concatenate conditional variables to input
        x = torch.cat([x, v1, v2, v3], 1)
        # print('torch.cat([x, vc], 1)', x.shape)

        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv1 shape:', x.shape)

        x = self.conv1b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv1b shape:', x.shape)

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv2 shape:', x.shape)

        x = self.conv2b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv2b shape:', x.shape)

        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv3 shape:', x.shape)

        x = self.conv3b(x)
        x = F.leaky_relu(x, 0.2)
        # print('conv3b shape:', x.shape)

        x = torch.squeeze(x, dim=3)
        # print('torch.squeeze shape:', x.shape)

        x = self.fc0(x)
        x = F.leaky_relu(x, 0.2)
        # print('fc0 shape:', x.shape)

        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2)
        # print('fc1 shape:', x.shape)

        x = self.fc1b(x)
        x = F.leaky_relu(x, 0.2)
        # print('fc1b shape:', x.shape)

        # flsatten to input into Last FC layer
        x = x.view(-1, 64 * 100)
        # print('view(-1, 64 * 100):', x.shape)

        out = self.fc2(x)
        # print('fc2 shape:', out.shape)

        return out


class Generator(nn.Module):

    def __init__(self, z_size):
        super(Generator, self).__init__()

        # complete init function

        # first, fully-connected layer
        # input: (1,100) noise vector
        self.fc00 = nn.Linear(z_size, 150, bias=False)
        # output: (1,150)
        self.batchnorm00 = nn.BatchNorm1d(1)
        # output: (1,150) | apply expanddim  | (1,150,1) out

        #  ---- get embeddings for the conditional variables ----
        self.embed1 = embed(1, 150)
        self.embed1 = embed(1, 150)
        self.embed1 = embed(1, 150)

        # ------------------------------------------------------

        # output after concatenating conditional variables: | (2, 150, 1) out

        self.conv0 = nn.Conv2d(4, 6, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (6, 150, 1) | apply batchnorm s
        self.batchnorm0 = nn.BatchNorm2d(6)
        # output: (6, 150, 1)

        self.conv0b = nn.Conv2d(6, 6, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (6,150,1) | apply batchnorm
        self.batchnorm0b = nn.BatchNorm2d(6)
        # output: (6,150,1)

        self.conv0c = nn.Conv2d(6, 1, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (1,150,1) | apply batchnorm
        self.batchnorm0c = nn.BatchNorm2d(1)
        # output: (1,150,1) | apply torch.squeeze | (1,150) out

        self.fc01 = nn.Linear(150, 250, bias=False)
        # output: (1,250)
        self.batchnorm01 = nn.BatchNorm1d(1)
        # output: (1,250) | apply reshape | (2, 125, 1) out

        self.resizenn1 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )
        # utput: (2, 250, 1)

        self.conv1 = nn.Conv2d(2, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (16, 250, 1)
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.conv1b = nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (16, 250, 1)
        self.batchnorm1b = nn.BatchNorm2d(16)

        self.resizenn2 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )
        # output: (16, 500, 1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (32, 500, 1)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv2b = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (32, 500, 1)
        self.batchnorm2b = nn.BatchNorm2d(32)

        #-- resize operation --
        self.resizenn3 = nn.Upsample(scale_factor=(2, 1), mode='nearest', )
        # output: (64, 1000, 1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (64, 1000, 1)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv3b = nn.Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (64, 1000, 1)
        self.batchnorm3b = nn.BatchNorm2d(64)

        # reduce number of conv channels

        self.conv3c = nn.Conv2d(64, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (32, 1000, 1)
        self.batchnorm3c = nn.BatchNorm2d(32)

        self.conv3d = nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (32, 1000, 1)
        self.batchnorm3d = nn.BatchNorm2d(32)

        self.conv3e = nn.Conv2d(32, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (16, 1000, 1)
        self.batchnorm3e = nn.BatchNorm2d(16)

        self.conv3f = nn.Conv2d(16, 16, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (16, 1000, 1)
        self.batchnorm3f = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 1, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), )
        # output: (1, 1000, 1)

        # tanh activation function for generator output
        self.tanh4 = nn.Tanh()
        # output: (1, 1000, 1)

    def forward(self, x, vc):
        # print('----------------- Generator -------------')
        # fully-connected + reshape
        # print('shape x init:', x.shape)

        x = self.fc00(x)
        x = self.batchnorm00(x)
        # print('fc00 shape:', x.shape)

        # expand dimension
        x = torch.unsqueeze(x, 3)
        # print('torch.unsqueeze shape:', x.shape)


        # ----------- Conditional variables  ------------
        #  apply embeddings for the conditional variables
        v1 = self.embed1(v1)
        v2 = self.embed2(v2)
        v3 = self.embed3(v3)

        # # reshape
        v1 = v1.view(-1, 1, 1000, 1)
        v2 = v2.view(-1, 1, 1000, 1)
        v3 = v3.view(-1, 1, 1000, 1)

        # concatenate conditional variables to input
        x = torch.cat([x, v1, v2, v3], 1)
        # print('torch.cat([x, vc], 2) shape: ', x.shape)
        # ------------------------------------------------

        x = self.conv0(x)
        x = self.batchnorm0(x)
        x = F.relu(x)
        # print('conv0 shape:', x.shape)

        x = self.conv0b(x)
        x = self.batchnorm0b(x)
        x = F.relu(x)
        # print('conv0b shape:', x.shape)

        x = self.conv0c(x)
        x = self.batchnorm0c(x)
        x = F.relu(x)
        # print('conv0c shape:', x.shape)

        x = torch.squeeze(x, 3)
        # print('torch.squeeze shape:', x.shape)

        x = self.fc01(x)
        x = self.batchnorm01(x)
        x = F.relu(x)
        # print('fc01 shape:', x.shape)

        x = x.view(-1, 2, 125, 1)
        # print('x.view() shape:', x.shape)

        # resize_nearest_neighbor
        x = self.resizenn1(x)
        # print('resizenn1 shape:', x.shape)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        # print('conv1 shape:', x.shape)

        x = self.conv1b(x)
        x = self.batchnorm1b(x)
        x = F.relu(x)
        # print('conv1b shape:', x.shape)

        # resize_nearest_neighbor
        x = self.resizenn2(x)
        # print('resizenn2 shape:', x.shape)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        # print('conv2 shape:', x.shape)

        x = self.conv2b(x)
        x = self.batchnorm2b(x)
        x = F.relu(x)
        # print('cconv2b shape:', x.shape)

        # resize_nearest_neighbor
        x = self.resizenn3(x)
        # print('resizenn3 shape:', x.shape)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        # print('conv3 shape:', x.shape)

        x = self.conv3b(x)
        x = self.batchnorm3b(x)
        x = F.relu(x)
        # print('cconv3b shape:', x.shape)

        # final set of convolutional layers to obtained
        # the desired output shape
        x = self.conv3c(x)
        x = self.batchnorm3c(x)
        x = F.relu(x)
        # print('conv3c shape:', x.shape)

        x = self.conv3d(x)
        x = self.batchnorm3d(x)
        x = F.relu(x)
        # print('conv3d shape:', x.shape)

        x = self.conv3e(x)
        x = self.batchnorm3e(x)
        x = F.relu(x)
        # print('conv3e shape:', x.shape)

        x = self.conv3f(x)
        x = self.batchnorm3f(x)
        x = F.relu(x)
        # print('conv3f shape:', x.shape)

        #----

        x = self.conv4(x)
        # print('conv4 shape:', x.shape)

        # final outpsut
        x = self.tanh4(x)
        # print('tanh4 shape:', x.shape)

        # reshape for plotting purposes
        out = x.squeeze(1)

        return out
