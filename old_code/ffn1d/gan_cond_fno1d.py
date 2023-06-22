import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# -------------------------- Fourier Layer ----------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


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

class Generator(nn.Module):
    def __init__(self, modes, width, padding=9):
        super(Generator, self).__init__()

        """
        Generator Model
        4 Integral operator layers: u' = (W + K)(u)
            input:  1D GRF (f(t), t), x E [0,1] 
                    where f is and arbitrary function
            output: 1D Function: u(t)
        BASIC STEPS
        1. Lift the input to the desire channel dimension by self.fc0
        2. 4 layers of the integral operators u' = (W + K)(u).
                W defined by self.w; K defined by self.conv
        3. Project from the channel space to the output space by self.fc1 and self.fc2
        """

        #  get embeddings for the conditional variables
        self.embed1 = embed(1, 1000)


        self.modes1 = modes
        self.width = width
        # pad the domain if input is non-periodic
        self.padding = 2
        # input channel is 4: (f(t), t)
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, vc):
        # print("------------ Generator FNO 1d -------------")
        # print("shape x init:", x.shape)
        grid = self.get_grid(x.shape, x.device)
        # print("self.get_grid shape: ", grid.shape)
        # get embedding for conditional variables
        vc = self.embed1(vc)
        vc = vc.unsqueeze(2)
        # print("vc shape: ", vc.shape)

        x = torch.cat((x, vc, grid), dim=-1)
        # print("torch.cat((x, vc, grid), dim=-1) shape:", x.shape)
        x = self.fc0(x)
        # print("fc0 shape:", x.shape)
        # permute so it is easy to apply convolutions
        x = x.permute(0, 2, 1)
        # print("x.permute(0, 2, 1) shape: ", x.shape)
        # # apply padding last two input dimensions
        # # padding_left,padding_right, padding_top, padding_bottom
        # x = F.pad(x, [0,self.padding, 0,self.padding])
        # # print("F.pad shape: ", x.shape)

        # ----- Fourier layers ----------
        x1 = self.conv0(x)
        # print("Spec conv0 shape: ", x1.shape)
        x2 = self.w0(x)
        # print("Conv 2D w0 shape: ", x2.shape)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        # print("Spec conv1 shape: ", x1.shape)
        x2 = self.w1(x)
        # print("Conv 2D w1 shape: ", x2.shape)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        # print("Spec conv2 shape: ", x1.shape)
        x2 = self.w2(x)
        # print("Conv 2D w2 shape: ", x2.shape)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        # print("Spec conv3 shape: ", x1.shape)
        x2 = self.w3(x)
        # print("Spec w3 shape: ", x1.shape)
        x = x1 + x2

        # # remove the padding
        # x = x[..., :-self.padding, :-self.padding]
        # # print("x[..., :-self.padding, :-self.padding] shape: ", x.shape)
        x = x.permute(0, 2, 1)
        # print("x.permute(0, 2, 1) shape: ", x.shape)

        # Project back to physical space
        x = self.fc1(x)
        x = F.gelu(x)
        # print("fc1 + gelu shape: ", x.shape)

        x = self.fc2(x)
        # print("fc2 shape: ", x.shape)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class Discriminator(nn.Module):
    def __init__(self, modes, width, lt=1000, padding=9):
        super(Discriminator, self).__init__()

        """
        Discriminator Model
        4 Integral operator layers: u' = (W + K)(u)
            input: 1D u(t) (u(t), t), t E [0,1]
            output:  1D Function
            
        2 Output linear layers
        """
        
        #  get embeddings for the conditional variables
        self.embed1 = embed(1, 1000)

        self.modes1 = modes
        self.width = width
        # pad the domain if input is non-periodic
        self.padding = 2
        self.lt = lt
        # input channel is 2: (f(t), t)
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        # final linear set of operations mapping to R
        self.fcd1 = nn.Linear(self.lt, 512)
        self.fcd2 = nn.Linear(512, 1)

    def forward(self, x, vc):
        # print("------------ Discriminator FNO 1d -------------")
        # print("shape x init:", x.shape)
        grid = self.get_grid(x.shape, x.device)
        # print("self.get_grid shape: ", grid.shape)
        # get emabedding for conditional variabless
        vc = self.embed1(vc)
        vc = vc.unsqueeze(2)
        # print("embed1 shape:", vc.shape)

        x = torch.cat((x, vc, grid), dim=-1)
        # print("torch.cat((x, vc, grid), dim=-1) shape:", x.shape)
        x = self.fc0(x)
        # print("fc0 shape:", x.shape)
        # permute so it is easy to apply convolutions
        x = x.permute(0, 2, 1)
        # print("x.permute(0, 2, 1) shape: ", x.shape)
        # # apply padding last two input dimensions
        # # padding_left,padding_right, padding_top, padding_bottom
        # x = F.pad(x, [0,self.padding, 0,self.padding])
        # # print("F.pad shape: ", x.shape)

        # ----- Fourier layers ----------
        x1 = self.conv0(x)
        # print("Spec conv0 shape: ", x1.shape)
        x2 = self.w0(x)
        # print("Conv 2D w0 shape: ", x2.shape)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        # print("Spec conv1 shape: ", x1.shape)
        x2 = self.w1(x)
        # print("Conv 2D w1 shape: ", x2.shape)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        # print("Spec conv2 shape: ", x1.shape)
        x2 = self.w2(x)
        # print("Conv 2D w2 shape: ", x2.shape)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        # print("Spec conv3 shape: ", x1.shape)
        x2 = self.w3(x)
        # print("Spec w3 shape: ", x1.shape)
        x = x1 + x2

        # # remove the padding
        # x = x[..., :-self.padding, :-self.padding]
        # # print("x[..., :-self.padding, :-self.padding] shape: ", x.shape)
        x = x.permute(0, 2, 1)
        # print("x.permute(0, 2, 1) shape: ", x.shape)

        # Project back to physical space
        x = self.fc1(x)
        x = F.gelu(x)
        # print("fc1 + gelu shape: ", x.shape)

        x = self.fc2(x)
        # print("fc2 shape: ", x.shape)

        # ----- end Fourier Layers: outputs a function f(t) ------
        # flatten function f(t)
        x = x.view(-1, self.lt)
        # print("x.view(-1, self.lt) shape: ", x.shape)
        x = self.fcd1(x)
        x = F.gelu(x)
        # print("fcd1(x) shape: ", x.shape)

        # final linear operation
        x = self.fcd2(x)
        # print("fcd2(x) shape: ", x.shape)

        return x


    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)



# class Discriminator(nn.Module):
#     def __init__(self, modes1, modes2,  width, lx=64, padding=5):
#         super(Discriminator, self).__init__()
#
#         """
#         Discriminator Model
#         4 Integral operator layers: u' = (W + K)(u)
#             input: 2D GRF (grf(x, y), x, y), x E [0,1], y E [0,1]
#             output:  2D Function
#
#         2 Output linear layers
#         """
#
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.width = width
#         # pad the domain if input is non-periodic
#         self.padding = padding
#         self.lx = lx
#         self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
#
#         self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.w0 = nn.Conv2d(self.width, self.width, 1)
#         self.w1 = nn.Conv2d(self.width, self.width, 1)
#         self.w2 = nn.Conv2d(self.width, self.width, 1)
#         self.w3 = nn.Conv2d(self.width, self.width, 1)
#
#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)
#
#         # final linear set of operations mapping to R
#         self.fcd1 = nn.Linear(self.lx*self.lx, 512)
#         self.fcd2 = nn.Linear(512, 1)
#
#     def forward(self, x):
#         # print("------------ Discriminator FNO 2d -------------")
#         # print("shape x init:", x.shape)
#         grid = self.get_grid(x.shape, x.device)
#         # print("self.get_grid shape: ", grid.shape)
#         x = torch.cat((x, grid), dim=-1)
#         # print("torch.cat((x, grid), dim=-1) shape:", x.shape)
#         x = self.fc0(x)
#         # print("fc0 shape:", x.shape)
#         # permute so it is easy to apply convolutions
#         x = x.permute(0, 3, 1, 2)
#         # print("x.permute shape: ", x.shape)
#         # apply padding last two input dimensions
#         # padding_left,padding_right, padding_top, padding_bottom
#         x = F.pad(x, [0,self.padding, 0,self.padding])
#         # print("F.pad shape: ", x.shape)
#
#         # ----- Fourier layers ----------
#         x1 = self.conv0(x)
#         # print("Spec conv0 shape: ", x1.shape)
#         x2 = self.w0(x)
#         # print("Conv 2D w0 shape: ", x2.shape)
#         x = x1 + x2
#         x = F.gelu(x)
#
#         x1 = self.conv1(x)
#         # print("Spec conv1 shape: ", x1.shape)
#         x2 = self.w1(x)
#         # print("Conv 2D w1 shape: ", x2.shape)
#         x = x1 + x2
#         x = F.gelu(x)
#
#         x1 = self.conv2(x)
#         # print("Spec conv2 shape: ", x1.shape)
#         x2 = self.w2(x)
#         # print("Conv 2D w2 shape: ", x2.shape)
#         x = x1 + x2
#         x = F.gelu(x)
#
#         x1 = self.conv3(x)
#         # print("Spec conv3 shape: ", x1.shape)
#         x2 = self.w3(x)
#         # print("Spec w3 shape: ", x1.shape)
#         x = x1 + x2
#
#         # remove the padding
#         x = x[..., :-self.padding, :-self.padding]
#         # print("x[..., :-self.padding, :-self.padding] shape: ", x.shape)
#         x = x.permute(0, 2, 3, 1)
#         # print("x.permute(0, 2, 3, 1) shape: ", x.shape)
#
#         # Project back to physical space
#         x = self.fc1(x)
#         x = F.gelu(x)
#
#         # print("fc1 + gelu shape: ", x.shape)
#
#         x = self.fc2(x)
#         # print("fc2 shape: ", x.shape)
#
#         # ----- end Fourier Layers: outputs a function f(x,y) ------
#         # flatten function f(x,y)
#         x = x.view(-1, self.lx*self.lx)
#         # print("x.view(-1, self.lx*self.lx) shape: ", x.shape)
#         x = self.fcd1(x)
#         x = F.gelu(x)
#         # print("fcd1(x) shape: ", x.shape)
#
#         # final linear operation
#         x = self.fcd2(x)
#         # print("fcd2(x) shape: ", x.shape)
#
#         return x
#
#     def get_grid(self, shape, device):
#         batchsize, size_x, size_y = shape[0], shape[1], shape[2]
#         gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
#         gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
#         gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
#         gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
#         return torch.cat((gridx, gridy), dim=-1).to(device)

