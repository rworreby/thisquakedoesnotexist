#!/usr/bin/env python

"""Tests for `gan1d` package."""

import pytest
import torch

from thisquakedoesnotexist.condensed_code import gan1d
from thisquakedoesnotexist.utils import random_fields
from thisquakedoesnotexist.utils.data import WaveDatasetDist, DataLoader


@pytest.fixture
def constants():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    pytest.device = torch.device("cpu")
    pytest.noise_dim = 100
    pytest.batchsize = 40
    pytest.dist_bins = 20


@pytest.fixture
def get_data_set(constants):
    dataset = WaveDatasetDist(
        data_file='thisquakedoesnotexist/data/japan/waveforms.npy', 
        attr_file='thisquakedoesnotexist/data/japan/attributes.csv', 
        ndist_bins=pytest.dist_bins, 
        dt=0.04
        )

    return dataset
    

def test_train_loader(get_data_set):
    # ----- test Loader-----
    x_r, y_r = next(iter(get_data_set))
    print('shapes x_r, y_r: ', x_r.shape, y_r.shape)
    Nb = x_r.size(0)

    x_r = x_r.squeeze()


def test_random_field(constants):
    # Test random field dim=1
    # Parameter for gaussian random field

    grf = random_fields.rand_noise(1, pytest.noise_dim, device=pytest.device)
    z = grf.sample(20)
    # print("z shape: ", z.shape)
    z = z.squeeze()
    z = z.detach().cpu()
    
    assert z.size(0) == 20, f'Checking dimension of random field.'


def test_discriminator(constants, get_data_set):
    """Sample pytest test function with the pytest fixture as an argument."""
    # ----- test Discriminator-----
    data = DataLoader(get_data_set, batch_size=pytest.batchsize, shuffle=True)
    x_r, vc_r = next(iter(data))
    # get discriminator
    Nb = x_r.size(0)
    x_r = x_r.to(pytest.device)
    vc_r = vc_r.to(pytest.device)
    print('shapes x_r: ', x_r.shape)
    D = gan1d.Discriminator().to(pytest.device)
    nn_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print("Number discriminator parameters: ", nn_params)
    print(D)
    x_d = D(x_r, vc_r)
    print("D(x_r, vc_r) shape: ", x_d.shape)

"""
def test_gradient_penalty(constants, get_data_set):
    # # ----- test gradient penalty -----
    data = DataLoader(get_data_set, batch_size=pytest.batchsize, shuffle=True)
    x_r, vc_r = next(iter(data))
    
    print("x_r: ", x_r.shape)
    print("vc_r: ", vc_r.shape)
    
    # get discriminator
    Nb = x_r.size(0)
    x_r = x_r.to(pytest.device)
    vc_r = vc_r.to(pytest.device)

    G = gan1d.Generator(z_size=pytest.noise_dim).to(pytest.device)
    # print(G)
    nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print("Number Generator parameters: ", nn_params)
    grf = random_fields.rand_noise(1, pytest.noise_dim, device=pytest.device)
    z = grf.sample(pytest.dist_bins)
    print("z shape: ", z.shape)
    # get random batch of conditional variables
    data = get_data_set
    vc_g = data.get_rand_cond_v(pytest.dist_bins)

    x_g = G(z, vc_g)
    print('G(z) shape: ', x_g.shape)
    x_g = x_g.squeeze(1)
    # make plot
    x_g = x_g.squeeze()
    x_g = x_g.detach().cpu()


    # random alpha
    alpha = torch.rand(Nb, 1, 1, device=pytest.device)
    # Get random interpolation between real and fake samples
    # Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
    print("x_r: ", x_r.shape)
    print("x_g: ", x_g.shape)
    
    print("alpha * x_r: ", (alpha * x_r).shape)
    print("1- alpha * x_g: ", ((1 - alpha) * x_g).shape)
    print("both added: ", ((alpha * x_r) + ((1 - alpha) * x_g)).shape)
    Xp = (alpha * x_r + ((1 - alpha) * x_g)).requires_grad_(True)
    # apply dicriminator
    D = gan1d.Discriminator().to(pytest.device)
    D_xp = D(Xp, vc_r)
    #Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
    Xout = gan1d.Variable(torch.ones(Nb, 1, device=pytest.device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    grads = torch.autograd.grad(
        outputs=D_xp,
        inputs=Xp,
        grad_outputs=Xout,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    regularizer = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    print('grads.shape', grads.shape)
    print('regularizer.shape', regularizer.shape)
"""
    
def test_generator(constants, get_data_set):
    #----- Test generator -----------
    NPLT_MAX = pytest.dist_bins
    G = gan1d.Generator(z_size=pytest.noise_dim).to(pytest.device)
    print(G)
    nn_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print("Number Generator parameters: ", nn_params)
    grf = random_fields.rand_noise(1, pytest.noise_dim, device=pytest.device)
    z = grf.sample(NPLT_MAX)
    print("z shape: ", z.shape)
    # get random batch of conditional variables
    data = get_data_set
    vc_g = data.get_rand_cond_v(NPLT_MAX)
    vc_g = vc_g.to(pytest.device)
    x_g = G(z, vc_g)
    print('G(z) shape: ', x_g.shape)
    x_g = x_g.squeeze(1)
    # make plot
    x_g = x_g.squeeze()
    x_g = x_g.detach().cpu()
    
    data = DataLoader(get_data_set, batch_size=pytest.batchsize, shuffle=True)

    x_r, vc_r = next(iter(data))
    # get discriminator
    Nb = x_r.size(0)
    x_r = x_r.to(pytest.device)
    vc_r = vc_r.to(pytest.device)

    z = grf.sample(Nb)
    x_g = G(z,vc_r)

    