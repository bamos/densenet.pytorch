#!/usr/bin/env python3

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import numdifftools as nd

torch.manual_seed(0)
cuda = True

class FcCat(nn.Module):
    def __init__(self, nIn, nOut):
        super(FcCat, self).__init__()
        self.fc = nn.Linear(nIn, nOut, bias=False)

    def forward(self, x):
        out = torch.cat((x, self.fc(x)), 1)
        return out

class Net(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden2):
        super(Net, self).__init__()
        self.l1 = FcCat(nFeatures, nHidden1)
        self.l2 = FcCat(nFeatures+nHidden1, nHidden2)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return out


nBatch, nFeatures, nHidden1, nHidden2 = 2, 2, 3, 4
x = Variable(torch.randn(nBatch, nFeatures))
expected = Variable(torch.randn(nBatch, nFeatures+nHidden1+nHidden2))
net = Net(nFeatures, nHidden1, nHidden2)
criterion = torch.nn.loss.MSELoss()

if cuda:
    x = x.cuda()
    net = net.cuda()
    expected = expected.cuda()

predicted = net(x)
loss = criterion(predicted, expected)
loss.backward()

W1, W2 = list(net.parameters())

x_np = x.data.cpu().numpy()
exp_np = expected.data.cpu().numpy()

def f_loss(W12_flat):
    """A function that has all of the parameters flattened
    as input for numdifftools."""
    W1, W2 = unpack(W12_flat)
    out1 = x_np.dot(W1.T)
    out1 = np.concatenate((x_np, out1), axis=1)
    out2 = out1.dot(W2.T)
    out2 = np.concatenate((out1, out2), axis=1)

    mse_batch = np.mean(np.square(out2-exp_np), axis=1)
    mse = np.mean(mse_batch)
    return mse

def unpack(W12_flat):
    W1, W2 = np.split(W12_flat, [nFeatures*nHidden1])
    W1 = W1.reshape(nHidden1, nFeatures)
    W2 = W2.reshape(nHidden2, nFeatures+nHidden1)
    return W1, W2

W12_flat = torch.cat((W1.data.view(-1), W2.data.view(-1))).cpu().numpy()
print('The PyTorch loss is {:.3f}. f_loss for numeric diff is {:.2f}.'.format(
    loss.data[0], f_loss(W12_flat)))

assert(np.abs(loss.data[0] - f_loss(W12_flat)) < 1e-4)

g = nd.Gradient(f_loss)
dW12_flat = g(W12_flat)
dW1, dW2 = unpack(dW12_flat)

def printGrads(tag, W, dW):
    print('\n' + '-'*40 + '''
The gradient w.r.t. {0} from PyTorch is:

{1}

The gradient w.r.t. {0} from numeric differentiation is:

{2}'''.format(tag, W.grad, dW))


printGrads('W1', W1, dW1)
printGrads('W2', W2, dW2)
