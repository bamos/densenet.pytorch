#!/usr/bin/env python3

import torch

import torch.nn as nn
import torch.legacy as legacy
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import torch.utils.serialization

import sys
import math

import densenet

torch.manual_seed(0)

net = densenet.DenseNet(growthRate=12, depth=40, reduction=0.5,
                        bottleneck=True, nClasses=10)

net_th = torch.utils.serialization.load_lua('dn.t7')

convs = []
fcs = []

def printnorm_f(self, input, output):
    print('{} norm: {}'.format(self.__class__.__name__, output.data.norm()))

# def printnorm_back(self, grad_input, grad_output):
    # import IPython, sys; IPython.embed(); sys.exit(-1)
    # print('{} grad_out norm: {}'.format(self.__class__.__name__, self.weight.grad.data.norm()))

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        convs.append(m.weight.data)
        # m.register_forward_hook(printnorm_f)
        # m.register_backward_hook(printnorm_back)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fcs.append(m.weight.data)
        m.bias.data.zero_()


global convI, fcI
convI = fcI = 0

def init(mods):
    global convI, fcI
    for m in mods:
        if isinstance(m, legacy.nn.SpatialConvolution):
            m.weight = convs[convI]
            convI += 1
        elif isinstance(m, legacy.nn.Linear):
            m.weight = fcs[fcI]
            fcI += 1
        elif isinstance(m, legacy.nn.Concat) or \
             isinstance(m, legacy.nn.Sequential):
            init(m.modules)

init(net_th.modules)

print(convI, fcI, len(convs), len(fcs))
assert(convI == len(convs))
assert(fcI == len(fcs))

x = torch.randn(7, 3, 32, 32)
x_v = Variable(x)

pyOut = net(x_v)
print('out: {}'.format(pyOut))

print('===')
luaOut_1 = net_th.forward(x)
lsm = legacy.nn.LogSoftMax()
luaOut = lsm.forward(luaOut_1)

def printM(mods):
    for m in mods:
        if isinstance(m, legacy.nn.SpatialConvolution):
            print('Conv2d norm: {}'.format(torch.norm(m.output)))
        elif isinstance(m, legacy.nn.Linear):
            pass
        elif isinstance(m, legacy.nn.Concat) or \
             isinstance(m, legacy.nn.Sequential):
            printM(m.modules)

# printM(net_th.modules)
print('out: {}'.format(luaOut))

print('===')

print('PyTorch weight gradients:')

target = torch.LongTensor([3,2,1,3,2,3,4])
target_v = Variable(target)
loss = F.nll_loss(pyOut, target_v)
loss.backward()

l = []
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        l.append(m.weight.grad.data.norm())
    elif isinstance(m, nn.Linear):
        l.append(m.weight.grad.data.norm())
print(l)

print('===')

print('LuaTorch weight gradients:')

criterion = legacy.nn.ClassNLLCriterion()
loss_lua = criterion.forward(luaOut, target)
t1 = criterion.backward(luaOut, target)
t2 = lsm.backward(luaOut_1, t1)
t3 = net_th.backward(x, t2)

l = []
def getM(mods):
    for m in mods:
        if isinstance(m, legacy.nn.SpatialConvolution):
            m.gradWeight[m.gradWeight.ne(m.gradWeight)] = 0
            l.append(torch.norm(m.gradWeight))
        elif isinstance(m, legacy.nn.Linear):
            l.append(torch.norm(m.gradWeight))
        elif isinstance(m, legacy.nn.Concat) or \
             isinstance(m, legacy.nn.Sequential):
            getM(m.modules)

getM(net_th.modules)
print(l)
