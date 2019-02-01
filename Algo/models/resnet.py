import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np



class ResNet(nn.Module):
    def __init__(self, input_shape, repetitions,  out_nfilter, **kwargs):
        super(ResNet, self).__init__()
        first_nfilter = kwargs.get('first_nfilter', int(input_shape[0]*1.5))
        self.first_layer = nn.Conv1d(input_shape[0], first_nfilter, 7, padding = 3)
        self.ResBlocks = nn.Sequential()
        length = input_shape[1]
        nfilter = first_nfilter
        for i, rep in enumerate(repetitions):
            for j in range(rep):
                if j == rep -1:
                    self.ResBlocks.add_module('rep'+str(i)+'block'+str(j), ResBlocks(nfilter, nfilter*2))
                    nfilter *= 2
                    length = int(length/2) + 1
                else:
                    self.ResBlocks.add_module('rep'+str(i)+'block'+str(j), ResnetBlock(nfilter,nfilter))
            if i == repetitions - 1:
                self.ResnetBlocks.add_module('rep'+str(i)+'avgpool', nn.AvgPool1d(length, stride = 1, padding = 0))
        	else:
                self.ResnetBlocks.add_module('rep'+str(i)+'avgpool', nn.AvgPool1d(3, stride = 2, padding =1))
        self.fc_out = nn.Linear(nfilter, out_nfilter)

    def forward(self, x):
        out = self.first_layer(x)
        out = self.ResBlocks(out)
        out = self.avgpool(out)
        out = self.fc_out(actvn(out))
        return out

        

  


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(ResnetBlock, self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv1d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv1d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv1d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

