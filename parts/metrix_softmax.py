import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

import math

from torch.nn.utils.weight_norm import WeightNorm


class MetrixSoftmax(nn.Module):
    def __init__(self, in_features, out_features, temperature):
        super(MetrixSoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

        WeightNorm.apply(self.weight, 'weight', dim=0)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        input = input.unsqueeze(1).repeat(1, self.out_features, 1)
        metrix = self.weight.unsqueeze(0).repeat(input.size()[0], 1, 1)
        metrix = torch.norm((metrix - input), p=2, dim=2)
        metrix = F.softmax(-self.temperature * metrix, dim=1)
        return metrix

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class GTrainer(nn.Module):
    def __init__(self, in_features, out_features, temperature):
        super(GTrainer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.g = Parameter(torch.Tensor(in_features, in_features))
        self.classifier = MetrixSoftmax(in_features, out_features, temperature)
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.eye_(self.g)

    def forward(self, input):
        out = input.matmul(self.g)
        out = self.classifier(out)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )