# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate


class SimpleNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, loss_type='softmax'):
        super(SimpleNet, self).__init__(model_func, n_way, n_support)
        self.loss_type = loss_type
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()
        z_proto = z_support.view(self.n_way, self.n_support, self.feat_dim).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, self.feat_dim)

        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1).view(-1, self.feat_dim)
        z_proto_norm = torch.norm(z_proto_ext, p=2, dim=1).unsqueeze(1).expand_as(z_proto_ext)
        z_proto_normalized = z_proto_ext.div(z_proto_norm + 0.00001)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1).contiguous().view(-1, self.feat_dim)
        z_query_norm = torch.norm(z_query_ext, p=2, dim=1).unsqueeze(1).expand_as(z_query_ext)
        z_query_normalized = z_query_ext.div(z_query_norm + 0.00001)
        distance = z_proto_normalized.unsqueeze(1).bmm(z_query_normalized.unsqueeze(2)).view(-1, self.n_way)

        return distance

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        distance = self.set_forward(x) * 2

        return self.loss_fn(distance, y_query)
