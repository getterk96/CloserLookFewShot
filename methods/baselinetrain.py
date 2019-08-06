import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from center_loss import CenterLoss


class BaselineTrain(nn.Module):
    def __init__(self, model_func, num_class, alpha=0.01, loss_type='softmax', centered=True):
        super(BaselineTrain, self).__init__()
        self.feature = model_func()
        self.centered = centered
        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.feature.final_feat_dim, num_class)
        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_softmax = nn.CrossEntropyLoss()
        self.alpha = alpha
        if centered:
            self.loss_center = CenterLoss(num_classes=num_class, feat_dim=self.feature.final_feat_dim)

    def forward(self, x):
        x = Variable(x.cuda())
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores, out

    def forward_loss(self, x, y):
        scores, feature = self.forward(x)
        y = Variable(y.cuda())
        if self.centered:
            return self.loss_softmax(scores, y) + self.alpha * self.loss_center(feature, y)
        else:
            return self.loss_softmax(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            if self.centered:
                for param in self.loss_center.parameters():
                    param.grad.data *= (1. / self.alpha)
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader),
                                                                        avg_loss / float(i + 1)))

    def test_loop(self, val_loader):
        return -1  # no validation, just save model during iteration
