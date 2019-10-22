import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from parts.center_loss import CenterLoss


class BaselineFinetune(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, temperature=4, loss_type="softmax", margin=1, centered=False):
        super(BaselineFinetune, self).__init__(model_func, n_way, n_support, centered, alpha=0.01)
        self.loss_type = loss_type
        self.temperature = temperature
        self.margin = margin

    def set_forward(self, iter, x, writer, is_feature=True):
        return self.set_forward_adaptation(iter, x, writer, is_feature)  # Baseline always do adaptation

    def set_forward_adaptation(self, iter, x, writer, is_feature=True):
        assert is_feature == True, 'Baseline only support testing with feature'
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
            linear_clf.bias.data.fill_(0)
        elif self.loss_type == 'dist':  # Baseline ++
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way, self.temperature)
        elif self.loss_type == 'lsoftmax':
            linear_clf = LSoftmaxLinear(self.feat_dim, self.n_way, self.margin)
            linear_clf.reset_parameters()
        elif self.loss_type == 'ldist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way, self.temperature, self.margin, True)

        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()
        if self.centered:
            loss_center = CenterLoss(num_classes=self.n_way, feat_dim=self.feature.final_feat_dim)
            loss_center = loss_center.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            linear_clf.train()
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                if self.loss_type[0] == 'l':
                    scores = linear_clf(z_batch, y_batch)
                else:
                    scores = linear_clf(z_batch)
                if self.centered:
                    loss = loss_function(scores, y_batch) + self.alpha * loss_center(z_batch, y_batch)
                else:
                    loss = loss_function(scores, y_batch)
                writer.add_scalar('loss/test_loss', loss.item(), ((iter * 100 + epoch) * support_size + i) // batch_size)
                loss.backward()
                if self.centered:
                    for param in loss_center.parameters():
                        param.grad.data *= (1. / self.alpha)
                set_optimizer.step()

        linear_clf.eval()
        scores = linear_clf(z_query, None)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.n_way), self.n_query)
        acc = np.mean(pred == y) * 100
        writer.add_scalar('acc/test_acc', acc.item(), iter)
        return scores

    def set_forward_loss(self, x):
        raise ValueError('Baseline predict on pretrained feature and do not support finetune backbone')
