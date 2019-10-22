# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.meta_template import MetaTemplate


class SimpleNetTrain(MetaTemplate):
    def __init__(self, model_func, num_class, num_way, num_shot, num_query, train_param, test_param):
        super(SimpleNetTrain, self).__init__(model_func, num_way, num_shot)
        if train_param['shake']:
            self.feature = model_func(shake_config=train_param['shake_config'])
        else:
            self.feature = model_func()
        self.num_class = num_class
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.train_loss_type = train_param['loss_type']
        self.train_temperature = train_param['temperature']
        self.train_margin = train_param['margin']
        self.train_lr = train_param['lr']
        self.shake = train_param['shake']
        self.shake_config = train_param['shake_config']
        self.test_loss_type = test_param['loss_type']
        self.test_temperature = test_param['temperature']
        self.test_margin = test_param['margin']
        self.test_lr = test_param['lr']
        self.final_feat_dim = self.feature.final_feat_dim

        self.classifier = nn.Linear(self.feat_dim, num_class)
        self.classifier.bias.data.fill_(0)

        self.train_loss_softmax = nn.CrossEntropyLoss().cuda()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_proto = z_support.contiguous().view(self.n_way, self.n_support, self.feat_dim).mean(1).squeeze()
        z_query = z_query.contiguous().view(self.n_way * self.n_query, self.feat_dim)

        z_proto_linear = self.classifier.forward(z_proto)
        z_query_linear = self.classifier.forward(z_query)
        z_proto_ext = z_proto_linear.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1).view(-1, self.num_class)
        z_query_ext = z_query_linear.unsqueeze(0).repeat(self.n_way, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1).contiguous().view(-1, self.num_class)

        z_proto_norm = torch.norm(z_proto_ext, p=2, dim=1).unsqueeze(1).expand_as(z_proto_ext)
        z_proto_normalized = z_proto_ext.div(z_proto_norm + 0.00001)
        z_query_norm = torch.norm(z_query_ext, p=2, dim=1).unsqueeze(1).expand_as(z_query_ext)
        z_query_normalized = z_query_ext.div(z_query_norm + 0.00001)
        distance = z_proto_normalized.unsqueeze(1).bmm(z_query_normalized.unsqueeze(2)).view(-1, self.n_way) * self.ratio

        return distance, z_support

    def set_forward_loss(self, x):
        distance, feature = self.set_forward(x)
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y = Variable(y.cuda())
        return self.loss_softmax(distance, y)

    def test_loop(self, cl_data_file):
        class_list = cl_data_file.keys()
        select_class = random.sample(class_list, self.num_way)

        z_all = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(self.num_shot + self.num_query)])
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

        if self.loss_type == 'softmax':
            linear_clf = nn.Linear(self.feat_dim, self.n_way)
        elif self.loss_type == 'dist':
            linear_clf = backbone.distLinear(self.feat_dim, self.n_way)
        linear_clf = linear_clf.cuda()

        set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                        weight_decay=0.001)

        loss_function = nn.CrossEntropyLoss()
        loss_function = loss_function.cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                set_optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch)
                if self.centered:
                    loss = loss_function(scores, y_batch) + self.alpha * loss_center(z_batch, y_batch)
                else:
                    loss = loss_function(scores, y_batch)
                loss.backward()
                if self.centered:
                    for param in loss_center.parameters():
                        param.grad.data *= (1. / self.alpha)
                set_optimizer.step()
        scores = linear_clf(z_query)
        return scores
