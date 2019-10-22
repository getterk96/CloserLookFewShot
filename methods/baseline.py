import backbone
from parts.large_margin_softmax import LSoftmaxLinear

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from parts.initializer import *
from parts.metrix_softmax import MetrixSoftmax, GTrainer


class Baseline(nn.Module):
    def __init__(self, model_func, entropy, train_param, test_param):
        super(Baseline, self).__init__()
        if train_param['shake']:
            self.feature = model_func(shake_config=train_param['shake_config'])
        else:
            self.feature = model_func()
        self.num_classes = train_param['num_classes']
        self.train_num_way = train_param['num_way']
        self.train_num_shot = train_param['num_shot']
        self.train_num_query = train_param['num_query']
        self.test_num_way = test_param['num_way']
        self.test_num_shot = test_param['num_shot']
        self.test_num_query = test_param['num_query']
        self.episodic = train_param['episodic']
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
        self.entropy = entropy

        if self.train_loss_type == 'softmax':
            self.train_classifier = nn.Linear(self.final_feat_dim, self.num_classes)
            self.train_classifier.bias.data.fill_(0)
            self.train_classifier.apply(init_weights)
        elif self.train_loss_type == 'dist':  # Baseline ++
            self.train_classifier = backbone.distLinear(self.final_feat_dim, self.num_classes, self.train_temperature,
                                                        None)
        elif self.train_loss_type == 'lsoftmax':
            self.train_classifier = LSoftmaxLinear(self.final_feat_dim, self.num_classes, self.train_margin)
            self.train_classifier.reset_parameters()
        elif self.train_loss_type == 'ldist':
            self.train_classifier = backbone.distLinear(self.final_feat_dim, self.num_classes, self.train_temperature,
                                                        self.train_margin)
        elif self.train_loss_type == 'msoftmax':
            self.train_classifier = MetrixSoftmax(self.final_feat_dim, self.num_classes, self.train_temperature)

        if self.episodic and self.entropy:
            self.entropy_layer = nn.Linear(self.final_feat_dim, self.num_classes)
            self.entropy_layer.apply(init_weights)

        self.train_loss_softmax = nn.CrossEntropyLoss()

    def forward_loss(self, x, y):
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        out = self.feature(x)
        if self.train_loss_type in ['softmax', 'msoftmax']:
            scores = self.train_classifier(out)
        else:
            scores = self.train_classifier(out, y)

        return self.train_loss_softmax(scores, y)

    def set_forward(self, x):
        z_support, z_query = self.parse_feature(x, False)

        if self.entropy:
            entropy_support = z_support.contiguous().view(self.train_num_way * self.train_num_shot, z_support.size()[2])
            entropy_output = self.entropy_layer(entropy_support)
            entropy = torch.sum(-F.softmax(entropy_output, dim=1) * F.log_softmax(entropy_output, dim=1),
                                dim=(1,)).view(self.train_num_way, self.train_num_shot)
            entropy_norm = torch.mean(entropy, dim=(1,)).unsqueeze(1).expand_as(entropy)
            entropy_normalized = entropy.div(entropy_norm + 0.00001).unsqueeze(2).expand_as(z_support)
            z_support = z_support * entropy_normalized

        z_proto = z_support.contiguous().view(self.train_num_way, self.train_num_shot, self.final_feat_dim).mean(
            1).squeeze()
        z_query = z_query.contiguous().view(self.train_num_way * self.train_num_query, self.final_feat_dim)

        z_proto_ext = z_proto.unsqueeze(0).repeat(self.train_num_query * self.train_num_way, 1, 1).view(-1,
                                                                                                        self.final_feat_dim)
        z_query_ext = z_query.unsqueeze(0).repeat(self.train_num_way, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1).contiguous().view(-1, self.final_feat_dim)

        z_proto_norm = torch.norm(z_proto_ext, p=2, dim=1).unsqueeze(1).expand_as(z_proto_ext)
        z_proto_normalized = z_proto_ext.div(z_proto_norm + 0.00001)
        z_query_norm = torch.norm(z_query_ext, p=2, dim=1).unsqueeze(1).expand_as(z_query_ext)
        z_query_normalized = z_query_ext.div(z_query_norm + 0.00001)
        distance = z_proto_normalized.unsqueeze(1).bmm(z_query_normalized.unsqueeze(2)).view(-1, self.train_num_way)

        return distance

    def set_forward_loss(self, x):
        x = Variable(x.cuda())
        distance = self.set_forward(x)
        y = torch.from_numpy(np.repeat(range(self.train_num_way), self.train_num_query))
        y = Variable(y.cuda())
        return self.train_loss_softmax(self.train_temperature * distance, y)

    def train_loop(self, epoch, runtime_num_way, runtime_num_query, train_loader, optimizer, scheduler, writer):
        print_freq = 10
        avg_loss = 0
        epoch_len = len(train_loader)
        self.train_num_way = runtime_num_way
        self.train_num_query = runtime_num_query

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            if self.episodic:
                loss = self.set_forward_loss(x)
            else:
                loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            writer.add_scalar('loss/total_loss', loss.item(), epoch * epoch_len + i)

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, epoch_len, avg_loss / float(i + 1)))

        if scheduler is not None:
            scheduler.step(epoch)
            print('Epoch:', epoch, 'LR:', scheduler.get_lr())

    def test_loop(self, cl_data_file):
        if self.test_loss_type == 'softmax':
            classifier = nn.Linear(self.final_feat_dim, self.test_num_way)
            classifier.bias.data.fill_(0)
        elif self.test_loss_type == 'dist':  # Baseline ++
            classifier = backbone.distLinear(self.final_feat_dim, self.train_num_way, self.test_temperature, None)
        elif self.test_loss_type == 'lsoftmax':
            classifier = LSoftmaxLinear(self.final_feat_dim, self.test_num_way, self.test_margin)
            classifier.reset_parameters()
        elif self.test_loss_type == 'ldist':
            classifier = backbone.distLinear(self.final_feat_dim, self.test_num_way, self.test_temperature,
                                                  self.test_margin)

        classifier = classifier.cuda()
        test_loss_softmax = nn.CrossEntropyLoss().cuda()

        class_list = cl_data_file.keys()
        select_class = random.sample(class_list, self.test_num_way)

        z_all = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in
                          range(self.test_num_shot + self.test_num_query)])  # stack each batch
        z_all = torch.from_numpy(np.array(z_all))
        z_all = Variable(z_all.cuda())
        z_support, z_query = self.parse_feature(z_all, True)
        z_support = z_support.contiguous().view(self.test_num_way * self.test_num_shot, -1)
        z_query = z_query.contiguous().view(self.test_num_way * self.test_num_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.test_num_way), self.test_num_shot))
        y_support = Variable(y_support.cuda())

        optimizer = torch.optim.SGD(classifier.parameters(), lr=self.test_lr, momentum=0.9, dampening=0.9,
                                    weight_decay=0.001)

        batch_size = 4
        support_size = self.test_num_way * self.test_num_shot
        for episode in range(100):
            rand_id = np.random.permutation(support_size)
            classifier.train()
            for i in range(0, support_size, batch_size):
                optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                if self.test_loss_type == 'softmax':
                    scores = classifier(z_batch)
                else:
                    scores = classifier(z_batch, y_batch)
                loss = test_loss_softmax(scores, y_batch)
                loss.backward()
                optimizer.step()

        classifier.eval()
        scores = classifier(z_query, None)
        pred = scores.data.cpu().numpy().argmax(axis=1)
        y = np.repeat(range(self.test_num_way), self.test_num_query)
        acc = np.mean(pred == y) * 100
        return acc

    def fast_adapt(self, cl_data_file):
        g_trainer = GTrainer(self.final_feat_dim, self.test_num_way, self.test_temperature).cuda()

        class_list = cl_data_file.keys()
        select_class = random.sample(class_list, self.test_num_way)

        z_all = []
        for cl in select_class:
            img_feat = cl_data_file[cl]
            perm_ids = np.random.permutation(len(img_feat)).tolist()
            z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in
                          range(self.test_num_shot + self.test_num_query)])  # stack each batch
        z_all = torch.from_numpy(np.array(z_all))
        z_all = Variable(z_all.cuda())
        z_support, z_query = self.parse_feature(z_all, True)
        z_support = z_support.contiguous().view(self.test_num_way * self.test_num_shot, -1)
        z_query = z_query.contiguous().view(self.test_num_way * self.test_num_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.test_num_way), self.test_num_shot))
        y_support = Variable(y_support.cuda())

        optimizer = torch.optim.Adam(g_trainer.parameters(), lr=self.test_lr, weight_decay=0.001)

        batch_size = 4
        support_size = self.test_num_way * self.test_num_shot
        for epoch in range(20):
            rand_id = np.random.permutation(support_size)
            g_trainer.train()
            for i in range(0, support_size, batch_size):
                optimizer.zero_grad()
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                scores = g_trainer(z_batch)
                loss = F.cross_entropy(scores, y_batch)
                loss.backward()
                optimizer.step()

        z_proto = z_support.matmul(g_trainer.g).view(self.test_num_way, self.test_num_shot, -1)
        z_proto = torch.mean(z_proto, dim=(1,)).squeeze()
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.test_num_query * self.test_num_way, 1, 1).view(-1, self.final_feat_dim)
        z_query_ext = z_query.unsqueeze(0).repeat(self.test_num_way, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1).contiguous().view(-1, self.final_feat_dim)
        distance = z_proto_ext.unsqueeze(1).bmm(z_query_ext.unsqueeze(2)).view(-1, self.test_num_way)
        pred = distance.data.cpu().numpy().argmin(axis=1)
        y = np.repeat(range(self.test_num_way), self.test_num_query)
        acc = np.mean(pred == y) * 100
        return acc

    def parse_feature(self, x, is_feature):
        if is_feature:
            z_all = x
            z_support = z_all[:, :self.test_num_shot]
            z_query = z_all[:, self.test_num_shot:]
        else:
            x = x.contiguous().view(self.train_num_way * (self.train_num_shot + self.train_num_query), *x.size()[2:])
            z_all = self.feature(x)
            z_all = z_all.view(self.train_num_way, self.train_num_shot + self.train_num_query, -1)
            z_support = z_all[:, :self.train_num_shot]
            z_query = z_all[:, self.train_num_shot:]

        return z_support, z_query
