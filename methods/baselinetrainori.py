import backbone
from parts.large_margin_softmax import LSoftmaxLinear

import torch.nn as nn
from torch.autograd import Variable


class Baseline(nn.Module):
    def __init__(self, model_func, num_class, num_shot, train_temperature, test_temperature, loss_type, margin):
        super(Baseline, self).__init__()
        self.feature = model_func()
        self.num_class = num_class
        self.num_shot = num_shot
        self.train_temperature = train_temperature
        self.test_temperature = test_temperature
        self.loss_type = loss_type
        self.margin = margin
        self.final_feature_dim = self.feature.final_feat_dim

        if loss_type == 'softmax':
            self.classifier = nn.Linear(self.final_feat_dim, self.num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'dist':  # Baseline ++
            self.classifier = backbone.distLinear(self.final_feat_dim, self.num_class, self.temperature)
        elif loss_type == 'lsoftmax':
            self.classifier = LSoftmaxLinear(self.final_feat_dim, self.num_class, self.margin)
            self.classifier.reset_parameters()
        elif loss_type == 'ldist':
            self.classifier = backbone.distLinear(self.final_feat_dim, self.num_class, self.temperature, self.margin)

        self.loss_softmax = nn.CrossEntropyLoss().cuda()

    def forward_loss(self, x, y):
        x = Variable(x.cuda())
        y = Variable(y.cuda())
        out = self.feature.forward(x)
        if self.loss_type[0] == 'l':
            scores = self.classifier.forward(out, y)
        else:
            scores = self.classifier.forward(out)

        return self.loss_softmax(scores, y)

    def train_loop(self, epoch, train_loader, optimizer, writer):
        print_freq = 10
        avg_loss = 0
        epoch_len = len(train_loader)

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            writer.add_scalar('loss/total_loss', loss.item(), epoch * epoch_len + i)

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, epoch_len, avg_loss / float(i + 1)))

    def test_loop(self, epoch, test_loader, writer):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous().view(self.n_way * self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support))
        y_support = Variable(y_support.cuda())

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
                writer.add_scalar('loss/test_loss', loss.item(),
                                  ((iter * 100 + epoch) * support_size + i) // batch_size)
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
