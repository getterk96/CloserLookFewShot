import h5py
import numpy as np
from torch.autograd import Variable
import torch.optim
from torch.optim.lr_scheduler import *
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm

import configs
import data.feature_loader as feat_loader
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baseline import Baseline
from io_utils import model_dict, parse_args, get_resume_file, get_best_file


class Experiment():
    def __init__(self, params):
        np.random.seed(10)

        if params.train_dataset == 'cross':
            base_file = configs.data_dir['miniImagenet'] + 'all.json'
            val_file = configs.data_dir['CUB'] + 'val.json'
        elif params.train_dataset == 'cross_char':
            base_file = configs.data_dir['omniglot'] + 'noLatin.json'
            val_file = configs.data_dir['emnist'] + 'val.json'
        else:
            base_file = configs.data_dir[params.train_dataset] + 'base.json'
            val_file = configs.data_dir[params.train_dataset] + 'val.json'

        if 'Conv' in params.model:
            if params.train_dataset in ['omniglot', 'cross_char']:
                image_size = 28
            else:
                image_size = 84
        else:
            image_size = 224

        if params.train_dataset in ['omniglot', 'cross_char']:
            assert params.model == 'Conv4' and not params.train_aug, 'omniglot only support Conv4 without augmentation'
            params.model = 'Conv4S'

        if params.train_dataset == 'omniglot':
            assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        if params.train_dataset == 'cross_char':
            assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'

        params.train_num_query = max(1, int(params.test_num_query * params.test_num_way / params.train_num_way))
        if params.episodic:
            train_few_shot_params = dict(n_way=params.train_num_way, n_support=params.train_num_shot, n_query=params.train_num_query)
            base_datamgr = SetDataManager(image_size, **train_few_shot_params)
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
        else:
            base_datamgr = SimpleDataManager(image_size, batch_size=32)
            base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)

        if params.test_dataset == 'cross':
            novel_file = configs.data_dir['CUB'] + 'novel.json'
        elif params.test_dataset == 'cross_char':
            novel_file = configs.data_dir['emnist'] + 'novel.json'
        else:
            novel_file = configs.data_dir[params.test_dataset] + 'novel.json'

        val_datamgr = SimpleDataManager(image_size, batch_size=64)
        val_loader = val_datamgr.get_data_loader(novel_file, aug=False)

        novel_datamgr = SimpleDataManager(image_size, batch_size=64)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)

        optimizer = params.optimizer

        if params.stop_epoch == -1:
            if params.train_dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.train_dataset in ['CUB']:
                params.stop_epoch = 200  # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.train_dataset in ['miniImagenet', 'cross']:
                params.stop_epoch = 300
            else:
                params.stop_epoch = 300

        shake_config = {'shake_forward': params.shake_forward, 'shake_backward': params.shake_backward,
                        'shake_picture': params.shake_picture}
        train_param = {'loss_type': params.train_loss_type, 'temperature': params.train_temperature,
                       'margin': params.train_margin, 'lr': params.train_lr, 'shake': params.shake,
                       'shake_config': shake_config, 'episodic': params.episodic, 'num_way': params.train_num_way,
                       'num_shot': params.train_num_shot, 'num_query': params.train_num_query, 'num_classes': params.num_classes}
        test_param = {'loss_type': params.test_loss_type, 'temperature': params.test_temperature,
                      'margin': params.test_margin, 'lr': params.test_lr, 'num_way': params.test_num_way,
                      'num_shot': params.test_num_shot, 'num_query': params.test_num_query}

        model = Baseline(model_dict[params.model], params.entropy, train_param, test_param)

        model = model.cuda()

        key = params.tag
        writer = SummaryWriter(log_dir=os.path.join(params.vis_log, key))

        params.checkpoint_dir = '%s/checkpoints/%s/%s' % (configs.save_dir, params.train_dataset, params.checkpoint_dir)

        if not os.path.isdir(params.vis_log):
            os.makedirs(params.vis_log)

        outfile_template = os.path.join(params.checkpoint_dir.replace("checkpoints", "features"), "%s.hdf5")

        if params.mode == 'train' and not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if params.resume or params.mode == 'test':
            if params.mode == 'test':
                self.feature_model = model_dict[params.model]().cuda()
                resume_file = get_best_file(params.checkpoint_dir)
                tmp = torch.load(resume_file)
                state = tmp['state']
                state_keys = list(state.keys())
                for i, key in enumerate(state_keys):
                    if "feature." in key:
                        newkey = key.replace("feature.", "")
                        state[newkey] = state.pop(key)
                    else:
                        state.pop(key)
                self.feature_model.load_state_dict(state)
                self.feature_model.eval()
            else:
                resume_file = get_resume_file(params.checkpoint_dir)
                tmp = torch.load(resume_file)
                state = tmp['state']
                model.load_state_dict(state)
                params.start_epoch = tmp['epoch'] + 1

            print('Info: Model loaded!!!')

        self.params = params
        self.val_file = val_file
        self.base_file = base_file
        self.image_size = image_size
        self.optimizer = optimizer
        self.outfile_template = outfile_template
        self.novel_loader = novel_loader
        self.base_loader = base_loader
        self.val_loader = val_loader
        self.writer = writer
        self.model = model
        self.key = key

    def train(self):
        if self.optimizer == 'Adam':
            train_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.train_lr)
            train_scheduler = StepLR(train_optimizer, step_size=75, gamma=0.1)
        elif self.optimizer == 'SGD':
            train_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.train_lr, momentum=0.9,
                                              weight_decay=0.001)
            train_scheduler = StepLR(train_optimizer, step_size=75, gamma=0.1)
        else:
            raise ValueError('Unknown optimizer, please define by yourself')

        max_acc = 0
        start_epoch = self.params.start_epoch
        stop_epoch = self.params.stop_epoch
        test_start_epoch = int(stop_epoch * self.params.test_start_epoch)
        for epoch in range(start_epoch, stop_epoch):
            self.model.train()
            train_num_way = self.params.train_num_way
            train_num_query = self.params.train_num_query
            if self.params.curriculum:
                train_num_way = int(self.params.train_num_way - (self.params.train_num_way - self.params.test_num_way) * (
                        epoch - start_epoch) / (stop_epoch - start_epoch))
                train_num_query = max(1, int(self.params.test_num_query * self.params.test_num_way / train_num_way))
                train_few_shot_params = dict(n_way=train_num_way, n_support=self.params.train_num_shot,
                                             n_query=train_num_query)
                self.base_datamgr = SetDataManager(self.image_size, **train_few_shot_params)
                self.base_loader = self.base_datamgr.get_data_loader(self.base_file, aug=params.train_aug)
                self.writer.add_scalar('way/curriculum_way', train_num_way, epoch)
            self.model.train_loop(epoch, train_num_way, train_num_query, self.base_loader, train_optimizer, train_scheduler,
                                  self.writer)

            if epoch >= test_start_epoch and (epoch + 1) % 5 == 0:
                self.model.eval()

                acc = self.test('val', epoch)

                if acc > max_acc:  # for baseline and baseline++, we don't use validation here so we let acc = -1
                    print("best model! save...")
                    max_acc = acc
                    outfile = os.path.join(self.params.checkpoint_dir, 'best_model.tar')
                    torch.save({'epoch': epoch, 'state': self.model.state_dict()}, outfile)

            if (epoch % self.params.save_freq == 0) or (epoch == stop_epoch - 1):
                outfile = os.path.join(self.params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch': epoch, 'state': self.model.state_dict()}, outfile)

    def test(self, split='novel', epoch=0):
        self.outfile = self.outfile_template % split
        if split == 'novel':
            self.save_feature(self.novel_loader)
        else:
            self.save_feature(self.val_loader)
        cl_data_file = feat_loader.init_loader(self.outfile)

        acc_all = []
        for i in tqdm(range(self.params.test_epoch)):
            if self.params.fast_adapt:
                acc = self.model.fast_adapt(cl_data_file)
            else:
                acc = self.model.test_loop(cl_data_file)
            acc_all.append(acc)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (
            self.params.test_epoch, acc_mean, 1.96 * acc_std / np.sqrt(self.params.test_epoch)))
        if self.params.mode != 'test':
            self.writer.add_scalar('acc/%s_acc' % split, acc_mean, epoch)

        return acc_mean

    def save_feature(self, data_loader):
        print('Info: Saving feature...')
        dirname = os.path.dirname(self.outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        f = h5py.File(self.outfile, 'w')
        max_count = len(data_loader) * data_loader.batch_size
        all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
        all_feats = None
        count = 0
        for i, (x, y) in enumerate(data_loader):
            x = Variable(x.cuda())
            if self.params.mode == 'train':
                feats = self.model.feature(x)
            else:
                feats = self.feature_model(x)
            if all_feats is None:
                all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
            all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
            all_labels[count:count + feats.size(0)] = y.cpu().numpy()
            count = count + feats.size(0)

        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count

        f.close()
        print('Info: Done saving feature!!!')

    def run(self):
        if self.params.mode == 'train':
            self.train()
        elif self.params.mode == 'test':
            self.test()


if __name__ == '__main__':
    params = parse_args()
    exp = Experiment(params)
    exp.run()
