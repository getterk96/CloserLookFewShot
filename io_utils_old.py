import numpy as np
import os
import time
import glob
import argparse
import backbone

model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101)


def parse_args():
    parser = argparse.ArgumentParser(description='few-shot script')
    parser.add_argument('--dataset', default='miniImagenet', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model', default='ResNet18',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')  # 50 and 101 are not used in the paper
    parser.add_argument('--method', default='baseline',
                        help='baseline/protonet/matchingnet/relationnet{_softmax}/maml{_approx}')  # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way', default=5, type=int,
                        help='class num to classify for training')  # baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way', default=5, type=int,
                        help='class num to classify for testing (validation) ')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')  # baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--loss_type', default='lsoftmax', help='loss type for the baseline')
    parser.add_argument('--temperature', default=30, type=int, help='temperature of the softmax layer')
    parser.add_argument('--margin', default=1, type=int, help='margin of the large margin layer')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training')  # still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--centered', action='store_true',
                        help='use center loss or not')  # use center loss or not

    parser.add_argument('--num_classes', default=200, type=int,
                        help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in base class
    parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
    parser.add_argument('--test_start_epoch', default=0.8, type=float, help='From which epoch start testing')
    parser.add_argument('--stop_epoch', default=-1, type=int,
                        help='Stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
    parser.add_argument('--resume', action='store_true',
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--warmup', action='store_true',
                        help='continue from baseline, neglected if resume is true')  # never used in the paper
    parser.add_argument('--split', default='novel',
                        help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
    parser.add_argument('--save_iter', default=-1, type=int,
                        help='save feature from the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--split', default='novel',
                        help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
    parser.add_argument('--save_iter', default=-1, type=int,
                        help='saved feature from the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
