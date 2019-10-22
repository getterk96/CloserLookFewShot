import numpy as np
import os
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
    parser.add_argument('--mode', default='train', help='model: train/test')
    parser.add_argument('--train_dataset', default='miniImagenet', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--test_dataset', default='miniImagenet', help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model', default='ResNet18',
                        help='model: Conv{4|6} / ResNet{10|18|34|50|101}')  # 50 and 101 are not used in the paper
    parser.add_argument('--optimizer', default='SGD',
                        help='the optimizer of the training process')  # 50 and 101 are not used in the paper
    parser.add_argument('--num_classes', default=200, type=int,
                        help='total number of classes in softmax, only used in baseline')
    parser.add_argument('--train_num_way', default=5, type=int,
                        help='class num to classify for training')
    parser.add_argument('--train_num_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--test_num_way', default=5, type=int,
                        help='class num to classify for testing')
    parser.add_argument('--test_num_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--test_num_query', default=15, type=int,
                        help='the number of query examples in each episode')
    parser.add_argument('--train_loss_type', default='lsoftmax', help='loss type for the training classifier')
    parser.add_argument('--test_loss_type', default='lsoftmax', help='loss type for the testing classifier')
    parser.add_argument('--train_temperature', default=30, type=int, help='temperature of the training softmax layer')
    parser.add_argument('--test_temperature', default=4, type=int, help='temperature of the testing softmax layer')
    parser.add_argument('--train_margin', default=1, type=int, help='margin of the training large margin layer')
    parser.add_argument('--test_margin', default=1, type=int, help='margin of the testing large margin layer')
    parser.add_argument('--train_lr', default=0.01, type=float, help='training learning rate')
    parser.add_argument('--test_lr', default=0.01, type=float, help='testing learning rate')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training')  # still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--shake', action='store_true', help='perform shake-shake regularization or not')
    parser.add_argument('--shake_forward', action='store_true', help='shake-shake on forward process')
    parser.add_argument('--shake_backward', action='store_true', help='shake-shake on backward process')
    parser.add_argument('--shake_picture', action='store_true', help='shake-shake on every image in a batch')

    parser.add_argument('--save_freq', default=50, type=int, help='save frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
    parser.add_argument('--test_start_epoch', default=0.8, type=float, help='from which epoch start testing')
    parser.add_argument('--test_epoch', default=600, type=int, help='total epoch number while testing')
    parser.add_argument('--stop_epoch', default=-1, type=int,
                        help='stopping epoch')  # for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
    parser.add_argument('--resume', action='store_true',
                        help='continue from previous trained model with largest epoch')
    parser.add_argument('--episodic', action='store_true', help='train in an episodic manner')
    parser.add_argument('--entropy', action='store_true', help='train calculating the class prototype with entropy')
    parser.add_argument('--curriculum', action='store_true', help='train in a curriculum episodic manner')
    parser.add_argument('--fast_adapt', action='store_true', help='fast adapt in stage-2')
    parser.add_argument('--warmup', action='store_true',
                        help='continue from baseline, neglected if resume is true')  # never used in the paper
    parser.add_argument('--vis_log', default='/home/gaojinghan/closer/vis_log', help='the tensorboard log storage dir')
    parser.add_argument('--tag', default='', help='the tag of the experiment')
    parser.add_argument('--checkpoint_dir', default='', help='the place to store checkpoints')

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
