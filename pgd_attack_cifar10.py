from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from post_utils import get_train_loaders_by_class, post_train, attack_pgd

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

# post train parameters
parser.set_defaults(mixup=True, type=bool)
parser.add_argument('--no-mixup', dest='mixup', action='store_false')
parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'rand', 'ori_neigh_train'], type=str)
parser.add_argument('--pt-method', default='adv', choices=['adv', 'normal'], type=str)
parser.add_argument('--pt-iter', default=5, type=int)
parser.add_argument('--pt-lr', default=0.01, type=float)
parser.set_defaults(rs_neigh=True, type=bool)
parser.add_argument('--no-rs-neigh', dest='rs_neigh', action='store_false')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
# trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    # devide by batch size
    err /= len(y)
    err_pgd /= len(y)
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_whitebox_post(model, X, y, train_loaders_by_class,
                       epsilon=args.epsilon, num_steps=args.num_steps, step_size=args.step_size,):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    # strong attack from FGSM-RS
    # alpha = (2 / 255)
    # X_pgd = attack_pgd(model, X, y, epsilon, alpha, 50, 10).detach() + X.detach()
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()

    post_model, original_class, neighbour_class, loss_list, acc_list = post_train(model, X_pgd, None, train_loaders_by_class, args)
    err_pgd_post = (post_model(X_pgd).data.max(1)[1] != y.data).float().sum()
    neighbour_acc = 1 if neighbour_class == y or original_class == y else 0

    # double attack
    X_pgd = Variable(X.data.detach(), requires_grad=True)
    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(post_model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data.detach(), -epsilon, epsilon)
        X_pgd = Variable(X.data.detach() + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    # strong attack from FGSM-RS
    # alpha = (2 / 255)
    # X_pgd = attack_pgd(post_model, X, y, epsilon, alpha, 50, 10).detach() + X.detach()
    err_pgd_double = (post_model(X_pgd).data.max(1)[1] != y.data).float().sum()

    # devide by batch size
    err /= len(y)
    err_pgd /= len(y)
    err_pgd_post /= len(y)
    err_pgd_double /= len(y)
    # print('err pgd (white-box): ', err_pgd)
    # print('err pgd post (white-box): ', err_pgd_post)
    return err, err_pgd, err_pgd_post, err_pgd_double, neighbour_acc


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()

    # devide by batch size
    err /= len(y)
    err_pgd /= len(y)
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    batch_count = len(test_loader)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural

    # divide by batch count
    natural_err_total /= batch_count
    robust_err_total /= batch_count
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def eval_adv_test_whitebox_post(model, device):
    """
    evaluate model by white-box attack
    """

    # create separate train and test loaders
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, **kwargs)
    train_loaders_by_class = get_train_loaders_by_class('../data', batch_size=64)

    model.eval()
    natural_err_total = 0
    robust_err_total = 0
    robust_err_total_post = 0
    robust_err_total_double = 0
    batch_count = len(test_loader)

    neighbour_total_acc = 0

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust, err_robust_post, err_robust_double, neighbour_acc = _pgd_whitebox_post(model, X, y, train_loaders_by_class)
        natural_err_total += err_natural
        robust_err_total += err_robust
        robust_err_total_post += err_robust_post
        robust_err_total_double += err_robust_double
        neighbour_total_acc += neighbour_acc
        print('batch {:}: robust error: {:.4f}({:.4f})\t robust post error: {:.4f}({:.4f}) \t '
              'robust double error: {:.4f}({:.4f}) neighbour acc: {:.4f}({:.4f})'
              .format(i, err_robust, robust_err_total/(i+1), err_robust_post, robust_err_total_post/(i+1),
                      err_robust_double, robust_err_total_double/(i+1), neighbour_acc, neighbour_total_acc/(i+1)))

    # divide by batch count
    natural_err_total /= batch_count
    robust_err_total /= batch_count
    robust_err_total_post /= batch_count
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
    print('robust_err_total_post: ', robust_err_total_post)


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0
    batch_count = len(test_loader)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural

    # divide by batch count
    natural_err_total /= batch_count
    robust_err_total /= batch_count
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():
    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        model = WideResNet().to(device)
        model.load_state_dict(torch.load(args.model_path))

        # eval_adv_test_whitebox(model, device, test_loader)
        eval_adv_test_whitebox_post(model, device)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
