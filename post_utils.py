import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision import datasets, transforms
import apex.amp as amp
import numpy as np

from trades import trades_loss


cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)


mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()


upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def cal_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    # collect the correct predictions for each class
    correct = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1
    return correct / total


def merge_images(train_images, val_images, ratio, device):
    batch_size = len(train_images)
    repeated_val_images = val_images.repeat(batch_size, 1, 1, 1)
    merged_images = ratio * train_images.to(device) + (1 - ratio) * repeated_val_images.to(device)
    # image[0][channel] = 0.5 * image[0][channel].to(device) + 0.5 * val_images[0][channel].to(device)
    return merged_images


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_train_loaders_by_class(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    indices_list = [[] for _ in range(10)]
    for i in range(len(train_dataset)):
        label = int(train_dataset[i][1])
        indices_list[label].append(i)
    dataset_list = [Subset(train_dataset, indices) for indices in indices_list]
    train_loader_list = [
        torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        ) for dataset in dataset_list
    ]
    return train_loader_list


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    epsilon = torch.ones([3, 1, 1]).cuda() * epsilon
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd_trades(model, data, label, epsilon, alpha, step_count, random_start, device):
    X, y = Variable(data, requires_grad=True), Variable(label)
    X_pgd = Variable(X.data, requires_grad=True)
    if random_start:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    for _ in range(step_count):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = alpha * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd



def post_train(model, images, train_loader, train_loaders_by_class, args):
    alpha = (10 / 255)
    epsilon = (8 / 255)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    # attack_model = torchattacks.PGD(model, eps=(8/255)/std, alpha=(2/255)/std, steps=20)
    optimizer = torch.optim.SGD(lr=0.01,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    # target_bce_loss_func = TargetBCELoss()
    # target_bl_loss_func = TargetBLLoss()
    images = images.detach()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)
        # neighbour_images = attack_model(images, original_class)
        # neighbour_delta = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=20, restarts=1)
        # neighbour_images = neighbour_delta + images
        neighbour_images = attack_pgd_trades(fix_model, images, original_class, epsilon, alpha, 20, False, device)
        neighbour_delta = (neighbour_images - images).detach()
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            print('original class == neighbour class')
            if args.pt_data == 'ori_neigh':
                return model, original_class, neighbour_class, None, None

        loss_list = []
        acc_list = []
        for _ in range(args.pt_iter):
            # # randomize neighbour
            # if args.pt_data == 'ori_rand':
            #     neighbour_class = (original_class + random.randint(1, 9)) % 10
            # elif args.pt_data == 'rand':
            #     original_class = (original_class + random.randint(0, 9)) % 10
            #     neighbour_class = (original_class + random.randint(0, 9)) % 10
            # else:
            #     raise NotImplementedError

            original_data, original_label = next(iter(train_loaders_by_class[original_class]))
            neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))

            # train_data, train_label = next(iter(train_loader))
            # data = train_data.to(device)
            # label = train_label.to(device)

            if args.pt_data == 'ori_neigh_train':
                raise NotImplementedError
                # data = torch.vstack([original_data, neighbour_data, train_data]).to(device)
                # label = torch.hstack([original_label, neighbour_label, train_label]).to(device)
            else:
                data = torch.vstack([original_data, neighbour_data]).to(device)
                label = torch.hstack([original_label, neighbour_label]).to(device)


            # if args.mixup:
            #     data = merge_images(data, images, 0.7, device)
            # target = torch.hstack([neighbour_label, original_label]).to(device)

            # # generate pgd adv examples
            # X, y = Variable(data, requires_grad=True), Variable(label)
            # X_pgd = Variable(X.data, requires_grad=True)
            # random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            # X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
            # for _ in range(20):
            #     opt = torch.optim.SGD([X_pgd], lr=1e-3)
            #     opt.zero_grad()
            #
            #     with torch.enable_grad():
            #         loss = nn.CrossEntropyLoss()(fix_model(X_pgd), y)
            #     loss.backward()
            #     eta = 0.003 * X_pgd.grad.data.sign()
            #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            #     eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            #     X_pgd = Variable(X.data + eta, requires_grad=True)
            #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
            # adv_input = X_pgd

            # # generate fgsm adv examples
            # delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
            # noise_input = data + delta
            # noise_input.requires_grad = True
            # noise_output = model(noise_input)
            # loss = loss_func(noise_output, label)  # loss to be maximized
            # # loss = target_bce_loss_func(noise_output, label, original_class, neighbour_class)  # bce loss to be maximized
            # input_grad = torch.autograd.grad(loss, noise_input)[0]
            # delta = delta + alpha * torch.sign(input_grad)
            # delta.clamp_(-epsilon, epsilon)
            # adv_input = data + delta
            # adv_input = data + (torch.randint(0, 2, size=()) - 0.5).to(device) * 2 * neighbour_delta
            # adv_input = data + -1 * torch.rand_like(data).to(device) * neighbour_delta
            adv_input = data + -1 * neighbour_delta
            # directed_delta = torch.vstack([torch.ones_like(original_data).to(device) * neighbour_delta,
            #                                 torch.ones_like(neighbour_data).to(device) * -1 * neighbour_delta])
            # adv_input = data + directed_delta

            # generate pgd adv example
            # attack_model.set_mode_targeted_by_function(lambda im, la: target)
            # adv_input = attack_model(data, label)

            normal_output = model(data.detach())
            if args.pt_method == 'adv':
                adv_output = model(adv_input.detach())
            elif args.pt_method == 'normal':
                adv_output = model(data.detach())  # non adv training
            else:
                raise NotImplementedError

            _, adv_output_class = torch.max(adv_output, 1)
            original_class_expanded = torch.ones_like(adv_output_class) * int(original_class)
            neighbour_class_expanded = torch.ones_like(adv_output_class) * int(neighbour_class)
            # print(adv_output.shape, normal_output.shape, adv_output_class.shape, original_class_expanded.shape, neighbour_class_expanded.shape)
            filter_condition = torch.logical_or(torch.eq(adv_output_class, original_class_expanded),
                                                torch.eq(adv_output_class, neighbour_class_expanded))
            filter_condition = filter_condition.unsqueeze(1).expand([len(filter_condition), 10])
            print(filter_condition.shape)
            adv_output = torch.where(filter_condition, adv_output, normal_output)

            # adv_class = torch.argmax(adv_output)
            # loss_pos = loss_func(adv_output, label)
            loss_norm = loss_func(normal_output, label)
            loss_kl = kl_loss(F.log_softmax(adv_output), F.softmax(normal_output))
            # loss_trades = trades_loss(model, data, label, optimizer)
            # loss_neg = loss_func(adv_output, target)
            # bce_loss = target_bce_loss_func(adv_output, label, original_class, neighbour_class)
            # bl_loss = target_bl_loss_func(adv_output, label, original_class, neighbour_class)

            # loss = torch.mean(loss_list)
            loss = loss_norm + 6 * loss_kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
            print('loss: {:.4f}  acc: {:.4f}'.format(loss, defense_acc))
    return model, original_class, neighbour_class, loss_list, acc_list
