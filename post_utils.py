import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
import apex.amp as amp
import numpy as np


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


def post_train(model, images, train_loaders_by_class):
    alpha = (10 / 255)
    epsilon = (8 / 255)
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model = copy.deepcopy(model)
    # model.train()
    fix_model = copy.deepcopy(model)
    # attack_model = torchattacks.PGD(model, eps=(8/255)/std, alpha=(2/255)/std, steps=20)
    optimizer = torch.optim.SGD(lr=0.001,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    # target_bce_loss_func = TargetBCELoss()
    # target_bl_loss_func = TargetBLLoss()
    with torch.enable_grad():
        # find neighbour
        original_output = fix_model(images)
        original_class = torch.argmax(original_output).reshape(1)
        # neighbour_images = attack_model(images, original_class)
        neighbour_images = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=20, restarts=1) + images
        neighbour_output = fix_model(neighbour_images)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            print('original class == neighbour class')
            # return model, original_class, neighbour_class, None, None

        loss_list = []
        acc_list = []
        for _ in range(50):
            # randomize neighbour
            neighbour_class = (original_class + random.randint(1, 9)) % 10

            original_data, original_label = next(iter(train_loaders_by_class[original_class]))
            neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))

            data = torch.vstack([original_data, neighbour_data]).to(device)
            data = merge_images(data, images, 0.7, device)
            label = torch.hstack([original_label, neighbour_label]).to(device)
            target = torch.hstack([neighbour_label, original_label]).to(device)

            # generate fgsm adv examples
            delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
            noise_input = data + delta
            noise_input.requires_grad = True
            noise_output = model(noise_input)
            loss = loss_func(noise_output, label)  # loss to be maximized
            # loss = target_bce_loss_func(noise_output, label, original_class, neighbour_class)  # bce loss to be maximized
            input_grad = torch.autograd.grad(loss, noise_input)[0]
            delta = delta + alpha * torch.sign(input_grad)
            delta.clamp_(-epsilon, epsilon)
            adv_input = data + delta

            # generate pgd adv example
            # attack_model.set_mode_targeted_by_function(lambda im, la: target)
            # adv_input = attack_model(data, label)

            adv_output = model(adv_input.detach())
            # adv_class = torch.argmax(adv_output)
            loss_pos = loss_func(adv_output, label)
            loss_neg = loss_func(adv_output, target)
            # bce_loss = target_bce_loss_func(adv_output, label, original_class, neighbour_class)
            # bl_loss = target_bl_loss_func(adv_output, label, original_class, neighbour_class)

            # loss = torch.mean(loss_list)
            loss = loss_pos
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            defense_acc = cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
            # print('loss: {:.4f}  acc: {:.4f}'.format(loss, defense_acc))
    return model, original_class, neighbour_class, loss_list, acc_list