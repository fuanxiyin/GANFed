#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import torch

import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from torch.autograd import Variable

import torch.nn as nn

import numpy as np
from options import args_parser
from update import test_inference, DatasetSplit
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, exp_details
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #    torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    img_shape = (args.channels, args.img_size, args.img_size)


    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2))  # , inplace=False
                return layers

            self.model = nn.Sequential(
                *block(int(np.prod(img_shape)), 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )

        def forward(self, img):
            # x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
            # img = img.view(img.size(0), *img_shape)
            img = img.view(img.size(0), -1)
            img = self.model(img)

            return img


    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2),  # , inplace=False
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, args.num_users),
                nn.Sigmoid(),
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)

            return validity


    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Set the model to train and send it to device.
    # global_model.to(device)
    global_model.train()
    print(global_model)
    F_global_model = copy.deepcopy(global_model)
    F_generator = copy.deepcopy(generator)
    # copy weights
    global_weights = global_model.state_dict()
    global_G_weights = generator.state_dict()
    global_FG_weights = generator.state_dict()
    F_global_weights = F_global_model.state_dict()
    global_D_weights = discriminator.state_dict()

    # Training

    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    test_acc_iter, test_loss_iter, F_test_acc_iter, F_test_loss_iter = [], [], [], []
    F_train_loss, train_loss = [], []
    for epoch in tqdm(range(args.epochs)):
        local_G_weights, local_FG_weights, local_D_weights, local_weights, F_local_weights, F_local_losses, local_G_losses, local_D_losses, local_losses = copy.deepcopy(
            generator.state_dict()), copy.deepcopy(generator.state_dict()), copy.deepcopy(
            discriminator.state_dict()), copy.deepcopy(global_model.state_dict()), copy.deepcopy(
            global_model.state_dict()), [], [], [], []
        for key in local_G_weights:
            local_G_weights[key] = torch.zeros_like(local_G_weights[key])  # len(local_G_weights[key]).fill_(0.0)
        for key in local_FG_weights:
            local_FG_weights[key] = torch.zeros_like(local_FG_weights[key])  # len(local_G_weights[key]).fill_(0.0)
        for key in local_D_weights:
            local_D_weights[key] = torch.zeros_like(local_D_weights[key])  # len(local_D_weights[key]).fill_(0.0)
        for key in local_weights:
            local_weights[key] = torch.zeros_like(local_weights[key])  # len(local_weights[key]).fill_(0.0)
        for key in F_local_weights:
            F_local_weights[key] = torch.zeros_like(F_local_weights[key])  # len(F_local_weights[key]).fill_(0.0)

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        num_com_count = 0
        for idx in idxs_users:
            GAN_dataloader = DataLoader(DatasetSplit(dataset=train_dataset, idxs=user_groups[idx]),
                                        batch_size=args.batch_size, shuffle=True, num_workers=0)
            Local_generator = copy.deepcopy(generator)
            Local_F_generator = copy.deepcopy(F_generator)
            Local_discriminator = copy.deepcopy(discriminator)
            local_model = copy.deepcopy(global_model)
            F_local_model = copy.deepcopy(F_global_model)
            # optimizer
            # optimizer_LG = torch.optim.Adam(Local_generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
            # optimizer_LD = torch.optim.Adam(Local_discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
            # optimizer_LFG = torch.optim.Adam(Local_F_generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
            optimizer_LG = torch.optim.SGD(Local_generator.parameters(), lr=1 * args.lr, momentum=0.5)
            optimizer_LD = torch.optim.SGD(Local_discriminator.parameters(), lr=1 * args.lr, momentum=0.5)
            optimizer_LFG = torch.optim.SGD(Local_F_generator.parameters(), lr=args.lr, momentum=0.5)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.5)
            F_optimizer = torch.optim.SGD(F_local_model.parameters(), lr=args.lr, momentum=0.5)
            F_local_model.train()
            local_model.train()
            Local_generator.train()
            Local_discriminator.train()
            Local_F_generator.train()
            #ii = 0
            num_com_count += 1
            for i, (imgs, labels) in enumerate(GAN_dataloader):
                GANlabel = Variable(Tensor(imgs.shape[0], args.num_users).fill_(0.0), requires_grad=False)
                GANlabel[:, idx - 1] = 1
                valid = Variable(Tensor(imgs.shape[0], args.num_users).fill_(1), requires_grad=False)

                real_imgs = Variable(imgs.type(Tensor))
                gen_imgs = Local_generator(copy.deepcopy(real_imgs))
                gen_F_imgs = Local_F_generator(copy.deepcopy(real_imgs))

                F_local_model.zero_grad()
                Local_F_generator.zero_grad()
                F_loss_function = torch.nn.CrossEntropyLoss()
                F_loss = F_loss_function(F_local_model(gen_F_imgs), labels)
                F_loss.backward()
                F_optimizer.step()
                optimizer_LFG.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Measure discriminator's ability to classify real from generated samples
                Local_discriminator.zero_grad()
                local_d_loss = adversarial_loss(Local_discriminator(gen_imgs), GANlabel)
                local_d_loss.backward(retain_graph=True)
                optimizer_LD.step()

                local_model.zero_grad()
                Local_generator.zero_grad()
                loss_function = torch.nn.CrossEntropyLoss()
                loss = loss_function(local_model(gen_imgs), labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer_LG.step()
                # Loss measures generator's ability to fool the discriminator
                Local_generator.zero_grad()
                gen_imgs = Local_generator(copy.deepcopy(real_imgs))
                Local_g_loss = adversarial_loss(Local_discriminator(gen_imgs), valid)
                Local_g_loss.backward()
                optimizer_LG.step()

                """
                ii += 1
                if ii == 10:
                   break
"""
            G_w = Local_generator.state_dict()
            D_w = Local_discriminator.state_dict()
            FG_w = Local_F_generator.state_dict()
            w = local_model.state_dict()
            F_w = F_local_model.state_dict()
            F_local_losses.append(F_loss.item())
            local_losses.append(loss.item())

            # Accumulate local weights from different users
            for key in local_G_weights:
                local_G_weights[key] += G_w[key]
            for key in local_FG_weights:
                local_FG_weights[key] += FG_w[key]
            for key in local_D_weights:
                local_D_weights[key] += D_w[key]
            for key in local_weights:
                local_weights[key] += w[key]
            for key in F_local_weights:
                F_local_weights[key] += F_w[key]

        # update global weights
        for key in global_G_weights:
            global_G_weights[key] = (local_G_weights[key] / num_com_count)
        for key in global_FG_weights:
            global_FG_weights[key] = (local_FG_weights[key] / num_com_count)
        for key in global_D_weights:
            global_D_weights[key] = (local_D_weights[key] / num_com_count)
        for key in global_weights:
            global_weights[key] = (local_weights[key] / num_com_count)
        for key in F_global_weights:
            F_global_weights[key] = (F_local_weights[key] / num_com_count)

        # update global weights
        global_model.load_state_dict(global_weights)
        F_global_model.load_state_dict(F_global_weights)
        generator.load_state_dict(global_G_weights)
        F_generator.load_state_dict(global_FG_weights)
        discriminator.load_state_dict(global_D_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        F_loss_avg = sum(F_local_losses) / len(F_local_losses)
        F_train_loss.append(F_loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        """
        #list_acc, list_loss = [], []
        #global_model.eval()
        #for c in range(args.num_users):
            #local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))


        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            #print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
    """
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(train_loss)}')
            print(f'Fdavg Training Loss : {np.mean(F_train_loss)}')

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, generator, test_dataset)
        F_test_acc, F_test_loss = test_inference(args, F_global_model, F_generator, test_dataset)
        test_acc_iter.append(test_acc)
        test_loss_iter.append(test_loss)
        F_test_acc_iter.append(F_test_acc)
        F_test_loss_iter.append(F_test_loss)
        print(f' \n Results after {epoch + 1} global rounds of training:')
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        print("|----Fdavg Test Accuracy: {:.2f}%".format(100 * F_test_acc))
        torch.cuda.empty_cache()
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

