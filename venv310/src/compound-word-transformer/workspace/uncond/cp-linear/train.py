import sys
import os
import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import shutil
from main_cp import *
MODE = 'train'

###--- data ---###
path_data_root = './../../../dataset/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_test_data = os.path.join(path_data_root, 'test_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'dictionary.pkl')

###--- training config ---###
D_MODEL = 512
N_LAYER = 12
N_HEAD = 8
path_exp = './exp'
batch_size = 2
gid = 0
init_lr = 0.0001

max_grad_norm = 3
path_gendir = 'gen_midis'
num_songs = 5

init_lr = 0.0001

def get_train_data():
  dictionary = pickle.load(open(path_dictionary, 'rb'))
  event2word, word2event = dictionary
  train_data = np.load(path_train_data)
  return train_data, event2word, word2event, dictionary

def get_test_data():
  dictionary = pickle.load(open(path_dictionary, 'rb'))
  event2word, word2event = dictionary
  test_data = np.load(path_test_data)
  return test_data, event2word, word2event, dictionary

def train(info_load_model = None, n_epoch = 200):
    # Load data
    train_data, event2word, word2event, dictionary = get_train_data()

    # create saver
    saver_agent = saver.Saver(path_exp)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # log
    print('num of classes:', n_class)

    # init
    net = TransformerModel(n_class)
    net.cuda()
    net.train()
    n_parameters = network_paras(net)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    # load model
    if info_load_model:
        path_ckpt = info_load_model[0] # path to ckpt dir
        loss = info_load_model[1] # loss
        name = 'loss_' + str(loss)
        path_saved_ckpt = os.path.join(path_ckpt, name + '_params.pt')
        print('[*] load model from:',  path_saved_ckpt)
        net.load_state_dict(torch.load(path_saved_ckpt))

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=init_lr)

    # unpack
    train_x = train_data['x']
    train_y = train_data['y']
    train_mask = train_data['mask']
    num_batch = len(train_x) // batch_size

    print('num_batch:', num_batch,'\ntrain_x:', train_x.shape,'\ntrain_y:', train_y.shape,'\ntrain_mask:', train_mask.shape)

    # run
    start_time = time.time()
    for epoch in range(n_epoch):
        acc_loss = 0
        acc_losses = np.zeros(7)

        with tqdm(range(num_batch)) as bar:
          for bidx in range(num_batch): # num_batch
              saver_agent.global_step_increment()

              # index
              bidx_st = batch_size * bidx
              bidx_ed = batch_size * (bidx + 1)

              # unpack batch data
              batch_x = train_x[bidx_st:bidx_ed]
              batch_y = train_y[bidx_st:bidx_ed]
              batch_mask = train_mask[bidx_st:bidx_ed]

              # to tensor
              batch_x = torch.from_numpy(batch_x).long().cuda()
              batch_y = torch.from_numpy(batch_y).long().cuda()
              batch_mask = torch.from_numpy(batch_mask).float().cuda()

              # run
              losses = net.train_step(batch_x, batch_y, batch_mask)
              loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6]) / 7

              # Update
              net.zero_grad()
              loss.backward()
              if max_grad_norm is not None:
                  clip_grad_norm_(net.parameters(), max_grad_norm)
              optimizer.step()

              # print
              sys.stdout.write('{}/{} | Loss: {:06f} | {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}, {:04f}\r'.format(
                  bidx, num_batch, loss, losses[0], losses[1], losses[2], losses[3], losses[4], losses[5], losses[6]))
              sys.stdout.flush()
              bar.update()

              # acc
              acc_losses += np.array([l.item() for l in losses])
              acc_loss += loss.item()

              # log
              saver_agent.add_summary('batch loss', loss.item())

        # epoch loss
        runtime = time.time() - start_time
        epoch_loss = acc_loss / num_batch
        acc_losses = acc_losses / num_batch
        print('------------------------------------')
        print('epoch: {}/{} | Loss: {} | time: {}'.format(
            epoch+1, n_epoch, epoch_loss, str(datetime.timedelta(seconds=runtime))))

        saver_agent.add_summary('epoch loss', epoch_loss)

        # save model, with policy

        loss = epoch_loss
        if 0.4 < loss <= 0.8:
            fn = int(loss * 10) * 10
            saver_agent.save_model(net, name='loss_' + str(fn))

        elif 0.05 < loss <= 0.40:
            fn = int(loss * 100)
            saver_agent.save_model(net, name='loss_' + str(fn))

        elif loss <= 0.05:
            print('Finished')
            return
        else:
            saver_agent.save_model(net, name='loss_high'+ "_epoch_" + str(epoch))
        saved_files = os.listdir('./exp')
        for saved in saved_files:
            Mydrive_saved_model = "./drive/MyDrive/music_model"
            if not (saved in os.listdir('./drive/MyDrive/music_model')):

              shutil.copy("./exp/"+saved , "./drive/MyDrive/music_model")


train(info_load_model = info_load_model , n_epoch = 8)