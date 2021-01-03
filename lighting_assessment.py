#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:34:48 2020

@author: petar
"""

#import cv2 as cv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torchvision
import torch.optim as optim


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=5)

    self.fc1 = nn.Linear(5376, 1000)
    self.fc2 = nn.Linear(1000, 200)
    self.fc3 = nn.Linear(200, 3)

    self.batch = nn.BatchNorm1d(1000)
    
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = x.view(-1, 5376)
    x = self.fc1(x)
    x = self.relu(x)

    x = self.batch(x)

    x = self.fc2(x)
    x = self.relu(x)

    x = self.fc3(x)
    x = self.relu(x)
    
    return torch.tanh(x)

def get_images(root, last_image_index):
    train_images = []
    test_images = []
    
    for i in range(1, last_image_index):
      name = str(i) + ".JPG"
      img = Image.open(root + '/' + name)
      basewidth = 128
      wpercent = (basewidth/float(img.size[0]))
      hsize = int((float(img.size[1])*float(wpercent)))
      img = img.resize((basewidth,hsize), Image.ANTIALIAS)
      a = np.asarray(img, dtype='float32')
      a = a * 256
      a = a - 2048
      a[a<0] = 0
      a /= 65536
      a = torch.from_numpy(a)
      a = a.transpose(0, 2)
      if i < int(0.8*last_image_index):
        train_images.append(a)
      else:
        test_images.append(a)
        
    return train_images, test_images

train_images, test_images = get_images("./../JPG", 1708)

train_images = torch.stack(train_images)
test_images = torch.stack(test_images)

print(train_images.shape, test_images.shape)

def get_labels(root):
    labels = []
    
    labels_file = open(root, 'r')
    lines = labels_file.readlines()
    
    for line in lines:
        rgb = []
        line = line.split(" ")
        rgb.append(float(line[0]))
        rgb.append(float(line[1]))
        rgb.append(float(line[2]))
        
        labels.append(rgb)
        
    return labels

labels = torch.FloatTensor(get_labels("./dataset/cube+_gt.txt"))

train_labels = labels[:int(0.8*len(labels))]
test_labels = labels[int(0.8*len(labels)):]

train_images = train_images.type(torch.FloatTensor)
test_images = test_images.type(torch.FloatTensor)

train_labels = train_labels.type(torch.FloatTensor)
test_labels = test_labels.type(torch.FloatTensor)

print(train_images.shape, train_labels.shape)

train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


n_epochs = 10
learning_rate = 0.00005
device = 'cpu'

def train_step(network, train_data, epoch, device):
  losses = []
  counter = []

  network.train()
  for idx, (data, target) in enumerate(train_data):
    data = data.to(device)
    target = target.to(device)

    network.zero_grad()
    output = network(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    if idx % 5 == 0:
      print('Train Epoch: {:2d}, ({:2.0f}%), Loss: {:.5f}'.format(
          epoch, 100*idx*64/len(train_data.dataset), loss
      ))
      losses.append(loss)
      counter.append(idx*64 + (epoch-1)*len(train_data.dataset))

  return losses, counter

def loss(output, target):
    return 1


def test(network, test_data, device):
  network.eval()

  loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_data:
      data = data.to(device)
      target = target.to(device)

      output = network(data)
      loss += F.mse_loss(output, target, reduction="sum").item()

  loss /= len(test_data.dataset)
  accuracy = 100 * correct / len(test_data.dataset)

  return loss, accuracy

def train(network, train_data, test_data, device='cpu'):
  test_loss = []
  test_acc = []
  train_loss = []
  counter = []

  # testiramo pocetni model
  loss_t, acc_t = test(network, test_data, device)
  test_loss.append(loss_t)
  test_acc.append(acc_t)

  for epoch in range(1, n_epochs+1):
    # korak i testsad 
    loss, cnt = train_step(network, train_data, epoch, device)
    loss_t, acc_t = test(network, test_data, device)

    test_loss.append(loss_t)
    test_acc.append(acc_t)
    train_loss.append(loss)
    counter.append(cnt)

  return test_loss, test_acc, train_loss, counter

network = Net()

optimizer = optim.Adam(network.parameters(), lr=learning_rate)
test_loss, test_acc, train_loss, counter = train(network, train_loader, test_loader, device)


