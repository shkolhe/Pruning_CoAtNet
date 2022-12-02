import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from model.coatnet import *

from torchvision.datasets import CIFAR10

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform  = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)
    
    test_set  = torchvision.datasets.CIFAR10(root="data",
                                             train=False,
                                             download=True,
                                             transform=test_transform)
    
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler  = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=eval_batch_size,
                                               sampler=test_sampler,
                                               num_workers=num_workers)
    classes = train_set.classes

    return train_loader, test_loader, classes


def load_model(model_name, model_dir, device):
    model_path = os.path.join(model_dir, model_name)
    model = torch.load(model_path).to(device)
    return model


def save_model(model, model_name, model_dir):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    torch.save(model, model_path)


def evaluate_model(model, test_loader, device, loss_func=None):
    model.eval()
    model.to(device)

    total_loss = 0
    correct_preds = 0

    for inputs, labels in test_loader:
        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        if loss_func is not None:
            loss = loss_func(outputs, labels).item()
        else:
            loss = 0

        # Collect Statistics
        total_loss += loss * inputs.size(0)
        correct_preds += torch.sum(predictions == labels.data)
    
    eval_loss = total_loss / len(test_loader.dataset)
    eval_acc  = correct_preds / len(test_loader.dataset)

    return eval_loss, eval_acc


def train_model(model, train_loader, test_loader, learning_rate, weight_decay, num_epochs, device):

    model.to(device)

    # Use Cross Entropy Loss function
    criterion = nn.CrossEntropyLoss()

    # Select between Adam and SGD optimizer
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate,
    #                       betas=(0.9,0.999), eps=1e-08,
    #                       weight_decay=weight_decay, amsgrad=False)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)

    # Use a Multistep Scheduler to iteratively update the learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150],
                                               gamma=0.1, last_epoch=-1)

    # Evaluate the model before training
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model, test_loader, device, criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(0, eval_loss, eval_accuracy))

    # Start training the model
    for epoch in range(num_epochs):
        
        model.train()

        total_loss = 0
        correct_preds = 0

        for inputs, labels in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Set the gradient to Zero
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(inputs)

            # Loss
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward Pass
            loss.backward()

            # Optimizer Step
            optimizer.step()

            # Collect Statistics
            total_loss += loss.item() * inputs.size(0)
            correct_preds += torch.sum(preds == labels.data)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = correct_preds / len(train_loader.dataset)

        # Evaluate model after training for 1 epoch
        model.eval()
        eval_loss, eval_acc = evaluate_model(model, test_loader, device, criterion)
        print("Epoch: {:03d} | Train Loss: {:.3f}, Train Acc: {:.3f} | Eval Loss: {:.3f}, Eval Acc: {:.3f}"
              .format(epoch+1, train_loss, train_acc, eval_loss, eval_accuracy))

        # Update Learning Rate
        scheduler.step()

    return model

