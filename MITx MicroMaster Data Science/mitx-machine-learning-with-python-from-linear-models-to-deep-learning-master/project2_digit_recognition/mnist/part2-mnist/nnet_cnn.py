#! /usr/bin/env python

import _pickle as c_pickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model, Flatten
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def main():
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Reshape the data back into a 1x28x28 image
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28))

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    # Baseline parameters
    baseline_batch_size = 32
    baseline_hidden_size = 128
    baseline_learning_rate = 0.1
    baseline_momentum = 0
    baseline_activation = nn.ReLU

    

    def create_model(hidden_size, activation):
        model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            activation(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(5408, hidden_size),
            activation(),
            nn.Linear(hidden_size, num_classes)
        )
        return model.to(device)

    train_batches = batchify_data(X_train, y_train, baseline_batch_size)
    dev_batches = batchify_data(X_dev, y_dev, baseline_batch_size)
    model = create_model(baseline_hidden_size, baseline_activation)
    accuracy = train_model(train_batches, dev_batches, model, lr=baseline_learning_rate, momentum=baseline_momentum, nesterov=False, n_epochs=10)

    print(f'Baseline Accuracy: {accuracy}')

    best_accuracy = 0
    best_params = {}
    training_results = []
    # Hyperparameters for grid search
    hyperparameters = {
        'batch_size': [64],
        'hidden_size':[128],
        'learning_rate': [0.01],
        'momentum': [0.9],
        'activation': [nn.LeakyReLU]
    }
    # Mini grid search over each hyperparameter
    for batch_size in hyperparameters['batch_size']:
        train_batches = batchify_data(X_train, y_train, batch_size)
        dev_batches = batchify_data(X_dev, y_dev, batch_size)
        model = create_model(baseline_hidden_size, baseline_activation)
        accuracy = train_model(train_batches, dev_batches, model, lr=baseline_learning_rate, momentum=baseline_momentum, nesterov=False, n_epochs=10)
        training_results.append(('batch_size', batch_size, accuracy))
        print(f'Hyperparameter: batch_size = {batch_size}, Accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'batch_size': batch_size, 'hidden_size': baseline_hidden_size, 'learning_rate': baseline_learning_rate, 'momentum': baseline_momentum, 'activation': baseline_activation}

    for learning_rate in hyperparameters['learning_rate']:
        train_batches = batchify_data(X_train, y_train, baseline_batch_size)
        dev_batches = batchify_data(X_dev, y_dev, baseline_batch_size)
        model = create_model(baseline_hidden_size, baseline_activation)
        accuracy = train_model(train_batches, dev_batches, model, lr=learning_rate, momentum=baseline_momentum, nesterov=False, n_epochs=10)
        training_results.append(('learning_rate', learning_rate, accuracy))
        print(f'Hyperparameter: learning_rate = {learning_rate}, Accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'batch_size': baseline_batch_size, 'hidden_size': baseline_hidden_size, 'learning_rate': learning_rate, 'momentum': baseline_momentum, 'activation': baseline_activation}

    for momentum in hyperparameters['momentum']:
        train_batches = batchify_data(X_train, y_train, baseline_batch_size)
        dev_batches = batchify_data(X_dev, y_dev, baseline_batch_size)
        model = create_model(baseline_hidden_size, baseline_activation)
        accuracy = train_model(train_batches, dev_batches, model, lr=baseline_learning_rate, momentum=momentum, nesterov=False, n_epochs=10)
        training_results.append(('momentum', momentum, accuracy))
        print(f'Hyperparameter: momentum = {momentum}, Accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'batch_size': baseline_batch_size, 'hidden_size': baseline_hidden_size, 'learning_rate': baseline_learning_rate, 'momentum': momentum, 'activation': baseline_activation}

    for activation in hyperparameters['activation']:
        train_batches = batchify_data(X_train, y_train, baseline_batch_size)
        dev_batches = batchify_data(X_dev, y_dev, baseline_batch_size)
        model = create_model(baseline_hidden_size, activation)
        accuracy = train_model(train_batches, dev_batches, model, lr=baseline_learning_rate, momentum=baseline_momentum, nesterov=False, n_epochs=10)
        training_results.append(('activation', activation, accuracy))
        print(f'Hyperparameter: momentum = {activation}, Accuracy: {accuracy}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'batch_size': baseline_batch_size, 'hidden_size': baseline_hidden_size, 'learning_rate': baseline_learning_rate, 'momentum': baseline_momentum, 'activation': activation}

    print("Best hyperparameters found: ", best_params)
    
if __name__ == "__main__":
    main()