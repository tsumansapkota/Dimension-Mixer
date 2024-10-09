import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np

import random, time, json
from tqdm import tqdm

from ptflops import get_model_complexity_info

################################################
device = torch.device("cuda:0")
################################################

#### for when torch.compile does not work
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
################################################
## For CIFAR datasets

def get_c10_dataset():
    cifar_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
            std=[0.2023, 0.1994, 0.2010],  # std=[0.2009, 0.1984, 0.2023] for cifar100
        ),
    ])

    cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
            std=[0.2023, 0.1994, 0.2010],  # std=[0.2009, 0.1984, 0.2023] for cifar100
        ),
    ])

    lib_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(lib_dir, '_Datasets/cifar10/')
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=cifar_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=cifar_test)
    return train_dataset, test_dataset


def get_c100_dataset():
    cifar_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2009, 0.1984, 0.2023],
        ),
    ])

    cifar_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4865, 0.4409],
            std=[0.2009, 0.1984, 0.2023],
        ),
    ])
    lib_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(lib_dir, '_Datasets/cifar100/')
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=cifar_train)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=cifar_test)

    return train_dataset, test_dataset


################################################
def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return


### For Every dataloader, works like seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_loaders(seed, ds):
    if ds == 'c10':
        BS = 64
        train_dataset, test_dataset = get_c10_dataset()
    elif ds == 'c100':
        BS = 128
        train_dataset, test_dataset = get_c100_dataset()

    # g = torch.Generator()
    # g.manual_seed(seed)
    # torch.manual_seed(seed)
    seed_all(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BS, shuffle=True, num_workers=4,
                                               worker_init_fn=seed_worker)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BS, shuffle=False, num_workers=4)
    return train_loader, test_loader


########################################################
########################################################

# EPOCHS = 50
# criterion = nn.CrossEntropyLoss()


## Following is adapted from
### https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# Training
def train(epoch, model, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    # for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # break ## temporarily to debug

    loss = train_loss / (batch_idx + 1)
    acc = 100. * correct / total
    return loss, acc


# best_acc = -1
def test(epoch, model, optimizer, best_acc, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    latency = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            start = time.time()
            outputs = model(inputs)
            ttaken = time.time() - start

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            latency.append(ttaken)

            # break ## temporarily to debug

    loss = test_loss / (batch_idx + 1)
    # acc = 100. * correct / total

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc

    return loss, acc, best_acc, latency


########################################################
########################################################

### Do all experiments in repeat

SAVE_PATH = "./logs"

def check_if_training_completed(model_name, epochs):
    file = f'{SAVE_PATH}/{model_name}_stats.json'
    if not os.path.exists(file):
        print(f"! - Saved model not found: {model_name}")
        return False

    with open(file, 'r') as f:
        data = json.load(f)
    if len(data["test_acc"]) >= epochs:
        print(f"! - Saved model found: {model_name}, Escape !")
        return True
    else:
        print(f"! - Saved model found: {model_name}, Incomplete Training !")
        return False


def train_model(model, lr, model_name, dataset, epochs=200, seed=0, optim="Adam"):
    ## This will be updated in each train_model call.. so no problem
    global criterion, train_loader, test_loader

    if check_if_training_completed(model_name, epochs):
        print(f"!!\nEscaping: {model_name}")
        return None, None

    train_loader, test_loader = get_data_loaders(seed, dataset)

    best_acc = -1
    model = model.to(device)

    # torch._dynamo.config.verbose = False ## still verbose
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Begin Training for {model_name}")
    stats = {'num_param': -1, 'macs': -1, 'latency': [],
             'train_acc': [], 'train_loss': [],
             'test_acc': [], 'test_loss': []
             }
    latencies = []

    for epoch in tqdm(range(epochs)):
        trloss, tracc = train(epoch, model, optimizer)
        teloss, teacc, best_acc, laten = test(epoch, model, optimizer, best_acc, model_name)
        scheduler.step()

        latencies += laten
        stats['train_acc'].append(tracc)
        stats['test_acc'].append(teacc)
        stats['train_loss'].append(trloss)
        stats['test_loss'].append(teloss)

        #     print()
        latency = np.array(latencies)
        mu, std = np.mean(latency), np.std(latency)
        min, max = np.min(latency), np.max(latency)
        stats['latency'] = {'mean': mu, 'std': std, 'min': min, 'max': max}

        if epoch == epochs -1 :
            model.eval()
            macs, _n_params = get_model_complexity_info(model, (3, 32, 32),
                                                        # as_string=True,
                                                        # ignore_modules = ['channel_change'],
                                                        print_per_layer_stat=False,
                                                        verbose=False)
            n_params = sum(p.numel() for p in model.parameters())
            print('{:<30}  {:<8}'.format('MACs', macs))
            print('{:<30}  {:<8}'.format('params', n_params), _n_params)
            stats['num_param'] = n_params
            stats['macs'] = macs

        ### Save stats of the model
        with open(f'{SAVE_PATH}/{model_name}_stats.json', 'w') as f:
            json.dump(stats, f)
    print("Training Completed")
    return stats, best_acc


########################################################
########################################################

#  -----------------------------------------------------
