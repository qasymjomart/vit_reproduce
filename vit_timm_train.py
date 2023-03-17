"""

Based on implementation from https://github.com/kentaroy47/vision-transformers-cifar10

"""

import time
import os
import timm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

from randomaug import RandAugment
from utils import progress_bar

from models import Vision_Transformer

# Helpers
def get_n_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)

# Data
print('==> Preparing data..')
size = 32
DATA_PATH = '/home/qasymjomart/data/'
epochs = 400
use_pretrained = False

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

aug=True
# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))
    print('Using RandAugment')

# vit = timm.create_model('vit_base_patch16_384')
# vit.head = nn.Linear(vit.head.in_features, 10)
if size == 32:
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bs = 512
    custom_config = {
        "img_size": [3, size, size],
        "in_chans": 3,
        "patch_size": 4,
        "embed_dim": 512,
        "depth": 6,
        "n_heads": 8,
        "qkv_bias": True,
        "mlp_ratio": 2.0,
        "n_classes": 10,
        "p": 0.1, 
        "attn_p": 0.1
    }

    vit = Vision_Transformer(**custom_config)
    # vit.head = nn.Linear(vit.head.in_features, 10)

elif size == 384:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bs = 128
    custom_config = {
        "img_size": [3, size, size],
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "n_classes": 1000,
        "p": 0.0, 
        "attn_p": 0.0
    }

    vit = Vision_Transformer(**custom_config)
    vit.head = nn.Linear(vit.head.in_features, 10)

# vit = timm.create_model('vit_base_patch16_384')
# vit.head = nn.Linear(vit.head.in_features, 10)

if use_pretrained:
    net_official = timm.create_model("vit_base_patch16_384", pretrained=use_pretrained)
    for (n_0, p_0), (n_c, p_c) in zip(net_official.named_parameters(), vit.named_parameters()):
        assert p_0.numel() == p_c.numel(), n_c
        print(f"{n_0} | {n_c}")

        p_c.data[:] = p_0.data

    assert_tensors_equal(p_c.data, p_0.data)

# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    vit = torch.nn.DataParallel(vit) # make parallel
    torch.backends.cudnn.benchmark = True

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Loss is CE
criterion = nn.CrossEntropyLoss()
opt = 'adam'
lr = 1e-4
if opt == "adam":
    optimizer = optim.Adam(vit.parameters(), lr=lr)
elif opt == "sgd":
    optimizer = optim.SGD(vit.parameters(), lr=lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

use_amp = True
##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# usewandb = false
# if usewandb:
#     wandb.watch(net)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    vit.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = vit(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1)


##### Validation
def test(epoch):
    global best_acc
    vit.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vit(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)

    return test_loss, acc

list_loss = []
list_acc = []

vit.cuda()

for epoch in range(start_epoch, epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # print(list_loss)

