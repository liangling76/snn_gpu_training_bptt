'''
training:  CUDA_VISIBLE_DEVICES=0 python train_test_cifar.py --if_train train
inference: CUDA_VISIBLE_DEVICES=0 python train_test_cifar.py --if_train test
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from models import CIFAR_BN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--if_train', type=str, default='train')
args = parser.parse_args()

checkpoint = './ckpt/cifar.pth.tar'
batch = 64

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_dataset   = torchvision.datasets.CIFAR10(root= '/cifar/train/data', train=True, download=True, transform=transform_train)
train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=0,drop_last =True)

transform_test = transforms.Compose([transforms.ToTensor()])
test_dataset   = torchvision.datasets.CIFAR10(root= '/cifar/test/data', train=False, download=True, transform=transform_test)
test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=0,drop_last =True)


snn = CIFAR_BN().cuda()
optimizer = torch.optim.SGD(snn.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda()

num_epochs = 80
best_acc = 0
best_epoch = 0

if args.if_train == 'test':
    num_epochs = 1
    snn.load_state_dict(torch.load(checkpoint))


for epoch in range(num_epochs):
    if epoch == 50:
        optimizer = optimizer = torch.optim.SGD(snn.parameters(), lr=1e-3, momentum=0.9)

    if args.if_train == 'train':
        snn.train()
        snn.set_bn(True)
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

            outputs = snn(images)
            loss = criterion(outputs, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()

            if (i + 1) % 300 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f' % (
                epoch, num_epochs - 1, i + 1, len(train_dataset) // batch, running_loss / 300))
                running_loss = 0
                print('Time elasped:', time.time() - start_time)


    snn.eval()
    snn.set_bn(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = snn(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.cpu().max(1)
            total += float(labels.size(0))

            correct += float(predicted.eq(labels.cpu()).sum().item())


    acc = 100. * float(correct) / float(total)
    print('Test Accuracy of the model on the test images: %.3f' % acc)
    if args.if_train == 'train':
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(snn.state_dict(), checkpoint)

        print('Best acc: ', best_acc, 'Best Epoch: ', best_epoch, '\n')


