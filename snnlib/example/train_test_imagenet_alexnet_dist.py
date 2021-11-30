'''
training:  python -m torch.distributed.launch --nproc_per_node=4 train_test_imagenet_alexnet_dist.py --if_train train
training:  python -m torch.distributed.launch --nproc_per_node=1 train_test_imagenet_alexnet_dist.py --if_train test 
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from models import Alexnet_BN_DIST

from apex.parallel import DistributedDataParallel as DDP

import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--if_train', type=str, default='train')
args = parser.parse_args()

torch.distributed.init_process_group('nccl',init_method='env://')
torch.cuda.set_device(args.local_rank)
print(args.local_rank)

checkpoint = './ckpt/imagenet_alexnet_dist.pth.tar'
batch = 32


print('==> Preparing data..')
traindir = '/imagenet/train/data'
testdir = '/imagenet/test/data'

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])
train_dataset = torchvision.datasets.ImageFolder(root= traindir, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=False, num_workers=8,pin_memory=True,drop_last=True,sampler=train_sampler)

#256 224
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_set = torchvision.datasets.ImageFolder(root= testdir,transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=8,pin_memory=True)
print('==> Finish data loading')

snn = Alexnet_BN_DIST().cuda()
optimizer = torch.optim.SGD(snn.parameters(), lr=1e-1, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda()

snn = DDP(snn)

num_epochs = 90
best_acc = 0
best_epoch = 0

if args.if_train == 'test':
    num_epochs = 1
    snn.load_state_dict(torch.load(checkpoint,map_location='cuda:0'))


for epoch in range(num_epochs):
    if epoch == 30:
        optimizer = optimizer = torch.optim.SGD(snn.parameters(), lr=1e-2, momentum=0.9)
    if epoch == 60:
        optimizer = optimizer = torch.optim.SGD(snn.parameters(), lr=1e-3, momentum=0.9)

    if args.if_train == 'train':
        snn.train()
        snn.module.set_bn(True)
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

            if (i + 1) % 100 == 0 or (i + 1) == (len(train_dataset) // batch):
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f' % (
                epoch, num_epochs - 1, i + 1, len(train_dataset) // batch, running_loss / 300))
                running_loss = 0
                print('Time elasped:', time.time() - start_time)



    snn.eval()
    snn.module.set_bn(False)
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


