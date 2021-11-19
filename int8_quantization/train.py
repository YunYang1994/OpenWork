#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2020-04-14 23:53:21
#   Description :
#
#================================================================

import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

#------------------------------------- define VGG16 model --------------------------#

class VGG16(nn.Module):

    def __init__(self, num_class):
        super(VGG16, self).__init__()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3,    64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,   64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)


        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
        nn.Linear(in_features=12544, out_features=2048, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048,  out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=256,   out_features=num_class, bias=True)
    )
        self.weights_init()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def weights_init(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#------------------------------------- define DataLoader ---------------------------#


class CifarDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = glob.glob(root_dir + "/*.jpg")
        self.transform = transforms.Compose([
            transforms.Resize(size=(112,112), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = int(image_path.split("/")[-1].split("_")[0])
        image = Image.open(image_path)
        image = self.transform(image)
        return {'image':image, 'label':label}

if __name__ == "__main__":

#------------------------------------- define TrainLoop ---------------------------#

    device        = torch.device("cuda:1")
    model         = VGG16(num_class=10).cuda(device)
    compute_loss  = nn.CrossEntropyLoss()
    optimizer     = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_dataset = CifarDataset("/data0/yyang/cifar10/train")
    train_loader  = DataLoader(train_dataset, 256, shuffle=True, num_workers=4)

    for epoch in range(1, 31):
        model.train()

        loss_value = 0.
        acc_value  = 0.
        num_batch  = 0.

        with tqdm(total=len(train_loader), desc="Epoch %2d/30" %epoch) as pbar:
            for sample in train_loader:
                image  = sample['image'].to(device)
                label  = sample['label'].to(device)

                optimizer.zero_grad()

                output  = model(image)
                loss    = compute_loss(output, label)
                _, pred = output.max(1)
                correct = pred.eq(label)

                loss.backward()
                optimizer.step()

                num_batch  += 1
                loss_value += loss.item()
                acc_value  += correct.sum().item() / label.size(0)

                pbar.set_postfix({'loss': '%.4f' %(loss_value / num_batch),
                                'acc':  '%.4f' %(acc_value  / num_batch)})
                pbar.update(1)
        if epoch > 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001



#------------------------------------- define TestLoop ---------------------------#

    test_dataset = CifarDataset("/data0/yyang/cifar10/test")
    test_loader  = DataLoader(test_dataset, 256, shuffle=True, num_workers=4)

    model.eval()

    acc_value  = 0.
    num_batch  = 0.

    with torch.no_grad():
        for sample in test_loader:
            image   = sample['image'].to(device)
            label   = sample['label'].to(device)

            output  = model(image)
            _, pred = output.max(1)
            correct = pred.eq(label)

            num_batch  += 1
            acc_value  += correct.sum().item() / label.size(0)

    acc_value  = acc_value / num_batch
    torch.save(model.state_dict(), 'VGG16-testAcc=%.4f.pth' %acc_value)


