#=========== Unlock this while running in Colab=======
#!pip install torchprofile
#!pip install matplotlib
#!pip install tqdm
#======================================================
import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import Union, List
from plotter import *
from Statistics import *

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from torchvision.datasets import  ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys


print(sys.version)
print("Numpy version:", np.__version__)

#========================= BASIC SETTINGS ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#-------------------------------------------------------------------






#========================= Prepare Data ============================

transform_function=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.390, 0.350, 0.310],  std=[0.010, 0.0200, 0.0190]),
    ])

train_data=CIFAR10(root='/home/subhransu/Documents/data/CIFAR10',
                train=True,
                transform=transform_function,
                download=True,)

test_data=CIFAR10(root='/home/subhransu/Documents/data/CIFAR10',
                train=False,
                transform=transform_function,
                download=True,)

train_loader=torch.utils.data.DataLoader(dataset=train_data, batch_size=32,  shuffle= True, pin_memory=True)
test_loader=torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True, pin_memory=True)

print("\n\n =============================================\n")
print(" Number of Training Samples = " ,len(train_data))
print(" Number of Test Samples     = " ,len(test_data))
print(" Classes: " , train_data.classes)
print(" ==============================================\n\n")

#--------------------------------------------------------------------






#************************* Model ************************************

class Vgg(nn.Module):
    ARCH = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',  512, 512, 'M']
    #ARCH = [64, 64, 'M', 64, 'M'] #=> Change line 114 :Replace 512 with 64
    def __init__(self) -> None:
        super().__init__()
        layers=[]
        counts =defaultdict(int)

        def add(name: str, layer: nn.Module)->None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name]+=1

        in_channel = 3

        for x in self.ARCH:
            if x != 'M':
                add("conv", nn.Conv2d(in_channel, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channel = x
            else:
                add("MAX_POOL", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
#--------------------------------------------------------------------



#************************* Training *********************************

def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    callbacks=None
) -> None:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_flops=0

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #-----------FLOP---------------------
        flops = profile_macs(model, inputs)
        total_flops += flops

    scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * (correct / total)

    return epoch_loss, epoch_acc, total_flops

#--------------------------------------------------------------------



#************************* Inference *********************************

@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, verbose=True) -> Union[float, float, int]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_flops=0

    for inputs, targets in tqdm(dataloader, desc='eval', leave=False, disable=not verbose):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        #-----------FLOP---------------------
        flops = profile_macs(model, inputs)
        total_flops += flops

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * (correct / total)

    return epoch_loss, epoch_acc, total_flops
#--------------------------------------------------------------------
# Basically our model ends here, Now plottting and Model Statistics
#--------------------------------------------------------------------





#*********************** Main Function ******************************
if __name__ == "__main__":
    
    
    model = Vgg().to(device)
    
    train_flop=[]
    valid_flop=[]

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.90 ** epoch)
    plot = plotter()


    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc, t_flop = train(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc, v_flop = evaluate(model, test_loader, criterion)
        

        plot.AddValue(train_loss, train_acc, val_loss, val_acc) #==== Plotter only======

        train_flop.append(t_flop)
        valid_flop.append(v_flop)



    plot.ShowFig()
    stat = Statistics(model=model, data_width=32, train_flops=train_flop, valid_flops=valid_flop)




# -------------------------Actual Statics ------------------------

#--------------------------------------------------------------------
