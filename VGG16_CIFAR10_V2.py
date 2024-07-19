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
#--------------------------------------------------------------------





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

print("\n\n =============================================\n Number of Training Samples = " ,len(train_data))
print(" Number of Test Samples     = " ,len(test_data))
print(" Classes: " , train_data.classes)
print(" ==============================================\n\n")

#--------------------------------------------------------------------






#************************* Model ************************************

class Vgg(nn.Module):
    #ARCH = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',  512, 512, 'M']
    ARCH = [64, 64, 'M', 64, 'M']
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
        self.classifier = nn.Linear(64, 10)



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

    scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * (correct / total)

    return epoch_loss, epoch_acc

#--------------------------------------------------------------------



#************************* Inference *********************************

@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, verbose=True) -> Union[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc='eval', leave=False, disable=not verbose):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * (correct / total)

    return epoch_loss, epoch_acc
#--------------------------------------------------------------------
# Basically our model ends here, Now plottting and Model Statistics
#--------------------------------------------------------------------




#********************** Plotter Function ****************************

def plot_training_results(history: dict) -> None:
    epochs = range(len(history['train_loss']))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss [VGG 16 with CIFAR 10]')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy [VGG 16 with CIFAR 10]')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


# ------------Save and show the plot----------------------
#    save_path = '/home/subhransu/My Own Research/DASR/Model/Pruned Model/results/training_results.png'
#    plt.savefig(save_path, format='png', dpi=300)
#    plt.show()
#--------------------------------------------------------------------





#************************ Model Spec ********************************

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
#--------------------------------------------------------------------




#*********************** Main Function ******************************
if __name__ == "__main__":
    model = Vgg().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.90 ** epoch)
    plot = plotter()


    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        plotter.AddValue(train_loss, train_acc, val_loss, val_acc)
        


    plot.ShowFig()




# -------------------------Actual Statics ------------------------
    #dense_model_accuracy = evaluate(model, test_loader)
    dense_model_size = get_model_size(model)
    #print(f"dense model has accuracy={dense_model_accuracy:.2f}%")
    print(f"\n\n Dense model has size={dense_model_size/MiB:.2f} MiB")
#--------------------------------------------------------------------
