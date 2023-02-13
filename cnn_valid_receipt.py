import torch
import torch.nn as nn
import torch.functional as func
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from cnn_base_recognition import BaseImageRecognition, fit

import matplotlib.pyplot as plt

class CnnValidReceipt(BaseImageRecognition):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(82944, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, xb):
        return self.network(xb)

def show_batch(dl: DataLoader):
    for images, labels in dl:
        print("Image Count:", len(images))
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        
        break

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()

train_data_dir = "images/train"
test_data_dir = "images/test"

def main():

    dataset = ImageFolder(train_data_dir, transform=transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]))
    #test_dataset = ImageFolder(test_data_dir, transform=transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]))

    model = CnnValidReceipt()

    num_epochs = 1
    opt_func = torch.optim.Adam
    lr = 0.001

    batch_size = 128
    val_size = 2000
    train_size = len(dataset) - val_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_dl = DataLoader(val_data, batch_size*2, num_workers=1, pin_memory=True)
    print("Show Batch")
    
    #show_batch(train_dl)

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_accuracies(history)

if __name__ == '__main__':
    main()
