import torch
from torch.utils.data.dataloader import DataLoader, Dataset

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

from cnn_base_recognition import BaseImageRecognition

import matplotlib.pyplot as plt

from PIL import Image


def show_batch(dl: DataLoader):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()

def eval_image(model: BaseImageRecognition, img_path: str, train_ds: Dataset):
    img = Image.open(img_path)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Resize((150, 150))])
    
    img_normalized = transform_norm(img).float()
    img_normalized = img_normalized.unsqueeze_(0)
    
    with torch.no_grad():
        model.eval()
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        classes = train_ds.classes
        class_name = classes[index]
        print(class_name)
    
def eval_images(model: BaseImageRecognition, test_dl: DataLoader, dataset: Dataset):
    with torch.no_grad():
        classes = dataset.classes
        model.eval()

        for images, labels in test_dl:
            for image, label in zip(images, labels):
                output = model(image.unsqueeze_(0))
                index = output.data.cpu().numpy().argmax()
                classes = dataset.classes
                class_name = classes[index]
                print(class_name, classes[label])
