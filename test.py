import os, argparse

import torch
from torch.utils.data import random_split

import torchvision
from torch.utils.data.dataloader import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from cnn_valid_receipt import CnnValidReceipt
from cnn_base_recognition import BaseImageRecognition

from cnn_base_recognition import fit

import utils


'''
    ARG PARSER INIT
'''
parser = argparse.ArgumentParser()


'''
    CONSTANTS
'''
NUM_EPOCHS = 10
NUM_WORKERS = 6
OPT_FUNC = torch.optim.Adam
LEARNING_RATE = 0.001
BATCH_SIZE = 128


MODEL_SAVE_FOLDER = "models/"
DATASET_FOLDER = "images/"


def train(model: BaseImageRecognition, dataset_folder_path: str, epochs=NUM_EPOCHS, plot=False):
    dataset = ImageFolder(dataset_folder_path, transform=transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]))

    val_size = len(dataset) // 2
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_dl = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_data, BATCH_SIZE*2, num_workers=NUM_WORKERS, pin_memory=True)

    print("Batch Count:\n\t- Train: {}. \n\t- Validate: {}.\n".format(len(train_dl), len(val_dl)))
    print("Epoch Count: {}.".format(epochs))

    history = fit(epochs, LEARNING_RATE, model, train_dl, val_dl, OPT_FUNC)

    if plot:
        utils.plot_accuracies(history)
        

def test(model: BaseImageRecognition, folder, image="", plot=False):
    dataset = ImageFolder(folder, transform=transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()]))
    test_dl = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    if not image == "":
        utils.eval_image(model, image, dataset)
    else:
        utils.eval_images(model, test_dl, dataset)


def main(args):

    model = CnnValidReceipt()
    if os.path.exists(MODEL_SAVE_FOLDER):
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_FOLDER, "model{}.pth".format("03"))))

    if args.Train:
        train(model, os.path.join(DATASET_FOLDER, "train"), epochs=args.Epochs, plot=args.Plot)
            

    if args.Test:
        test(model, folder=os.path.join(DATASET_FOLDER, "test"))

    if args.Eval:
        test(model, folder=os.path.join(DATASET_FOLDER, "test"), image=args.Eval)

    model.save(os.path.join(MODEL_SAVE_FOLDER, "model{}.pth".format("03")))
    

if __name__ == '__main__':
    parser.add_argument("-v", "--Verbose", help="Displays Output Informations to console.", action="store_false")
    parser.add_argument("-p", "--Plot", help="Defines if there should be visual ploting outputs.", action="store_true")
    parser.add_argument("--Test", help="Defines if the model should perform a testing cycle.", action="store_true")
    parser.add_argument("--Train", help="Defines if the model should perform a training cycle.", action="store_true")
    parser.add_argument("-o", "--Output", help="Streams Output Informations to file. ")
    parser.add_argument("--TestFolder", help="Defines Test Dataset Folder Path.")
    parser.add_argument("--TrainFolder", help="Defines Train Dataset Folder Path.")
    parser.add_argument("-e", "--Epochs", help="Defines Epoch Count.", type=int, default=NUM_EPOCHS, action="store")
    parser.add_argument("--Eval", help="Evaluates a single image.", type=str)
    args = parser.parse_args()

    main(args)