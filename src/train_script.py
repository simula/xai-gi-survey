import os
import random
import argparse
import pickle
import copy
import torch
import requests
import zipfile
import logging

import numpy as np

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from model import resnet18

random.seed(0)
np.random.seed(0)

argument_parser = argparse.ArgumentParser(description="This script trains a model")
argument_parser.add_argument("-o", "--output_dir", required=True, type=str)
argument_parser.add_argument("-t", "--train_dirpath", required=True, type=str)
argument_parser.add_argument("-v", "--valid_dirpath", required=True, type=str)
argument_parser.add_argument("-n", "--n_epochs", default=50, type=int)
argument_parser.add_argument("-l", "--lr", default=0.0006, type=float)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_and_unzip(url, directory):
    if not os.path.exists(directory):
        logging.info(f"Directory {directory} does not exist. Downloading zip file...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            zip_path = "temp.zip"
            with open(zip_path, "wb") as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(directory)
            logging.info(f"Downloaded and extracted zip file to {directory}")
            os.remove(zip_path)
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download the zip file: {e}")
    else:
        logging.info(
            f"Directory {directory} already exists. No need to download the zip file."
        )


def train(model, dataloaders, criterion, optimizer, model_path, num_epochs, lr):
    model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    best_wts, best_acc, val_acc_history = copy.deepcopy(model.state_dict()), 0.0, []

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}\n{"-"*10}')

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc, best_wts = epoch_acc, copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_path)

    logging.info(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_wts)
    return model, val_acc_history


def train_model(output_dir, train_dirpath, valid_dirpath, n_epochs=50):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_save_path = os.path.join(output_dir, "last_epoch_model.pt")
    history_path = os.path.join(output_dir, "history.pkl")

    model = resnet18()

    optimizer = torch.optim.Adam(lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_transforms = [
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    valid_transforms = [transforms.Resize(size=(224, 224)), transforms.ToTensor()]

    train_dataset = datasets.ImageFolder(train_dirpath, train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dirpath, valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    loaders = {"train": train_loader, "val": valid_loader}

    model, history = train(
        model=model,
        dataloaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        model_path=model_save_path,
        num_epochs=n_epochs,
        lr=lr,
    )

    torch.save(model.state_dict(), model_save_path)

    with open(history_path, "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":

    args = argument_parser.parse_args()

    output_dir = args.output_dir
    train_dirpath = args.train_dirpath
    valid_dirpath = args.valid_dirpath
    n_epochs = args.n_epochs
    lr = args.lr

    train_model(
        output_dir=output_dir,
        train_dirpath=train_dirpath,
        valid_dirpath=valid_dirpath,
        n_epochs=n_epochs,
        lr=lr,
    )
