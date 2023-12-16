import os

import numpy as np
import json
import torch
import torchvision
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Resize, InterpolationMode, Compose
from Segmentation.metrics import BinaryMetrics
from FilesHelpers.Nifti import Nifti
from Segmentation.dataset import MSDataset
from Segmentation.model import UNet

"""
This files contains methods to utilize Neural Network 
but also methods to support learning processes and visibility
"""

"""
Function returns dataloader for given dataset.
"""


def get_loaders(
        dataset,
        batch_size,
        split_ratio=(0.8, 0.2),
        num_workers=4,
        shuffle=True,
        pin_memory=True
):
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=list(split_ratio))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def save_state(state, path, filename):
    print("=> Saving state")
    # directory = os.path.join(path)
    os.makedirs(path, exist_ok=True)
    # print(os.listdir(directory))
    torch.save(state, os.path.join(path, filename))


def load_state(checkpoint, model):
    print("=> Loading state")
    model.load_state_dict(checkpoint["state_dict"])

def save_metrics(metrics: dict, path: os.path):
    print("=> Saving metrics")
    for key, value in metrics.items():
        with open(os.path.join(path, '{}.json'.format(key)), 'w') as fp:
            json.dump(value, fp)
def save_losses(training_losses, validation_losses, path: os.path):
    print("=> Saving losses")
    with open(os.path.join(path, 'losses.json'), 'w') as fp:
        json.dump({'training_losses': training_losses, 'validation_losses': validation_losses}, fp)

def save_metadata(metadata: dict[str, any], path: os.path):
    print("=> Saving metadata")
    os.makedirs(path, exist_ok=True)
    metadata['model'] = metadata['model'].__repr__()
    for key in ['dataset', 'optimizer', 'loss_fn', 'scaler']:
        metadata[key] = str(metadata[key].__class__.__name__)

    with open(os.path.join(path, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)

def calculate_metrics(loader, model, device=torch.device("cuda")):
    num_pixels = 0
    model.eval()

    metrics = BinaryMetrics()
    metrics.set_device(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            metrics.update(preds, y)
            num_pixels += torch.numel(preds)

    model.train()
    metrics.set_device(torch.device('cpu'))
    return metrics()


def append_metrics(to_append: dict, new_elements: dict):

    for key, value in new_elements.items():
        to_append[key].append(value.cpu().detach().numpy().tolist())

    return to_append



def save_predictions_as_imgs(
        loader, model, folder=os.makedirs(os.path.join("results", "images"), exist_ok=True), device=torch.device("cuda")
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()

def segment_nii(model: torch.nn.Module or None, nii_file: Nifti, batch_size: int = 1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    images = nii_file.get_data()[0]
    images = np.expand_dims(images, axis=1)
    masks_prediction = None

    for i in range(0, images.shape[0], batch_size):
        slices_batch = images[i:i + batch_size, ...]
        mask_prediction = torch.sigmoid(model(
            torch.from_numpy(slices_batch).float().to(
                device))).cpu().detach().numpy()
        if masks_prediction is not None:
            masks_prediction = np.concatenate((masks_prediction, mask_prediction), axis=0)
        else:
            masks_prediction = mask_prediction

    return masks_prediction


class TrainOutOfMemoryException(Exception):
    def __init__(self, last_epoch: int, output_path=os.path, message: str = ''):
        super().__init__(message)
        self.last_epoch = last_epoch
        self.output_path = output_path


if __name__ == "__main__":
    dataset = MSDataset(
        image_dir=r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images/',
        mask_dir=r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations/',
        transform=Compose([
            Resize(size=(128, 128), interpolation=InterpolationMode.BILINEAR, antialias=False)
        ]),
        target_transform=Compose([
            Resize(size=(128, 128), interpolation=InterpolationMode.NEAREST, antialias=False)
        ])
    )

    train_loader, val_loader = get_loaders(
        dataset=dataset,
        batch_size=16
    )
    modelx = UNet(initial_in_channels=1, initial_out_channels=1).to(torch.device("cuda"))

    # check_accuracy(val_loader, modelx, device=torch.device("cuda"))
