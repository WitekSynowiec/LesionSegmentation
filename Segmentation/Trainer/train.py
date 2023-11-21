import json
from collections import defaultdict
from datetime import datetime, timedelta
import os

import torch
from tqdm import tqdm
from Segmentation.utils import get_loaders, save_state, calculate_metrics, append_metrics
from time import time


def __forward(data, targets,  model, loss_fn):
    # forward
    with torch.cuda.amp.autocast():
        predictions = model(data)
        # print(predictions.dtype)
        # print(data.dtype)
        # print(targets.dtype)
        loss = loss_fn(predictions, targets)
    return loss


def __backward(loss, optimizer, scaler):
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loss_accumulated = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)


        # forward
        loss = __forward(data, targets, model, loss_fn)
        # with torch.cuda.amp.autocast():
        #     predictions = model(data)
        #     loss = loss_fn(predictions, targets)

        # backward
        __backward(loss, optimizer, scaler)
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # Increment loss accumulation
        loss_accumulated += loss.item()

    return loss_accumulated/len(loader.dataset)


def validate(loader, model, loss_fn, device=torch.device("cuda") ):
    model.eval()
    loss_accumulated = 0
    model.to(device)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)
            loss = __forward(data, targets, model, loss_fn)
            loss_accumulated += loss.item()
    model.train()
    return loss_accumulated/len(loader.dataset)


def fit(
        model: torch.nn.Module,
        dataset=None,
        optimizer=None,
        loss_fn=None,
        scaler=None,
        metrics=None,
        batch_size=None,
        epochs=1,
        verbose='auto',
        callbacks=None,
        split_ratio=(1.0, 0.0),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        num_workers=1,
        save_all_states=False,
        use_multiprocessing=False,
        pin_memory=True
):
    metadata = locals()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} processor".format(device))
    if torch.cuda.is_available() and device == torch.device("cuda"):
        print("Using {} device.".format(torch.cuda.get_device_name(device)))

    model = model.to(device)

    train_loader, val_loader = get_loaders(
        dataset=dataset,
        batch_size=batch_size,
        split_ratio=split_ratio,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle
    )

    print("Initiating training process...")
    start = time()
    current_date = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_data_path = os.path.join("results", current_date)
    training_losses = []
    validation_losses = []
    metrics = defaultdict(list)

    for epoch in range(epochs):
        print("Epoch {} of {}".format(epoch+1, epochs))

        # validation
        val_loss = validate(val_loader, model, loss_fn)
        validation_losses.append(val_loss)

        # train
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        training_losses.append(loss)

        print("Mean loss equals {}".format(loss))

        # save model
        if epoch == epochs - 1 or save_all_states:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_state(checkpoint, output_data_path, "epoch_" + str(epoch + 1) + ".pth")

        metrics = append_metrics(metrics, calculate_metrics(val_loader, model, device))

    metadata['model'] = metadata['model'].__repr__()

    for key in ['dataset', 'optimizer', 'loss_fn', 'scaler']:
        metadata[key] = str(metadata[key].__class__.__name__)

    with open(os.path.join(output_data_path, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp)

    with open(os.path.join(output_data_path, 'losses.json'), 'w') as fp:
        json.dump({'training_losses': training_losses, 'validation_losses': validation_losses}, fp)

    with open(os.path.join(output_data_path, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)

    # save_metadata(metadata, output_data_path)

    print("")
    print("Duration of training [hh:mm:ss]: {}".format(timedelta(seconds=time() - start)))
    print("Training has been finished.")
    print("")

    # print some examples to a folder
    # save_predictions_as_imgs(
    #     val_loader, model, folder=r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/results/', device=device
    # )