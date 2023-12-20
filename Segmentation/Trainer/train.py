import json
from collections import defaultdict
from datetime import datetime, timedelta
import os

import torch
from torch.cuda import OutOfMemoryError
from tqdm import tqdm

from Segmentation.classification_metrics.MetricsHandler import MetricsHandler
from Segmentation.utils import get_loaders, save_state, calculate_metrics, append_metrics, save_metrics, save_losses, \
    save_metadata, TrainOutOfMemoryException, load_state
from time import time


def __forward(data, targets, model, loss_fn):
    i = "AAA0"
    try:
        # forward
        with torch.cuda.amp.autocast():
            i = "AAA1"
            predictions = model(data)
        i = "AAA2"
        loss = loss_fn(predictions, targets)
        i = "AAA3"
        return loss

    except OutOfMemoryError:
        print("Error = {}".format(i))
        print(torch.unique(torch.floor(data)))
        torch.save(data, 'error_data.pt')
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&")



def __backward(loss, optimizer, scaler):
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def train_fn(loader, model, optimizer, loss_fn, scaler):
    i = 0
    loop = tqdm(loader)
    i = 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    i = 2
    loss_accumulated = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # forward
        loss = __forward(data, targets, model, loss_fn)

        # backward
        __backward(loss, optimizer, scaler)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # Increment loss accumulation
        loss_accumulated += loss.item()

    return loss_accumulated / len(loader.dataset)


def validate(loader, model, loss_fn, device=torch.device("cuda")):
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
    return loss_accumulated / len(loader.dataset)


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
        pin_memory=True,
        evaluation_thresholds=None,
        starting_date = None
):
    if evaluation_thresholds is None:
        evaluation_thresholds = [0.5]
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
    current_date = datetime.now().strftime("%d-%m-%Y_%H-%M") if starting_date is None else starting_date

    if starting_date is not None:
        path =  os.path.join("results", current_date)
        files = [f for f in os.listdir(path) if f.startswith("epoch_")]
        initial_epoch = int(files[-1].replace("epoch_", ""))
        output_data_path = os.path.join("results", current_date, files[-1])
        print(output_data_path)
        model.load_state_dict(torch.load(os.path.join(output_data_path, files[-1] + ".pth"))["state_dict"])
        print("Loading model trained on {}. Initializing from {}".format(current_date, files[-1]))

    training_losses = []
    validation_losses = []
    metrics = defaultdict(list)
    auroc_list = []

    # saving training metadata
    output_data_path = os.path.join("results", current_date)
    save_metadata(metadata, output_data_path)

    metrics_handler = MetricsHandler(val_loader, model, evaluation_thresholds, device)

    for epoch in range(initial_epoch, epochs):

        print("Epoch {} of {}".format(epoch + 1, epochs))
        print("Cuda memory allocated : {}%".format(
            torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100))
        # print("Cuda summary: {}".format(torch.cuda.memory_summary()))

        # validation
        val_loss = validate(val_loader, model, loss_fn)
        validation_losses.append(val_loss)

        # train
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        training_losses.append(loss)

        print("Mean loss equals {}".format(loss))

        # calculating classification_metrics
        print("Calculating classification_metrics...")
        # epoch_metrics = calculate_metrics(val_loader, model, device)
        epoch_metrics, auroc = metrics_handler.evaluate()
        print("Appending classification_metrics...")
        # metrics = append_metrics(metrics, epoch_metrics)
        metrics = metrics_handler.append_metrics(metrics, epoch_metrics)
        auroc_list.append(auroc)

        # save checkpoint
        if epoch == epochs - 1 or save_all_states or (epoch + 1) % 10 == 0:
            output_data_path = os.path.join("results", current_date, "epoch_" + str(epoch + 1))
            print("Epoch {} saving checkpoint".format(epoch + 1))
            print("Current duration of training [hh:mm:ss]: {}".format(timedelta(seconds=time() - start)))
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_state(checkpoint, os.path.join(output_data_path), "epoch_" + str(epoch + 1) + ".pth")

            metrics_handler.save_metrics(metrics, auroc_list, output_data_path)

            save_losses(training_losses, validation_losses, output_data_path)


        print("")

    print("Training has been finished.")
    print("")

