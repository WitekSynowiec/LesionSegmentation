import json
from collections import defaultdict
from datetime import datetime, timedelta
import os

import torch
from tqdm import tqdm
from Segmentation.utils import get_loaders, save_state, calculate_metrics, append_metrics, save_metrics, save_losses, save_metadata, TrainOutOfMemoryException
from time import time


def __forward(data, targets, model, loss_fn):
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
    training_losses = []
    validation_losses = []
    metrics = defaultdict(list)

    # saving training metadata
    output_data_path = os.path.join("results", current_date)
    save_metadata(metadata, output_data_path)

    for epoch in range(epochs):
        try:
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

            # calculating metrics
            epoch_metrics = calculate_metrics(val_loader, model, device)
            metrics = append_metrics(metrics, epoch_metrics)

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

                save_metrics(metrics, output_data_path)

                save_losses(training_losses, validation_losses, output_data_path)
        except torch.cuda.OutOfMemoryError as e:
            print("Caught {} exception. Continuing from saved model".format(e))
            epoch = (epoch + 1) % 10
            torch.cuda.empty_cache()
            model.load_state_dict(torch.load(os.path.join(output_data_path, "epoch_" + str(epoch + 1) + ".pth")))

        print("")

    print("Training has been finished.")
    print("")

