import argparse
import os
import time
from typing import Dict, List, Tuple

import torch
import wandb
import yaml
from torch.utils import data
from torchvision import datasets
from tqdm.auto import tqdm

from model_transforms import create_effnetb2_model


def train_step(epoch: int,
               model: torch.nn.Module,
               dataloader: data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = "cpu",
               disable_progress_bar: bool = False) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    epoch (int)
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    disable_progress_bar (Boolean): disable progress bad. Defaults to False

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Training Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    for batch, (x, y) in progress_bar:
        # Send data to target device
        x, y = x.to(device), y.to(device)

        # 1. Forward pass
        y_predict = model(x)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_predict, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_predicted_class = torch.argmax(torch.softmax(y_predict, dim=1), dim=1)
        train_acc += (y_predicted_class == y).sum().item() / len(y_predict)

        # Update progress bar
        progress_bar.set_postfix(
            {
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1),
            }
        )

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(epoch: int,
              model: torch.nn.Module,
              dataloader: data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = "cpu",
              disable_progress_bar: bool = False) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Loop through data loader data batches
    progress_bar = tqdm(
        enumerate(dataloader),
        desc=f"Testing Epoch {epoch}",
        total=len(dataloader),
        disable=disable_progress_bar
    )

    # Turn on inference context manager
    with torch.no_grad():  # no_grad() required for PyTorch 2.0, I found some errors with `torch.inference_mode()`,
        # Loop through DataLoader batches
        for batch, (X, y) in progress_bar:
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_predicted_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_predicted_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_predicted_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "test_loss": test_loss / (batch + 1),
                    "test_acc": test_acc / (batch + 1),
                }
            )

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: data.DataLoader,
          test_dataloader: data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device = "cpu",
          model_path: str = "",
          disable_progress_bar: bool = False) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    model_path: The path where the model to be saved
    disable_progress_bar (Boolean): disable progress bad. Defaults to False

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               "train_epoch_time": [],
               "test_epoch_time": []
               }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs), disable=disable_progress_bar):
        # Perform training step and time it
        train_epoch_start_time = time.time()
        train_loss, train_acc = train_step(epoch=epoch,
                                           model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           disable_progress_bar=disable_progress_bar)
        train_epoch_end_time = time.time()
        train_epoch_time = train_epoch_end_time - train_epoch_start_time

        # Perform testing step and time it
        test_epoch_start_time = time.time()
        test_loss, test_acc = test_step(epoch=epoch,
                                        model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device,
                                        disable_progress_bar=disable_progress_bar)
        test_epoch_end_time = time.time()
        test_epoch_time = test_epoch_end_time - test_epoch_start_time
        torch.save({
            "EPOCH": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_fn,
        },
            model_path
        )

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"train_epoch_time: {train_epoch_time:.4f} | "
            f"test_epoch_time: {test_epoch_time:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_epoch_time"].append(train_epoch_time)
        results["test_epoch_time"].append(test_epoch_time)

    return results


def read_params() -> dict:
    """Reads in a YAML configuration file.
    Returns:
        dict: dictionary of YAML configuration file.
    """
    with open("config/config.yaml") as file:
        config = yaml.safe_load(file)["params"]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new model')
    parser.add_argument('--model', type=str, help='model path', default=read_params()['model_path'])
    parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--model_path', type=str, help="model's path", default='models')
    parser.add_argument('--progress_bar', type=bool, help="disable training progress bar", default=False)

    args = parser.parse_args()

    model, transforms = create_effnetb2_model()

    wandb.login()

    wandb.init(project="image_classification",config=read_params())
    wandb.watch(model)

    train_dataset = datasets.ImageFolder(
        root=read_params()["train_data_path"],
        transform=transforms,
        target_transform=None
    )

    test_dataset = datasets.ImageFolder(
        root=read_params()["test_data_path"],
        transform=transforms,
    )

    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
            )

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
        loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=args.epochs,
        device=args.device,
        disable_progress_bar=args.progress_bar,
        model_path=args.model
    )

    wandb.log(results)

