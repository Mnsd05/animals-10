from tqdm import tqdm
from functools import partial
import torch
import torchvision
from typing import Dict, List, Tuple
from torch.utils import tensorboard
from pathlib import Path
from dataloader import CustomDataLoader
import time
import math
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

tqdm = partial(tqdm, position=0, leave=True)
DRIVE_PROJECT_PATH = "/kaggle/working"
MODEL_FOLDER = "models"

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch_num: int) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

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
    count = 1
    for batch, (X, y) in enumerate(pbar := tqdm(dataloader, desc = "Epoch " + str(epoch_num), unit = "batch")):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        avg_train_loss = train_loss / count
        avg_train_acc = train_acc / count
        pbar.set_postfix({"Loss": f"{avg_train_loss:.4f}", "Accuracy": f"{avg_train_acc:.4f}"})
        count += 1

    return avg_train_loss, avg_train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              label_names: List[str],
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a test dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of test loss and test accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    global_pred_labels = []
    global_true_labels = []
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

            global_pred_labels.extend(test_pred_labels.cpu().numpy())
            global_true_labels.extend(y.cpu().numpy())

    # Calculate the classification report and confusion matrix
    classification_rep = classification_report(global_true_labels, global_pred_labels, 
                                               labels=range(len(label_names)), target_names=label_names, zero_division=0)
    conf_matrix = confusion_matrix(global_true_labels, global_pred_labels, labels=range(len(label_names)))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc, classification_rep, conf_matrix

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          device: torch.device,
          label_names: List[str],
          writer: torch.utils.tensorboard.writer,
          epochs: int,
          checkpoint: dict = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and test the model
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

    Returns:
        A dictionary of training and test loss as well as training and
        test accuracy metrics. Each metric has a value in a list for
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
    results = {}
    early_threshold = 5
    best_acc = 0
    best_epoch = 0
    start_epoch = 1

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["saved_epoch"] + 1
        best_epoch = checkpoint["best_epoch"]
        best_acc = checkpoint["best_acc"]

    # Loop through training and test steps for a number of epochs
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device,
                                            epoch_num=epoch)
        test_loss, test_acc, classification_rep, conf_matrix  = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            label_names=label_names,
            device=device)

        # Print out what's happening
        print(
            f"Epoch {epoch}: "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["best_epoch"] = best_epoch
        results["best_acc"] = best_acc
        results["saved_epoch"] = epoch

        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Loss",
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

        # Add accuracy results to SummaryWriter
        writer.add_scalars(main_tag="Accuracy",
                           tag_scalar_dict={"train_acc": train_acc,
                                            "test_acc": test_acc},
                           global_step=epoch)
        
        # Add classification report to SummaryWriter
        writer.add_text(tag="Classification Report",
                        text_string=classification_rep,
                        global_step=epoch)
        
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=label_names)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix")
        disp.figure_.set_size_inches(10, 8)
        # Add confusion matrix to SummaryWriter
        writer.add_figure(tag="Confusion Matrix",
                          figure=disp.figure_,
                          global_step=epoch)

        # Track the PyTorch model architecture
        writer.add_graph(model=model,
                         # Pass in an example input
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))
        # Save the model at the end of each epoch for fault tolerance
        save_model(model=model,
                   optimizer=optimizer,
                   results=results,
                   target_dir=os.path.join(DRIVE_PROJECT_PATH, MODEL_FOLDER),
                   model_name=f"latest_{model.name}.pth")
        
        # Save the best model based on test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(model=model,
                   optimizer=optimizer,
                   results=results,
                   target_dir=os.path.join(DRIVE_PROJECT_PATH, MODEL_FOLDER),
                   model_name=f"best_{model.name}.pth")
        
        # Early stopping
        if epoch - best_epoch > early_threshold:
            print(f"Early stopping at epoch {epoch}")
            break

    # Close the writer
    writer.close()
    # Return the filled results at the end of the epochs
    return results


def save_model(model: torch.nn.Module,
               optimizer, 
               results: dict,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save({
        'saved_epoch': results["saved_epoch"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_epoch': results["best_epoch"],
        'best_acc': results["best_acc"],
    },
    f=model_save_path)

def batchsize_tuning(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: CustomDataLoader,
               device: torch.device,
               writer: torch.utils.tensorboard.writer):
    batch_size = 8
    min_time = float('inf')
    num_workers = os.cpu_count()
    initial_model_state = model.state_dict()
    initial_optimizer_state = optimizer.state_dict()
    while True:
        dataloaders = dataloader.create_data_loaders(batch_size, num_workers)
        val_dataloader = dataloaders[1]
        start = time.time()
        train_step(model, val_dataloader, loss_fn, optimizer, device, 0)
        end = time.time()
        time_taken = end - start
        writer.add_scalar("batch_size_tuning log2", time_taken, math.log2(batch_size))
        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optimizer_state)
        print(f"Batch size: {batch_size}, Time taken: {time_taken}")
        if time_taken < min_time:
            min_time = time_taken
            batch_size *= 2
        else:
            break
    return dataloaders
    
