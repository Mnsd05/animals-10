from training.engine import train, train_step
import torch
import time
import math
import os
import optuna
from importlib import reload
import training.hyperparams
reload(training.hyperparams)
from training.hyperparams import hyperparams
from dataloader import CustomDataLoader

DRIVE_PROJECT_PATH = "/kaggle/working"
DB = "tuning_db"

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

def optimizer_tuning(model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            optimizer_name: str,
            train_dataloader: CustomDataLoader,
            val_dataloader: CustomDataLoader,
            device: torch.device,
            label_names: list,
            writer: torch.utils.tensorboard.writer):

    initial_model_state = model.state_dict()

    def objective(trial):
        # Reset model
        model.load_state_dict(initial_model_state)
        lr_range = hyperparams["optimizers"][optimizer_name]["lr"]
        b1_range = hyperparams["optimizers"][optimizer_name]["beta1"]
        b2_range = hyperparams["optimizers"][optimizer_name]["beta2"]

        learning_rate = trial.suggest_float("lr", *lr_range, log=True)
        b1 = trial.suggest_float("beta1", *b1_range)
        b2 = trial.suggest_float("beta2", *b2_range)

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(b1, b2))
        elif optimizer_name == "nadam":
            optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(b1, b2))

        num_epochs = hyperparams["num_epochs"]
        results = train(model, train_dataloader, val_dataloader, optimizer, loss_fn, device, label_names, writer, num_epochs)
        return results["best_acc"]

    sampler = optuna.samplers.QMCSampler()
    db_folder = f"{DRIVE_PROJECT_PATH}/{DB}"
    # Create folder if it doesn't exist
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)
    storage = f"sqlite:///{db_folder}/{optimizer_name}.db"
    study = optuna.create_study(direction="maximize", sampler=sampler, 
                                storage=storage, study_name=optimizer_name, load_if_exists=True)
    study.optimize(objective, n_trials=hyperparams["num_trials"])
    trial = study.best_trial

    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    return study
