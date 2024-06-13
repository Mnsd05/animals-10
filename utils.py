
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter

def create_data_loaders(full_dataset, batch_size, num_workers, train_transform, test_transform):
    # stratify the dataset into train, validation and test sets
    valtrain_idx, test_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, 
                                            random_state=42, stratify=full_dataset.targets)
    train_idx, val_idx = train_test_split(valtrain_idx, test_size=0.2, random_state=42, 
                                        stratify=[full_dataset.targets[i] for i in valtrain_idx])
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = test_transform
    test_subset.dataset.transform = test_transform

    train_target = [full_dataset.targets[i] for i in train_idx]
    class_counts = Counter([target for target in train_target])
    sample_weights = [1 / class_counts[target] for target in train_target]
    sampler = WeightedRandomSampler(sample_weights, replacement = True,
                                    num_samples=int(len(train_target) * 1.5))
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, sampler = sampler,
                                num_workers = num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, num_workers = num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, num_workers = num_workers, pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader

def create_writer(experiment_name: str, 
                  model_name: str,
                  log_folder: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    if extra:
        # Create log directory path
        log_dir = os.path.join(log_folder, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(log_folder, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
