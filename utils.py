import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import os

def create_data_loaders(dataset, data_segments, split_rates, batch_size, num_workers):
    assert(sum(split_rates) == 1), "Split_rates not equals 1"
    assert(len(split_rates) == len(data_segments)), "Number of split rates and data segments do not match"

    generator = torch.Generator().manual_seed(42)
    splitted_data = random_split(dataset, split_rates, generator)
    dataloader_dict = {}
    for i in range(len(data_segments)):
        dataloader_dict[data_segments[i]] = DataLoader(
          splitted_data[i],
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers,
          pin_memory=True)
    return dataloader_dict

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
