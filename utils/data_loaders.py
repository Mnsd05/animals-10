import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

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
        #   num_workers=num_workers,
          pin_memory=True)
    return dataloader_dict
