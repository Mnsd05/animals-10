from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter

class CustomDataLoader:
    def __init__(self, full_dataset):
        self.full_dataset = full_dataset
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def split_data(self, train_transform, test_transform):
        # stratify the dataset into train, validation and test sets
        valtrain_idx, test_idx = train_test_split(list(range(len(self.full_dataset))), test_size=0.1, 
                                                random_state=42, stratify=self.full_dataset.targets)
        train_idx, val_idx = train_test_split(valtrain_idx, test_size=0.2, random_state=42, 
                                            stratify=[self.full_dataset.targets[i] for i in valtrain_idx])
        train_subset = Subset(self.full_dataset, train_idx)
        val_subset = Subset(self.full_dataset, val_idx)
        test_subset = Subset(self.full_dataset, test_idx)

        train_subset.dataset.transform = train_transform
        train_subset.dataset.train_idx = train_idx
        val_subset.dataset.transform = test_transform
        test_subset.dataset.transform = test_transform

        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_subset
    
    def create_data_loaders(self, batch_size, num_workers):
        train_target = [self.full_dataset.targets[i] for i in self.train_dataset.dataset.train_idx]
        class_counts = Counter([target for target in train_target])
        sample_weights = [1 / class_counts[target] for target in train_target]
        sampler = WeightedRandomSampler(sample_weights, replacement = True,
                                        num_samples=int(len(train_target) * 1.5))

        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, sampler = sampler,
                                    num_workers = num_workers, pin_memory=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory=True)
        return train_dataloader, val_dataloader, test_dataloader
