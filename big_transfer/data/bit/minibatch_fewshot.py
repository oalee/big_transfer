

import torchvision as tv
from torch.utils import data
import torch
import torch.utils.data
from .fewshot import find_fewshot_indices

def mini_batch_fewshot(
    train_set: data.Dataset,
    valid_set: data.Dataset,
    examples_per_class: int,
    batch: int,
    batch_split: int,
    workers: int,

):
    """Returns train and validation datasets."""
   

    if examples_per_class is not None:
        print(f"Looking for {examples_per_class} images per class...")
        indices = find_fewshot_indices(train_set, examples_per_class)
        train_set = torch.utils.data.Subset(train_set, indices=indices)

    print(f"Using a training set with {len(train_set)} images.")
    print(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = batch // batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=micro_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=micro_batch_size,
            num_workers=workers,
            pin_memory=True,
            sampler=torch.utils.data.RandomSampler(
                train_set, replacement=True, num_samples=micro_batch_size
            ),
        )

    return train_set, valid_set, train_loader, valid_loader

