import os
import numpy as np
import random
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

def read_files(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return torch.from_numpy(data[b"data"]), data[b"labels"]

class Cifar10(Dataset):
    def __init__(self, root_dir="./cifar-10", transform=None, target_transform=None, watermark_num_classes=1, seed=42):
        tmp_df = []
        for batch_file in os.listdir(os.path.abspath(root_dir)):
            if (batch_file.find("data") == 0) or (batch_file.find("test_batch") == 0):
                print(f"Reading {batch_file}...")
                features, labels = read_files(os.path.join(os.path.abspath(root_dir), batch_file))
                tmp_list = [{"image": features[i, :].numpy(), "label": labels[i]} for i in range(len(labels))]
                tmp_df.extend(tmp_list)
        self.df = pd.DataFrame(tmp_df)
        text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.classes = {k: v for k, v in enumerate(text_labels)}
        random.seed(seed)
        self.watermark_targets = random.sample(list(self.classes.keys()), watermark_num_classes)
        self.transform = transforms.Compose(transform)
        self.target_transform = transforms.Compose(self._prepend_transforms(target_transform, False))
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_row = self.df.iloc[idx].values
        # Change image into tensor of form (C, H, W)
        image = item_row[0].reshape((3, 32, 32)).transpose((1, 2, 0))
        if item_row[1] in self.watermark_targets:
            print(image)
            # Save image into orig_image dir
            # Watermark image and set this as the image we want to return
            # Save watermarked image into watermark_image dir
        image = self.transform(image)
        idx_label = item_row[1]
        label = np.zeros(10)
        label[idx_label] = 1
        label = torch.tensor(self.target_transform(label))
        return image, label
    
    def label_to_text(self, label):
        return self.classes[torch.argmax(label).item()]
    
    def _prepend_transforms(self, extra_transforms, for_img):
        orig_img_transform = [transforms.ToTensor()]
        new_transform = None
        if extra_transforms:
            if for_img:
                if (type(extra_transforms) == list):
                    new_transform = extra_transforms + orig_img_transform
                else:
                    new_transform = [extra_transforms] + orig_img_transform
            else:
                new_transform = extra_transforms if (type(extra_transforms) == list) else [extra_transforms]
            return new_transform
        else:
            return orig_img_transform if for_img else []
    
    def split_dataset(self, splits=(0.8, 0.1, 0.1)):
        train_data, val_data, test_data = random_split(self, splits)
        return train_data, val_data, test_data
    
    def get_dataloaders(self, splits=(0.8, 0.1, 0.1), batch_sizes=(64, 64, 64), do_shuffles=(True, True, True), num_workers=5, pin_memory=True):
        if not torch.accelerator.is_available():
            num_workers = 0
            pin_memory = False
        train_data, val_data, test_data = self.split_dataset(splits)
        train_dl = DataLoader(train_data, batch_size=batch_sizes[0], shuffle=do_shuffles[0], num_workers=num_workers, pin_memory=pin_memory)
        val_dl = DataLoader(val_data, batch_size=batch_sizes[1], shuffle=do_shuffles[1], num_workers=num_workers, pin_memory=pin_memory)
        test_dl = DataLoader(test_data, batch_size=batch_sizes[2], shuffle=do_shuffles[2], num_workers=num_workers, pin_memory=pin_memory)
        return train_dl, val_dl, test_dl