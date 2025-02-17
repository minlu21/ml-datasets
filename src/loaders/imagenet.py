import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms


def generate_mappings(filename="mapping.csv", orig_map_filepath="imagenet10K/synset_mapping.txt"):
    with open(filename, "w") as wf:
        wf.write("orig_lab,norm_lab\n")
        print(f"Generating {filename}...")
        with open(orig_map_filepath, "r") as rf:
            for line in rf:
                split_line = line.split(" ")
                orig_lab = split_line[0]
                norm_lab = split_line[1].strip(", \r\n")
                wf.write(f"{orig_lab},{norm_lab}\n")


<<<<<<<< HEAD:src/loaders/imagenet10k.py
def generate_dataset(img_root_dir="imagenet10K", mapping_file="mapping.csv", filename="imagenet10K.csv"):
========
def generate_dataset(mapping_file="mapping.csv", filename="imagenet10K.csv", dataset_dir="./"):
>>>>>>>> 3498b8da13b92b4e85320b02684619fe0a447ff1:src/loaders/imagenet.py
    df = pd.read_csv(mapping_file,names=["orig_lab", "norm_lab"], index_col=False)
    print("Generating ImageNet10K Dataset...")
    with open(filename, "w") as wf:
        wf.write("filename,filepath,str_label,idx_label\n")
<<<<<<<< HEAD:src/loaders/imagenet10k.py
        for filename in os.listdir(img_root_dir):
            f = os.path.join(os.path.abspath(img_root_dir), filename)
========
        for filename in os.listdir(os.path.join(dataset_dir, "imagenet10K")):
            f = os.path.join(os.path.abspath(os.path.join(dataset_dir, "imagenet10K")), filename)
>>>>>>>> 3498b8da13b92b4e85320b02684619fe0a447ff1:src/loaders/imagenet.py
            if not os.path.isfile(f):
                for subfilename in os.listdir(f):
                    subf = os.path.join(f, subfilename)
                    if os.path.isfile(subf):
                        dir_start = os.path.relpath(f).find(dataset_dir) + len(dataset_dir) + 1
                        orig_label = os.path.relpath(f)[dir_start + len("imagenet10K") + 1:]
                        str_label = df.loc[df["orig_lab"]==orig_label].values[0][1]
                        idx_label = df.loc[df["orig_lab"]==orig_label].index[0]
                        wf.write(f"{subfilename},{subf},{str_label.strip()},{idx_label}\n")


class ImageNet10K(Dataset):
    def __init__(self, dataset="imagenet10K.csv", labels="mapping.csv", transform=None, target_transform=None, seed=42):
        self.df = pd.read_csv(dataset)
        self.mappings = pd.read_csv(labels, header=0)
        self.transform = transforms.Compose(self._prepend_transforms(transform, True))
        self.target_transform = transforms.Compose(self._prepend_transforms(target_transform, False))
        torch.manual_seed(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_row = self.df.iloc[idx].values
        image = Image.open(item_row[1]).convert("RGB")
        idx_label = item_row[3]
        image = self.transform(image)
        label = np.zeros(len(self.mappings))
        label[idx_label - 1] = 1
        label = torch.tensor(self.target_transform(label))
        return image, label
    
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
        
    def get_row(self, img_name):
        return self.df[self.df["filename"] == img_name].index.values[0]

    def split_dataset(self, splits=(0.8, 0.1, 0.1)):
        train_data, val_data, test_data = random_split(self, splits)
        return train_data, val_data, test_data
    
    def get_dataloaders(self, splits=(0.8, 0.1, 0.1), batch_sizes=(64,64,64), do_shuffles=(True, True, True)):
        train_data, val_data, test_data = self.split_dataset(splits)
        train_dl = DataLoader(train_data, batch_size=batch_sizes[0], shuffle=do_shuffles[0])
        val_dl = DataLoader(val_data, batch_size=batch_sizes[1], shuffle=do_shuffles[1])
        test_dl = DataLoader(test_data, batch_size=batch_sizes[2], shuffle=do_shuffles[2])
        return train_dl, val_dl, test_dl