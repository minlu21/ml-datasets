import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

def generate_mappings(filename="mapping.csv", subset_filename="tiny-imagenet/wnids.txt", orig_filename="tiny-imagenet/words.txt"):
    with open(filename, "w") as out_file:
        out_file.write("orig_lab;text_lab\n")
        print(f"Generating mapping file: {filename}...")
        mappings = dict()
        with open(orig_filename, "r") as in_file:
            for line in in_file:
                split_line = line.split("\t")
                mappings[split_line[0]] = split_line[1].strip(", \n\r")
        with open(subset_filename, "r") as in_file:
            for line in in_file:
                orig_label = line.strip("\n")
                text_label = mappings[orig_label]
                out_file.write(f"{orig_label};{text_label}\n")


def generate_train_dataset(img_root_dir="tiny-imagenet/train", mapping_file="mapping.csv", filename="tinyimagenet_train.csv"):
    df = pd.read_csv(mapping_file,names=["orig_lab", "text_lab"], index_col=False, sep=";")
    print("Generating Tiny ImageNet Train Dataset...")
    with open(filename, "w") as wf:
        wf.write("filename;filepath;str_label;idx_label\n")
        for filename in os.listdir(img_root_dir):
            f = os.path.join(os.path.abspath(img_root_dir), filename)
            if not os.path.isfile(f):
                for subfilename in os.listdir(f):
                    subf = os.path.join(f, subfilename)
                    if not os.path.isfile(subf):
                        for img_file in os.listdir(subf):
                            orig_label = img_file.split("_")[0]
                            str_label = df.loc[df["orig_lab"]==orig_label].values[0][1]
                            idx_label = df.loc[df["orig_lab"]==orig_label].index[0]
                            wf.write(f"{orig_label};{os.path.join(subf, img_file)};{str_label.strip()};{idx_label}\n")


def generate_val_dataset(img_root_dir="tiny-imagenet/val/images", annot_file="tiny-imagenet/val/val_annotations.txt", mapping_file="mapping.csv", filename="tinyimagenet_val.csv"):
    df = pd.read_csv(mapping_file,names=["orig_lab", "norm_lab"], index_col=False, sep=";")
    print("Generating Tiny ImageNet Validate Dataset...")
    annotation_dict = dict()
    with open(annot_file, "r") as annotations:
        for line in annotations:
            split_line = line.split("\t")
            annotation_dict[split_line[0]] = split_line[1]
    with open(filename, "w") as wf:
        wf.write("filename;filepath;str_label;idx_label\n")
        for filename in os.listdir(img_root_dir):
           f = os.path.join(os.path.abspath(img_root_dir), filename)
           if filename.find(".JPEG") != -1:
               # get original label
               orig_label = annotation_dict[filename]
               # get string label from original label
               str_label = df.loc[df["orig_lab"]==orig_label].values[0][1]
               idx_label = df.loc[df["orig_lab"]==orig_label].index[0]
               wf.write(f"{filename};{f};{str_label.strip()};{idx_label}\n")


class TinyImagenet(Dataset):
    def __init__(self, train_dataset="tinyimagenet_train.csv", val_dataset="tinyimagenet_val.csv", labels="mapping.csv", transform=None, target_transform=None, seed=42):
        train_df = pd.read_csv(train_dataset, sep=";")
        val_df = pd.read_csv(val_dataset, sep=";")
        self.df = pd.concat([train_df, val_df], axis=0)
        self.mappings = pd.read_csv(labels, header=0, sep=";")
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
