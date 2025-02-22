import os
import numpy as np
import random
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from PIL import Image

def read_files(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return torch.from_numpy(data[b"data"]), data[b"labels"]

class Cifar10(Dataset):
    def __init__(self, root_dir="./cifar-10", transform=None, target_transform=None, seed=42, 
                 watermark_num_classes=0, marking_network="model_state_dict.pth", carrier_path="carriers.pth", watermark_epochs=10):
        self.root_dir = root_dir
        tmp_df = []
        for batch_file in os.listdir(os.path.abspath(self.root_dir)):
            if (batch_file.find("data") == 0) or (batch_file.find("test_batch") == 0):
                print(f"Reading {batch_file}...")
                features, labels = read_files(os.path.join(os.path.abspath(self.root_dir), batch_file))
                tmp_list = [{"image": features[i, :].numpy(), "label": labels[i]} for i in range(len(labels))]
                tmp_df.extend(tmp_list)
        self.df = pd.DataFrame(tmp_df)
        text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        self.classes = {k: v for k, v in enumerate(text_labels)}
        random.seed(seed)
        self.transform = transforms.Compose(transform)
        self.target_transform = transforms.Compose(self._prepend_transforms(target_transform, False))
        if watermark_num_classes > 0:
            self.watermark_targets = random.sample(list(self.classes.keys()), watermark_num_classes)
            self.marking_network = marking_network
            self.carrier_path = carrier_path
            self.watermark_epochs = watermark_epochs
        else:
            self.watermark_targets = None
 
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item_row = self.df.iloc[idx].values
        # Change Numpy array of shape (3072,) to numpy array of shape (3, 32, 32) then (32, 32, 3)
        image = item_row[0].reshape((3, 32, 32)).transpose((1, 2, 0))
        if self.watermark_targets and item_row[1] in self.watermark_targets:
            # Save image into orig_image dir
            orig_img_root_dir = os.path.join(os.path.abspath(self.root_dir), "orig_img")
            watermark_img_root_dir = os.path.join(os.path.abspath(self.root_dir), "watermark_img")
            img_filename = f"cifar10_{idx}"
            orig_img_filepath = os.path.join(orig_img_root_dir, f"{img_filename}.jpeg")
            if not os.path.exists(orig_img_filepath):
                self.save_as_pil(image, orig_img_root_dir=orig_img_root_dir, filename=f"{img_filename}.jpeg")
            # Watermark image and set this as the image we want to return
            # Save watermarked image into watermark_image dir
            watermark_img_filepath = os.path.join(watermark_img_root_dir, f"{img_filename}.npy")
            if not os.path.exists(watermark_img_filepath):
                self.watermark_image(orig_img_filepath, watermark_img_root_dir, carrier_path=self.carrier_path, carrier_id=item_row[1], num_epochs=self.watermark_epochs)
            # Open image as numpy array and this will be the image we want to return 
            image = np.load(watermark_img_filepath)
        image = self.transform(image)
        idx_label = item_row[1]
        label = np.zeros(10)
        label[idx_label] = 1
        label = torch.tensor(self.target_transform(label))
        return image, label
    
    def label_to_text(self, label):
        return self.classes[torch.argmax(label).item()]
    
    def save_as_pil(self, tensor_image, orig_img_root_dir="orig_img", filename="img.jpeg"):
        os.makedirs(orig_img_root_dir, exist_ok=True)
        to_pil = Image.fromarray(tensor_image.astype(np.uint8))
        to_pil.save(os.path.join(orig_img_root_dir, filename))

    def watermark_image(self, orig_image_filepath, 
                        watermark_dump_path="watermark_img", 
                        carrier_path="carriers.pth", 
                        carrier_id=0,
                        num_epochs=10):
        os.makedirs(watermark_dump_path, exist_ok=True)
        command = f"""python make_data_radioactive.py \
            --carrier_id {carrier_id} \
            --carrier_path {carrier_path} \
            --epochs {num_epochs} \
            --img_paths {orig_image_filepath} \
            --marking_network {self.marking_network} \
            --optimizer sgd,lr=1.0 \
            --dump_path {watermark_dump_path}"""
        os.system(command)
    
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