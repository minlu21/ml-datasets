import numpy as np
from PIL import Image
from torchvision import transforms

from loaders import imagenet10k as imgnet
from loaders import tinyimagenet as timgnet

def show_tensor_img(img, label):
    demo_array = np.moveaxis(img.numpy() * 255, 0, -1)
    Image._show((Image.fromarray(demo_array.astype(np.uint8))))
    print(f"Label: {label}")


def test_imagenet10k():
    imagenet = imgnet.ImageNet10K(transform=transforms.Resize((224, 224)))
    print(f"{len(imagenet.split_dataset()[0])}")
    train_dataloader, _, _ = imagenet.get_dataloaders()
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Label batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    show_tensor_img(img, label)

def test_tinyimagenet():
    tinyimagenet = timgnet.TinyImagenet()
    print(f"{len(tinyimagenet.split_dataset()[0])}")
    train_dataloader, _, _ = tinyimagenet.get_dataloaders()
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Label batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    show_tensor_img(img, label)
    

if __name__ == "__main__":
    # test_imagenet10k()
    test_tinyimagenet()