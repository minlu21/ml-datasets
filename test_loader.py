import numpy as np
from PIL import Image
from torchvision import transforms

from src.loaders import imagenet10k as imgnet
from src.loaders import tinyimagenet as timgnet
from src.loaders import cifar10 as cifar

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


def test_cifar10():
    cifar10 = cifar.Cifar10(watermark_num_classes=2, transform=[transforms.ToTensor()])
    print(f"{len(cifar10.split_dataset()[0])}")
    train_dataloader, _, _ = cifar10.get_dataloaders()
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Label batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    show_tensor_img(img, label)
    print(f"Text Label: {cifar10.label_to_text(label)}")
    

if __name__ == "__main__":
    # test_imagenet10k()
    # test_tinyimagenet()
    test_cifar10()