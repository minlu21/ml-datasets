from src.loaders import tinyimagenet as imgnet

imgnet.generate_mappings(filename="../mapping.csv", subset_filename="../scratch/tiny-imagenet/wnids.txt", orig_filename="../scratch/tiny-imagenet/words.txt")
imgnet.generate_train_dataset(img_root_dir="../scratch/tiny-imagenet/train", mapping_file="../mapping.csv", filename="../tinyimagenet_train.csv")
imgnet.generate_val_dataset(img_root_dir="../scratch/tiny-imagenet/val", annot_file="../scratch/tiny-imagenet/val/val_annotations.txt", mapping_file="../mapping.csv", filename="../tinyimagenet_val.csv")
