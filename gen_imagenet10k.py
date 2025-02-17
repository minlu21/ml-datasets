from src.loaders import imagenet as imgnet

imgnet.generate_mappings(filename="../research/radioactive_data/mapping.csv", orig_map_filepath="../scratch/imagenet10K/synset_mapping.txt")
imgnet.generate_dataset(mapping_file="../research/radioactive_data/mapping.csv", filename="../research/radioactive_data/imagenet10K.csv", dataset_dir="../scratch")