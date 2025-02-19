import torch
import os

def calc_feature_means_loop(dataloader, dim=0):
    feat_means = []
    for feat, _ in dataloader:
        feat_means.append(torch.mean(feat.double(), dim=dim))
    print(feat_means[0])
    return torch.tensor(sum(feat_means) / len(feat_means))


def calc_feature_std_loop(dataloader, dim=0):
    feat_stds = []
    for feat, _ in dataloader:
        feat_stds.append(torch.std(feat.double(), dim=dim))
    print(feat_stds[0])
    return torch.tensor(sum(feat_stds) / len(feat_stds))


def write_result(file, result):
    with open(file, "a") as wf:
        wf.write(result)