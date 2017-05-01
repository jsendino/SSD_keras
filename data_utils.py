import os
import pandas as pd
import numpy as np

from sklearn.utils import shuffle


def load_all_labels(aug_folders):
    aug_targets = pd.DataFrame(columns = ["img", "x","y","w","h", "id"])
    for folder in aug_folders:
        folder_name = os.path.basename(folder)
        
        print "Loading data augmentation folder:", folder_name
        targets = pd.read_csv(folder + '/boxes_with_id.csv', names = ["img", "x","y","w","h", "id"])
        targets = targets.sort_values(by = "img")
        targets["img"] = folder_name + '/' + targets["img"]
        print "Number of examples:", len(targets)
        print
        
        if folder_name != "original":
            aug_targets = aug_targets.append(targets)
        else:
            og_targets = targets
    print "total number of examples: ", len(aug_targets) + len(og_targets)
    return aug_targets, og_targets


def train_val_split(aug_labels, og_labels, train_perc, val_perc):
    N = len(og_labels)
    train_idx = np.random.choice(N, size=int(N*train_perc), replace=True)
    val_idx = np.setdiff1d(range(N), train_idx)
    
    # Make subset from original dataset
    train_og_labels = og_labels.iloc[train_idx]
    val_labels = og_labels.iloc[val_idx]
    # Find the augmented pictures corresponding to those in train split and add it to that set
    train_labels = train_og_labels.append(aug_labels.loc[aug_labels.id.isin(train_og_labels.id)])
    
    return train_labels, val_labels