from .dataset import Dataset
from .dataloader import BaseDataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset as DatasetTorch
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image
from .dataset_constants import DEFAULT_COLUMN

class DataLoaderMNIST(BaseDataLoader, DatasetTorch):
    def __init__(self, path_to_images=None, csv_file=None, transform=None):
        BaseDataLoader.__init__(self, path_to_images, csv_file, transform)
        DatasetTorch.__init__(self)

    def __getitem__(self, idx):

        # load the image
        image = None
        try:
            image = Image.open(
                os.path.join(self.path_to_images, str(self.df.index[idx]) + ".png")
            )
        except FileNotFoundError as e:
            pass

        # create an array with the size of pred_label
        label = np.zeros(len(self.PRED_LABEL), dtype=int)

        # get the item for idx
        label = self.df[DEFAULT_COLUMN.DEFAULT_LABEL_COLUMN].iloc[idx]

        if self.transform:
            image = self.transform(image)

        return (image, label)