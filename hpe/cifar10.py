from .dataset import Dataset
from .dataloader import BaseDataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset as DatasetTorch
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image
from .dataset_constants import DEFAULT_COLUMN


class DataLoaderCifar10(BaseDataLoader, DatasetTorch):
    def __init__(self, path_to_images=None, csv_file=None, transform=None):
        BaseDataLoader.__init__(self, path_to_images, csv_file, transform)
        DatasetTorch.__init__(self)

    def __getitem__(self, idx):

        # load the image
        image = None
        try:
            image = Image.open(
                os.path.join(str(self.df.index[idx]))
            )
        except FileNotFoundError as e:
            pass

        image = image.convert("RGB")

        # get the item for idx
        label = np.where(
            self.PRED_LABEL == self.df[DEFAULT_COLUMN.DEFAULT_LABEL_COLUMN].iloc[idx]
        )[0][0]

        if self.transform:
            image = self.transform(image)

        return (image, label)

