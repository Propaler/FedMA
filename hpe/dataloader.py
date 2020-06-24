from abc import ABC
from abc import abstractmethod
import pandas as pd
import numpy as np
from .dataset_constants import DEFAULT_COLUMN


class BaseDataLoader(ABC):
    def __init__(self, path_to_images, csv_file, transform):

        self.path_to_images = path_to_images
        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.df = self.df.set_index(DEFAULT_COLUMN.DEFAULT_ID_COLUMN)

        self.PRED_LABEL = np.unique(self.df[DEFAULT_COLUMN.DEFAULT_LABEL_COLUMN].values)

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
