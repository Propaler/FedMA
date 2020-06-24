from .dataset import Dataset
from .norm_configs import NORM_CONFIGS as nc
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset as DatasetTorch
from .composer import Traincomposer as trc
from .composer import Testcomposer as tsc
import torch
import os
import numpy as np
import pandas as pd
from PIL import Image

class DataLoaderNIH(DatasetTorch):
    def __init__(self, path_to_images, csv_file, transform, sample = 0):

        self.path_to_images = path_to_images
        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.df = self.df.set_index("id")
        self.PRED_LABEL = [
            "Atelectasis",
            #"Cardiomegaly",
            "Effusion",
            #"Infiltration",
            "Mass",
            #"Nodule",
            #"Pneumonia",
            "Pneumothorax",
            #"Consolidation",
            #"Edema",
            #"Emphysema",
            #"Fibrosis",
            #"Pleural_Thickening",
            #"Hernia",
            "No Finding",
        ]

        if sample > 0 and sample < len(self.df):
            self.df = self.df.samples(sample)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        # load the image
        image = None
        abs_path = ""
        while image is None:
            for path in os.listdir(self.path_to_images):
                if str(path).startswith("images"):
                    try:
                        abs_path = os.path.join(self.path_to_images, path, "images", self.df.index[idx])
                        image = Image.open(abs_path)
                        break     
                    except FileNotFoundError as e:
                        pass

        #print(abs_path)
        #print(last_images_dir)

        image = image.convert('RGB')

        # create an array with the size of pred_label
        label = self.PRED_LABEL.index(str(self.df["label"].iloc[idx]))

        # get the item for idx
        #labels_idx = self.df["label"].iloc[idx].split("|")
        #labels_list = "-".join(labels_idx)
 
        #with open("/home/jeferson/repo/hpe-puma-vision-3/src/examples/nih_absolute_paths.csv", "a") as f:
        #    f.write(f"{abs_path},{labels_list}\n")
        #    f.close()

        #for each_label in labels_idx:
        #    if each_label in self.PRED_LABEL:
        #        idx_pred_label = self.PRED_LABEL.index(str(each_label))
        #        label[idx_pred_label] = 1.

        if self.transform:
            image = self.transform(image)


        return (image, label)


class NIH(Dataset):
    def __init__(
        self,
        batch_size,
        output_dir,
        dataset_name,
        csv_file,
        path_to_images,
        img_dim=64,
        **kwargs
    ):

        self.batch_size = batch_size
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.csv_file = csv_file
        self.img_dim = img_dim
        self.path_to_images = path_to_images

        if kwargs == None:
            self.kwargs = {"num_workers": 2, "pin_memory": True}
        else:
            self.kwargs = kwargs

    def make_trainset(self):
        """
        Function that will build the entire dataset
        """

        mean, std = self.get_norms()
        composer, _ = self.get_composer(mean, std, self.img_dim)

        transformed_dataset_train = DataLoaderNIH(
            path_to_images=self.path_to_images,
            csv_file=self.csv_file,
            transform=composer,
        )

        train_dataloader = torch.utils.data.DataLoader(
            transformed_dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            **self.kwargs,
        )

        return train_dataloader

    def make_testset(self):
        """
        Function that will build the entire dataset
        """

        mean, std = self.get_norms()

        _, composer = self.get_composer(mean, std, self.img_dim)

        transformed_dataset_test = DataLoaderNIH(
            path_to_images=self.path_to_images,
            csv_file=self.csv_file,
            transform=composer,
        )

        test_dataloader = torch.utils.data.DataLoader(
            transformed_dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            **self.kwargs,
        )

        return test_dataloader