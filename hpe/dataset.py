from .composer import  Traincomposer as trc
from .composer import Testcomposer as tsc
from .norm_configs import NORM_CONFIGS as nc
from torchvision import datasets
import torch
import numpy as np


class Dataset:
    def __init__(
        self,
        batch_size,
        output_dir,
        dataset_name,
        img_dim,
        csv_file=None,
        path_to_images=None,
        dataloader=None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.img_dim = img_dim

        self.csv_file = csv_file
        self.path_to_images = path_to_images

        self.torch_dataloaders = {
            "cifar10": datasets.CIFAR10,
            "mnist": datasets.MNIST,
            "cifar100": datasets.CIFAR100,
        }

        self.mean, self.std = self.get_norms()

        self.composer_train, self.composer_test = self.get_composer(
            self.mean, self.std, self.img_dim
        )

        if kwargs == None:
            self.kwargs = {"num_workers": 1, "pin_memory": True, "device": "cpu"}
        else:
            self.kwargs = kwargs

        self.dataloader = dataloader

    def make_trainset(self):
        """
        Function that will build the entire dataset
        """
        train_loader = None

        if self.csv_file is not None and self.path_to_images is not None:

            transformed_dataset_train = self.dataloader(
                path_to_images=self.path_to_images,
                csv_file=self.csv_file,
                transform=self.composer_train,
            )

            train_loader = self.get_loader(
                transformed_dataset=transformed_dataset_train
            )

        else:

            train_loader = self.get_loader()

        return transformed_dataset_train

    def make_testset(self):
        """
        Function that will build the entire dataset
        """

        test_loader = None

        if self.csv_file is not None and self.path_to_images is not None:
            transformed_dataset_test = self.dataloader(
                path_to_images=self.path_to_images,
                csv_file=self.csv_file,
                transform=self.composer_test,
            )

            test_loader = self.get_loader(
                train_option=False, transformed_dataset=transformed_dataset_test
            )

        else:

            test_loader = self.get_loader(train_option=False)

        return transformed_dataset_test

    def get_norms(self):
        """
        Get the normalizations config for given dataset
        """
        nc_new = nc(self.dataset_name)
        mean, std = nc_new.return_norm_configs()

        return mean, std

    def get_composer(self, mean, std, img_dim):
        """
        Will return the necessary composer tho the dataset
        """
        train_composer, test_composer = None, None

        trc_new = trc(self.dataset_name, mean, std, img_dim)
        train_composer = trc_new.get_composer()

        tsc_new = tsc(self.dataset_name, mean, std, img_dim)
        test_composer = tsc_new.get_composer()

        return train_composer, test_composer

    def make_testset_few_classes(self, few_classes):
        """
        Function that will build just a few classes of the dataset
        """

        try:
            mean, std = self.get_norms()

            _, composer = self.get_composer(mean, std, self.img_dim)

            test_loader = self.torch_dataloaders[self.dataset_name](
                self.output_dir, train=False, download=True, transform=composer,
            )

            # get just the positions that the selected classes are
            idx = [
                np.where(np.array(test_loader.targets) == _class)[0].tolist()
                for _class in few_classes
            ]

            idx_concatenated = np.concatenate([each_idx for each_idx in idx])
            test_loader.targets = np.array(test_loader.targets)[
                idx_concatenated
            ].tolist()

            test_loader.data = test_loader.data[idx_concatenated]

            testset_fews = torch.utils.data.DataLoader(
                test_loader, batch_size=self.batch_size, shuffle=True, **self.kwargs
            )

            return testset_fews

        except TypeError as err:
            print(err)

    def get_loader(self, train_option=True, transformed_dataset=None):
        """
        Return the loader depending if the
        dataset receives csv/images_path or not
        """
        if transformed_dataset is None:
            return torch.utils.data.DataLoader(
                self.torch_dataloaders[self.dataset_name](
                    self.output_dir,
                    train=train_option,
                    download=True,
                    transform=(
                        self.composer_train
                        if train_option == True
                        else self.composer_test
                    ),
                ),
                batch_size=self.batch_size,
                shuffle=(True if train_option == True else False),
                **self.kwargs,
            )
        else:
            return torch.utils.data.DataLoader(
                transformed_dataset,
                batch_size=self.batch_size,
                shuffle=(True if train_option == True else False),
                **self.kwargs,
            )
