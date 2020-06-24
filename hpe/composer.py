from torchvision import transforms


class Traincomposer:
    def __init__(self, dataset_name, mean, std, img_dim):
        self.dataset_name = dataset_name.lower()
        self.mean = mean
        self.std = std
        self.img_dim = img_dim
        self.composers_train = {
            "cifar": transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(self.img_dim),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
            "mnist": transforms.Compose(
                [
                    transforms.Resize((self.img_dim, self.img_dim)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
            "nih": transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((self.img_dim, self.img_dim)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        }

    def get_composer(self):
        """
        Return composer (equals for cifar10 and 100)
        """
        if self.dataset_name.lower() in ["cifar10", "cifar100"]:
            return self.composers_train["cifar"]
        else:
            return self.composers_train[self.dataset_name]


class Testcomposer:
    def __init__(self, dataset_name, mean, std, img_dim):
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.img_dim = img_dim
        self.composers_test = {
            "cifar": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(self.mean, self.std),]
            ),
            "mnist": transforms.Compose(
                [
                    transforms.Resize((self.img_dim, self.img_dim)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
            "nih": transforms.Compose(
                [
                    transforms.Resize((self.img_dim, self.img_dim)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            ),
        }

    def get_composer(self):
        """
        Return composer (equals for cifar10 and 100)
        """
        if self.dataset_name.lower() in ["cifar10", "cifar100"]:
            return self.composers_test["cifar"]
        else:
            return self.composers_test[self.dataset_name]
