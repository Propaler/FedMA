class NORM_CONFIGS:
    """
    Return wich config the dataset going to use
    """

    def __init__(self, name):
        self._name = name.lower()
        self.norms = {
            "_mean": {
                "mnist": (0.1367,),
                "cifar10": (0.4914, 0.4822, 0.4465),
                "cifar100": (0.5071, 0.4867, 0.4408),
                "nih": (0.485, 0.456, 0.406),
            },
            "_std": {
                "mnist": (0.3081,),
                "cifar10": (0.2023, 0.1994, 0.2010),
                "cifar100": (0.2675, 0.2565, 0.2761),
                "nih": (0.229, 0.224, 0.225),
            },
        }

    def return_norm_configs(self):
        """
        Return the dataset config dependind on the name
        """

        mean_config = self.norms["_mean"][self._name]
        std_config = self.norms["_std"][self._name]

        return (mean_config, std_config)
