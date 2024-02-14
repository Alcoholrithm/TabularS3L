from typing import Union
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler, RandomSampler

from pytorch_lightning import LightningDataModule

class TabularS3LDataModule(LightningDataModule):
    """The pytorch lightning datamodule for VIME
    """
    def __init__(self, train_ds:Dataset, val_ds:Dataset, batch_size: int, train_sampler: Union["seq", "weighted", "random"], train_collate_fn = None, valid_collate_fn = None, n_gpus: int = 1, n_jobs: int = 32, drop_last: int = False, is_regression:bool = False):
        """Initialize the datamodule

        Args:
            train_ds (Dataset): The training dataset
            val_ds (Dataset): The validation dataset
            batch_size (int): The batch size of the dataset
            train_collate_fn (object): Collate function for train dataloader
            valid_collate_fn (object): Collate function for validation dataloader
            n_gpus (int, optional): The number of the gpus to use. Defaults to 1.
            n_jobs (int, optional): The number of the cpu core to use. Defaults to 32.
            drop_last (bool, optional): The flag to drop the last batch or not. Defaults to False.
            is_regression (bool, optional): The flag that determines whether the datamodule is for regression task or not. Defaults to False.
        """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.batch_size = batch_size
        
        assert train_sampler in ["seq", "weighted", "random"], 'The train_sampler must be one of the following: "seq", "weighted", or "random".'
        self.train_sampler = train_sampler
        
        self.train_collate_fn = train_collate_fn
        self.valid_collate_fn = valid_collate_fn
        
        self.n_gpus = n_gpus
        self.n_jobs = n_jobs
        self.is_regression = is_regression
        self.drop_last = drop_last

    def setup(self, stage: str):
        """Setup the datamodule for pytorch lightning module of VIME
        
        Use a weighted random sampler for the finetunning step of the classification task, otherwise use a random sampler.
        
        Args:
            stage (str): For compatibility, do not use
        """
        if self.train_sampler == "seq":
            sampler = SequentialSampler(self.train_ds)
            shuffle = True
        elif self.train_sampler == "weighted":
            sampler = WeightedRandomSampler(self.train_ds.weights, num_samples = len(self.train_ds))
            shuffle = False
        elif self.train_sampler == "random":
            sampler = RandomSampler(self.train_ds, num_samples = len(self.train_ds))
            shuffle = False

        self.train_dl = DataLoader(self.train_ds, 
                                    batch_size = self.batch_size, 
                                    shuffle=shuffle, 
                                    sampler = sampler,
                                    num_workers=self.n_jobs,
                                    drop_last=self.drop_last,
                                    collate_fn = self.train_collate_fn)
        self.val_dl = DataLoader(self.val_ds, batch_size = self.batch_size, shuffle=False, sampler = SequentialSampler(self.val_ds), num_workers=self.n_jobs, drop_last=False, collate_fn=self.valid_collate_fn)
    
    def train_dataloader(self):
        """Return the training dataloader
        """
        return self.train_dl

    def val_dataloader(self):
        """Return the validation dataloader
        """
        return self.val_dl