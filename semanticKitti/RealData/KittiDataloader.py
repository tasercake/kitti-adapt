import glob
import torch
from torchvision.transforms import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torch.optim import Adam
import numpy as np, os, sys




class DataLoader_vkitti(pl.LightningModule):
    def __init__(self, dataset, model, batch_size,val_split = 0.1,test_split = 0.1):
        super(DataLoader_vkitti,self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        ### Ratio for splitting data into the different sets
        self.val_split = val_split
        self.test_split = test_split

        self.train_loader = 0
        self.val_loader = 0
        self.test_loader = 0

        ### Needed as our model used is for image segmentation
        self.feature_extractor = model
        self.feature_extractor.eval()

    def forward(self, x):
        x = self.feature_extractor(x)['out']
        return x

    def split_data(self,dataset,split_ratio1,split_ratio2,seed=1337):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split_1 = int(np.floor(split_ratio1 * dataset_size))
        split_2 = int(np.floor(split_ratio2 * dataset_size))
        if 1:
            np.random.seed(seed)
            np.random.shuffle(indices)
        set1_indices, set2_indices, set3_indices = indices[:split_1], indices[split_1:split_2], indices[split_2:]
        set1_sample = torch.utils.data.sampler.SubsetRandomSampler(set1_indices)
        set2_sample = torch.utils.data.sampler.SubsetRandomSampler(set2_indices)
        set3_indices = torch.utils.data.sampler.SubsetRandomSampler(set3_indices)

        return set1_sample,set2_sample,set3_indices

    def prepare_data(self):
        train_sampler,val_sampler,test_sampler = self.split_data(self.dataset, 0.6, 0.8)

        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, sampler=train_sampler)
        self.val_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, sampler=val_sampler)
        self.test_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, sampler=test_sampler)



    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        criterion = torch.nn.MSELoss()
        loss = criterion(output, label)
        return {'loss': loss}
        # return loss (also works)

