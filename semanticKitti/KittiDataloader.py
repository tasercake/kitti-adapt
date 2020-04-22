import glob
import torch
from torchvision.transforms import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torch.optim import Adam
import numpy as np, os, sys
import torch.nn as nn


class DataLoader_kitti(pl.LightningModule):
    def __init__(self, rdataset, vdataset, model, batch_size, virtual_data_ratio, num_class, lr):
        super(DataLoader_kitti, self).__init__()
        self.lr = lr
        self.vdataset = vdataset
        self.rdataset = rdataset
        self.batch_size = batch_size

        ### Ratio for splitting data into the different sets
        self.virtual_ratio = virtual_data_ratio
        self.rdataset_length = len(rdataset)
        self.vdataset_length = len(vdataset)

        ### 60% of data used for testing, 20% for validation and 20% for testing (For 100% real data),
        # with synthetic data, amount of data used for testing remains the same while ratio for training and validation set changes.
        # rtrain_sampler, rval_sampler, rtest_sampler = self.split_data_3set(self.rdataset, 0.6,0.8)
        if virtual_data_ratio > 1:
            desired_r_train_size = np.int(self.rdataset_length * 0.6)
            desired_r_val_size = np.int(self.rdataset_length * 0.2)
            desired_r_test_size = np.int(self.rdataset_length * 0.2)
        else:
            desired_r_train_size = np.int(self.rdataset_length * 0.6 * (1 - virtual_data_ratio))
            desired_r_val_size = np.int(self.rdataset_length * 0.2 * (1 - virtual_data_ratio))
            desired_r_test_size = np.int(self.rdataset_length * 0.2)

        self.rtrain_indices, self.rval_indices, self.rtest_indices = self.split_data_return_indices_3(
            self.rdataset_length, desired_r_train_size, desired_r_val_size, desired_r_test_size)

        desired_v_train_size = np.int((self.rdataset_length * 0.6) * virtual_data_ratio)
        desired_v_val_size = np.int((self.rdataset_length * 0.2) * virtual_data_ratio)
        self.vtrain_indices, self.vval_indices = self.split_data_return_indices_2(self.vdataset_length,
                                                                                  desired_v_train_size,
                                                                                  desired_v_val_size)

        self.train_loader = 0
        self.val_loader = 0
        self.test_loader = 0

        ### Needed as our model used is for image segmentation
        self.feature_extractor = model
        self.feature_extractor.eval()
        self.conv = nn.Conv2d(21, num_class, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.feature_extractor(x)['out']
        # print(x)
        x = self.conv(x)
        return x

    def split_data_3set(self, dataset, split_ratio1, split_ratio2, seed=1337):
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

        return set1_sample, set2_sample, set3_indices

    def split_data_return_indices_2(self, dataset_size, sample_size1, sample_size2, seed=1337):
        indices = list(range(dataset_size))
        sample2_end_index = sample_size1 + sample_size2
        if 1:
            np.random.seed(seed)
            np.random.shuffle(indices)
        set1_indices, set2_indices = indices[:sample_size1], indices[sample_size1:sample2_end_index]
        # set1_sample = torch.utils.data.sampler.SubsetRandomSampler(set1_indices)
        # set2_sample = torch.utils.data.sampler.SubsetRandomSampler(set2_indices)
        return set1_indices, set2_indices

    def split_data_return_indices_3(self, dataset_size, sample_size1, sample_size2, sample_size3, seed=1337):
        indices = list(range(dataset_size))
        sample2_end_index = sample_size1 + sample_size2
        sample3_end_index = sample2_end_index + sample_size3
        if 1:
            np.random.seed(seed)
            np.random.shuffle(indices)
        set1_indices, set2_indices, set3_indices = indices[:sample_size1], indices[
                                                                           sample_size1:sample2_end_index], indices[
                                                                                                            sample2_end_index:sample3_end_index]
        # set1_sample = torch.utils.data.sampler.SubsetRandomSampler(set1_indices)
        # set2_sample = torch.utils.data.sampler.SubsetRandomSampler(set2_indices)
        # set3_sample = torch.utils.data.sampler.SubsetRandomSampler(set3_indices)
        return set1_indices, set2_indices, set3_indices

    def split_data(self, dataset, split_ratio1, seed=1337):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split_1 = int(np.floor(split_ratio1 * dataset_size))
        if 1:
            np.random.seed(seed)
            np.random.shuffle(indices)
        set1_indices, set2_indices = indices[:split_1], indices[split_1:]
        set1_sample = torch.utils.data.sampler.SubsetRandomSampler(set1_indices)
        set2_sample = torch.utils.data.sampler.SubsetRandomSampler(set2_indices)

        return set1_sample, set2_sample

    def prepare_data(self):
        vtrain_dataset = torch.utils.data.Subset(self.vdataset, self.vtrain_indices)
        vval_dataset = torch.utils.data.Subset(self.vdataset, self.vval_indices)

        rtrain_dataset = torch.utils.data.Subset(self.vdataset, self.rtrain_indices)
        rval_dataset = torch.utils.data.Subset(self.vdataset, self.rval_indices)
        rtest_dataset = torch.utils.data.Subset(self.vdataset, self.rtest_indices)
        print('vtrain len: %d vval len: %d rtrain len: %d rval len: %d rtest len: %d' % (len(vtrain_dataset),
                                                                                         len(vval_dataset),
                                                                                         len(rtrain_dataset),
                                                                                         len(rval_dataset),
                                                                                         len(rtest_dataset)))
        training_dataset = torch.utils.data.ConcatDataset([vtrain_dataset, rtrain_dataset])
        validation_dataset = torch.utils.data.ConcatDataset([vval_dataset, rval_dataset])
        self.train_loader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(rtest_dataset)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(output, label)
        #         print('output_pred data: ', output_pred.data)
        #         print('label data: ', label_max.data)
        #         if batch_idx % 20 == 0:
        #             print(str(batch_idx) + 'Loss ' + str(loss))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(output, label)
        output_pred = output.argmax(1)
        label_max = label.argmax(1)
        #         print('output_pred data',output_pred.data)
        #         print('label_max data',output_pred.data)
        #         print('length of prediction ', (output_pred.data == label_max.data).cpu().numpy())
        #         print('sum of prediction ', np.sum((output_pred.data == label_max.data).cpu().numpy()))
        running_corrects = np.sum((output_pred.data == label_max.data).cpu().numpy())
        running_total = label_max.shape[0] * label_max.shape[1] * label_max.shape[2]
        #         if batch_idx % 10 == 0:
        #             print('correct: ' + str(running_corrects) + ' total ' + str(running_total))
        return {'val_loss': loss, 'running_correct': running_corrects, 'running_total': running_total}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_corr = np.sum([x['running_correct'] for x in outputs])
        val_total = np.sum([x['running_total'] for x in outputs])
        #         print(val_corr,val_total)
        val_acc = float(val_corr) / float(val_total)
        print('validation accuracy', val_acc)
        return {'val_loss': val_loss_mean, 'accuracy': val_acc}

    def test_step(self, batch, batch_idx):
        x, label = batch
        output = self.forward(x)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        loss = criterion(output, label)
        output_pred = output.argmax(1)
        label_max = label.argmax(1)
        running_corrects = np.sum((output_pred.data == label_max.data).cpu().numpy())
        running_total = label_max.shape[0] * label_max.shape[1] * label_max.shape[2]
        return {'test_loss': loss, 'running_correct': running_corrects, 'running_total': running_total}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_corr = np.sum([x['running_correct'] for x in outputs])
        test_total = np.sum([x['running_total'] for x in outputs])
        test_acc = float(test_corr) / float(test_total)
        print('test accuracy', test_acc)
        return {'test_loss': test_loss_mean, 'accuracy': test_acc}