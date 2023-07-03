import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Code.base import BaseDataLoader
from embedding.smiles_embedding import smiles_emb
from sklearn.model_selection import train_test_split


class EmbDataset(Dataset):
    def __init__(self,
                 logger,
                 dataset):
        self.dataset = dataset
        self.logger = logger

        self.smiles_emb_data = smiles_emb.main(self.dataset)
        logger.info('Embedding {} SMILES.'.format(len(self.dataset)))
        # self.snp_emb_data = snp_emb.main(snp_dir)
        logger.info('Embedding {} SNP.'.format(len(self.dataset)))
        self.emb_data = np.inner(self.smiles_emb_data, self.snp_emb_data)
        self.ic50 = list(dataset['LN_IC50'])

    def __len__(self):
        return len(self.emb_data)

    def __getitem__(self, i):
        input_emb = self.emb_data[i]
        true_ic50 = self.ic50[i]
        return {"emb_input": torch.tensor(input_emb, dtype=torch.long),
                "ic50": torch.tensor(true_ic50, dtype=torch.long)}


class EmbeddedDataset(BaseDataLoader):
    def __init__(self,
                 logger,
                 smiles_dir,
                 seed,
                 batch_size,
                 validation_split,
                 test_split,
                 num_workers,
                 data_dir,
                 seq_dir,
                 test_size,
                 shuffle=True):
        self.logger = logger
        self.smiles_dir = smiles_dir
        self.seed = seed
        self.data_dir = data_dir
        self.seq_dir = seq_dir
        self.test_size = test_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=self.seed)

        self.smiles_dataset = self._load()

        train_df, test_df, valid_df = self._split_dataset(self.smiles_dataset)
        train_dataset = self._get_dataset(train_df)
        test_dataset = self._get_dataset(test_df)
        # valid_dataset = self._get_dataset(valid_df)
        super().__init__(train_dataset, batch_size, seed, shuffle, validation_split, test_split, num_workers)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    def _load(self):
        dataset = pd.read_csv(self.smiles_dir)
        return dataset

    def _get_dataset(self, split_dataset):
        emb_dataset = EmbDataset(dataset=split_dataset,
                                 logger=self.logger)
        return emb_dataset

    def _split_dataset(self, emb_data):

        train_dataset, test_dataset = train_test_split(emb_data,
                                                       test_size=self.test_size,
                                                       shuffle=self.shuffle)
        test_dataset, valid_dataset = train_test_split(test_dataset,
                                                       test_size=self.validation_split,
                                                       shuffle=self.shuffle)
        train_dataset.reset_index(inplace=True, drop=True)
        test_dataset.reset_index(inplace=True, drop=True)
        valid_dataset.reset_index(inplace=True, drop=True)

        return train_dataset, test_dataset, valid_dataset

    def get_test_dataloader(self):
        return self.test_dataloader
