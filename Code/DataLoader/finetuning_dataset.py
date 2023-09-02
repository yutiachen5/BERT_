import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from base import BaseDataLoader
from dataloader.embedding.smiles_embedding import smiles_emb
from sklearn.model_selection import train_test_split


class DrugSensitivityDataset(Dataset):
    def __init__(self,
                 logger,
                 dataset):
        self.dataset = dataset
        self.logger = logger

        # self.emb_data = np.inner(self.smiles_emb_data, self.snp_emb_data)
        self.ic50 = list(dataset['LN_IC50'])
        self.cell_line_name = list(dataset['CELL_LINE_NAME'])
        self.drug_name = list(dataset['DRUG_NAME'])

    def __len__(self):
        # return len(self.emb_data)
        return len(self.dataset)

    def __getitem__(self, i):
        # input_emb = self.emb_data[i]
        input_emb = self.dataset['SMILES_embedding'][i]
        true_ic50 = self.ic50[i]
        cell_line = self.cell_line_name[i]
        drug = self.drug_name[i]

        input_tensor = torch.squeeze(torch.tensor(input_emb, dtype=torch.float32))
        ic50_tensor = torch.tensor(true_ic50, dtype=torch.float32).unsqueeze(-1)
        return input_tensor, ic50_tensor, cell_line, drug


class EmbeddedDataset(BaseDataLoader):
    def __init__(self,
                 logger,
                 smiles_dir,
                 seed,
                 batch_size,
                 validation_split,
                 test_split,
                 num_workers,
                 snp_dir,
                 test_size,
                 shuffle=True):
        self.logger = logger
        self.smiles_dir = smiles_dir
        self.seed = seed
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.snp_dir = snp_dir
        self.test_size = test_size
        self.shuffle = shuffle

        self.smiles_dataset = self._load_smiles()
        logger.info('Embedding {} SMILES.'.format(len(self.smiles_dataset)))
        self.smiles_emb_dataset = smiles_emb.smiles_embedding(self.smiles_dataset)  # return a df
        # print(self.smiles_emb_dataset)

        # self.snp_dataset = self._load_snp()
        # logger.info('Embedding {} SNP.'.format(len(self.snp_dataset)))
        # self.snp_emb_dataset = snp_emb.snp_emb.py(self.snp_dataset)

        # self.dataset = self._merge(smiles_dataset = self.smiles_dataset, snp_dataset = self.snp_dataset)

        # train_df, test_df, valid_df = self._split_dataset(self.dataset)
        train_df, test_df, valid_df = self._split_dataset(self.smiles_emb_dataset)

        train_dataset = self._get_dataset(train_df)
        test_dataset = self._get_dataset(test_df)
        # valid_dataset = self._get_dataset(valid_df)
        super().__init__(train_dataset, batch_size, seed, shuffle, validation_split, test_split, num_workers)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    def _load_smiles(self):
        df_smiles = pd.read_csv(self.smiles_dir)
        df_smiles = df_smiles.loc[:100000, :]
        return df_smiles

    def _load_snp(self):
        ls_snp = list(np.load(self.snp_dir))
        return ls_snp

    def _merge(self, smiles_dataset, snp_dataset):
        merged_dataset = pd.concat(smiles_dataset, snp_dataset, ignore_index=True)
        merged_dataset.reset_index(drop=True, inplace=True)
        return merged_dataset

    def _get_dataset(self, split_dataset):
        dataset = DrugSensitivityDataset(dataset=split_dataset,
                                         logger=self.logger)
        return dataset

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
