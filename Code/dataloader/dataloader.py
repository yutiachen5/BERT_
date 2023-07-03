import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from embedding.smiles_embedding import smiles_emb


class SelfSupervisedDataset(Dataset):
    def __init__(self,
                 logger,
                 emb_data,
                 ic50):
        self.emb_data = emb_data
        self.ic50 = ic50
        self.logger = logger
        self.logger.info(f"Creating self supervised dataset with length of {len(self.emb_data)}.")

    def __len__(self):
        return len(self.emb_data)

    def __getitem__(self, i):
        input_emb = self.emb_data[i]
        true_ic50 = self.ic50[i]
        return {"input": torch.tensor(input_emb, dtype=torch.long),
                "ic50": torch.tensor(true_ic50, dtype=torch.long)}


class Dataset(object):
    def __init__(self,
                 config,
                 logger,
                 seed,
                 smiles_dir,
                 snp_dir,
                 test_split):
        self.config = config
        self.logger = logger
        self.seed = seed
        self.smiles_dir = smiles_dir
        self.snp_dir = snp_dir
        self.test_split = test_split

        self.ic50 = self.load_ic50()
        self.smiles_emb_data = smiles_emb.main(self.smiles_dir)
        # self.snp_emb_data = snp_emb.main(snp_dir)

        # preprocessing--inner product of SMILES & SNP
        self.emb_data = np.inner(self.smiles_emb_data, self.snp_emb_data)

    def load_ic50(self):
        ic50_true = list(pd.read_csv(self.smiles_dir)['LN_IC50'])
        return ic50_true

    def get_dataset(self):
        self_supervised_dataset = SelfSupervisedDataset(emb_data=self.emb_data,
                                                        logger=self.logger,
                                                        ic50=self.ic50)
        return self_supervised_dataset
