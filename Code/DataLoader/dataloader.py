import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from embedding.smiles_embedding import smiles_emb


class SelfSupervisedDataset(Dataset):
    def __init__(self,
                 logger,
                 # snp_emb_data,
                 smiles_emb_data,):
        # self.snp_emb_data = snp_emb_data
        self.smiles_emb_data = smiles_emb_data
        self.logger = logger

        self.logger.info(f"Creating self supervised smiles dataset with {len(self.smiles_emb_data)} smiles")
        # self.logger.info(f"Creating self supervised snp dataset with {len(self.snp_data)} sequences")

    def __len__(self):
        return len(self.smiles_emb_data)

    def __getitem__(self, i):
        # input_snp_id = self.snp[i]
        input_smiles_id = self.smiles_emb_data[i]
        # return {"input_ids_snp": torch.tensor(input_snp_id, dtype=torch.long),
        #         "input_ids_smiles": torch.tensor(input_smiles_id, dtype=torch.long)}
        return {"input_ids_smiles": torch.tensor(input_smiles_id, dtype=torch.long)}


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

        # self.snp_data, self.smiles_data = self.load()
        self.smiles_emb_data = smiles_emb.main(self.smiles_dir)
        # self.snp_emb_data = snp_emb.main(snp_dir)

    def load(self):
        smiles_data = np.load(self.smiles_dir)
        snp_data = np.load(self.snp_dir)
        return snp_data, smiles_data

    def get_dataset(self):
        self_supervised_dataset = SelfSupervisedDataset(# snp_emb_data=self.snp_emb_data,
                                                        smiles_emb_data=self.smiles_emb_data,
                                                         logger=self.logger)
        return self_supervised_dataset
