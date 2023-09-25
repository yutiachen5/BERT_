import pandas as pd
import os
import torch
from torch import nn
from transformers import BertModel, PreTrainedTokenizerFast


class SNPEmbedding(nn.Module):
    def __init__(self,
                 logger,
                 pretrained_mdl_dir,
                 smiles_dir,
                 downstream_data_dir):
        super().__init__()
        self.logger = logger
        self.pretrained_mdl_dir = pretrained_mdl_dir
        self.smiles_dir = smiles_dir
        self.downstream_data_dir = downstream_data_dir

        self.snp_bert = BertModel.from_pretrained(pretrained_mdl_dir)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_mdl_dir)

        self.MLP_predict = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )

    def forward(self):
        ls_cell_line = pd.read_csv(self.smiles_dir)['DEPMAP_ID'].unique()
        chr_emb = self.cell_line_embedding(ls_cell_line)
        output = self.MLP_predict(chr_emb)  # [24, 768] --> [1,768], emb of a cel line
        return output

    def chr_embedding(self, ls_cell_line):
        for cell_line in ls_cell_line:
            chr_emb = self.pos_embedding(cell_line)  # [24,768], emb of a chr given a cell line
        return chr_emb

    def pos_embedding(self, cell_line):
        ls_chr_emb = []
        ls_chr = [str(i) for i in range(1, 22)] + ['X', 'Y']

        for chr in ls_chr:
            # "chr_dir": "../ProcessedData/downstream_data/ACH-002397_chr1"
            chr_dir = self.downstream_data_dir + cell_line + '_chr' + chr
            if not os.path.exists(chr_dir):
                self.logger.info('Downstream data path does not exist.')

            idx = 0
            for pos in os.listdir(chr_dir):
                if idx == 0:
                    chr_pos_emb = self.snp_embedding(chr_dir+pos)  # ACH-002397_chr1+pos
                else:
                    chr_pos_emb = torch.concat((chr_pos_emb, self.snp_embedding(chr_dir+pos)), 0)
                idx += 1
            ls_chr_emb.append(torch.mean(chr_pos_emb, 1))  # [number of chr pos, 768] --> [1,768], embedding of a chr

        # concat embedding of all chrs
        chr_emb = ls_chr_emb[0]
        for i in range(1, 24):
            chr_emb = torch.concat((chr_emb, ls_chr_emb[i]), 0)  # [1,768] --> [24,768]
        return chr_emb

    def snp_embedding(self, pos_dir):
        tmp_chr_pos = open(pos_dir, 'rb').readlines()
        inputs = self.tokenizer(tmp_chr_pos, return_tensors="pt", max_length=512, truncation=True)
        chr_pos_emb = self.snp_bert(**inputs)[0]
        return chr_pos_emb


