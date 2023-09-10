import pandas as pd
import os
import torch
from torch import nn
from transformers import BertModel, PreTrainedTokenizerFast


class SNPEmbedding:
    def __init__(self,
                 logger,
                 pretrained_mdl_dir,
                 smiles_dir,
                 downstream_data_dir):
        self.logger = logger
        self.pretrained_mdl_dir = pretrained_mdl_dir
        self.smiles_dir = smiles_dir
        self.downstream_data_dir = downstream_data_dir

        self.snp_bert = BertModel.from_pretrained(pretrained_mdl_dir)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_mdl_dir)

    def cell_line_embedding(self):
        ls_cell_line = pd.read_csv(self.smiles_dir)['DEPMAP_ID'].unique()
        ls_cell_line_emb = []
        # embedding of all chrs on a cell line
        for cell_line in ls_cell_line:
            chr_emb = self.chr_embedding(cell_line)  # [24,768]
            ls_cell_line_emb.append(ChrMlp.forward(chr_emb))  # [24,768] --> [1,768]

        dic_cell_line_emb = {'DEPMAP_ID': ls_cell_line, 'SNP_EMB': ls_cell_line_emb}
        df_cell_line_emb = pd.DataFrame(dic_cell_line_emb)
        return df_cell_line_emb

    def chr_embedding(self, cell_line):
        ls_chr_emb = []
        ls_chr = [str(i) for i in range(1, 22)] + ['X', 'Y']

        for chr in ls_chr:
            # "chr_dir": "../ProcessedData/downstream_data/ACH-002397_chr1"
            chr_dir = self.downstream_data_dir + cell_line + '_chr' + chr
            if not os.path.exists(chr_dir):
                self.logger.info('Downstream data path does not exist')

            # embedding of all positions on a chr, folder name: ACH-002397_chr1
            idx = 0
            for pos in os.listdir(chr_dir):
                if idx == 0:
                    chr_pos_emb = self.snp_embedding(chr_dir+pos)
                else:
                    chr_pos_emb = torch.concat((chr_pos_emb, self.snp_embedding(chr_dir+pos)), 0)
                idx += 1
            ls_chr_emb.append(torch.mean(chr_pos_emb, 1))  # [1,768], embedding of a chr

        # concat embedding of all chrs
        chr_emb = ls_chr_emb[0]
        for i in range(1, 24):
            chr_emb = torch.concat((chr_emb, ls_chr_emb[i]), 0)
        return chr_emb  # [24,768]

    def snp_embedding(self, pos_dir):
        tmp_chr_pos = open(pos_dir, 'rb').readlines()
        inputs = self.tokenizer(tmp_chr_pos, return_tensors="pt", max_length=512, truncation=True)
        chr_pos_emb = self.snp_bert(**inputs)[0]
        return chr_pos_emb


class ChrMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )

    def forward(self, x):
        return self.layers(x)
