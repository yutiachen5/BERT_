import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedTokenizerFast


def snp_emb(pretrained_mdl_dir, smiles_dir, downstream_data_dir):
    # pretrained bert model and tokenizer
    SNPBert = BertModel.from_pretrained(pretrained_mdl_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_mdl_dir)

    # all cell-lines
    ls_cell_line = pd.read_csv(smiles_dir)['DEPMAP_ID'].unique()
    ls_cell_line_emb = []
    for i in range(len(ls_cell_line)):
        for chr in range(-1, 22):
            if chr == -1:
                temp_cell_line = open(downstream_data_dir+ls_cell_line[i]+"_chrX.txt", 'rb').readlines()
                inputs = tokenizer(temp_cell_line, return_tensors="pt", max_length=512, truncation=True)
                cell_line_emb = SNPBert(**inputs)[0]
            elif chr == 0:
                temp_cell_line = open(downstream_data_dir+ls_cell_line[i]+"_chrY.txt", 'rb').readlines()
                inputs = tokenizer(temp_cell_line, return_tensors="pt", max_length=512, truncation=True)
                cell_line_emb = torch.cat((cell_line_emb, SNPBert(**inputs))[0], 0)
            else:
                temp_cell_line = open(downstream_data_dir + ls_cell_line[i] + "_chr"+str(chr)+".txt", 'rb').readlines()
                inputs = tokenizer(temp_cell_line, return_tensors="pt", max_length=512, truncation=True)
                cell_line_emb = torch.cat((cell_line_emb, SNPBert(**inputs))[0], 0)
        # average 24 outputs from 24 chrs for a cell-line
        ls_cell_line_emb.append(torch.mean(cell_line_emb, 1))  # [1,768]
    # put cell-line name and corresponding embedding result into a df
    dic_cell_line_emb = {'DEPMAP_ID': ls_cell_line, 'SNP_EMB': ls_cell_line_emb}
    df_cell_line_emb = pd.DataFrame(dic_cell_line_emb)
    return df_cell_line_emb


