import torch
import torch.nn as nn
from transformers import BertModel, PreTrainedTokenizerFast

SNPBert = BertModel.from_pretrained('0811_113404/checkpoint-1878')
print(SNPBert)
tokenizer = PreTrainedTokenizerFast.from_pretrained('0811_113404')
# cell_line = open("ACH-002397_chr8.txt", 'rb')
# SNP = str(cell_line.readlines())
str = 'ATGC'*1000
inputs = tokenizer(str, return_tensors="pt",max_length = 512, truncation = True)
print(inputs['input_ids'].shape)
outputs = SNPBert(**inputs)
print(outputs[0].shape)
print(outputs[0])

