#!/usr/bin/env python
# coding: utf-8



import numpy as np




from tokenizers import SentencePieceBPETokenizer
tokenizer = SentencePieceBPETokenizer(
    vocab='./tokenizers/chr01_diploid-vocab.json',
    merges='./tokenizers/chr01_diploid-merges.txt',
    unk_token='[UNK]'
)


#####**Import and tokenize the text data**


output_txt_path = 'pretraining_data/pretraining_data_ch01.txt'




L=np.zeros(shape=(300000, 4096))
m=0
with open(output_txt_path, 'r', encoding='utf-8') as file:
    for line in file:
        #if m%3000==0:
            #print(m/3000,'%')
        encoded=tokenizer.encode(line.rstrip())
        input_ids=encoded.ids
        L[m]=np.pad(input_ids,(0,4096-len(input_ids)),constant_values=32000)
        m+=1
        if m==300000: #use 300000 sentences
            break



np.savetxt('tokenized.txt',L)



