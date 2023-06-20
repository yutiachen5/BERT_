#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import random
import tqdm
import json
from tokenizers import (ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer)

# In[ ]:


nt_biallele_code = json.load(open('./resource/snp_vocab.json', 'r'))
nt_to_index = json.load(open('./resource/nucleotide_to_index.json', 'r'))
index_to_nt = {v:k for k,v in nt_to_index.items()}



# # Generate sentences from Chromosome 19

# In[ ]:


num_iterations = 50 #for each chromosome
max_num_sentences = 3000000
    
possible_start_idxs = list(range(4096))
possible_segment_lens = list(range(512, 4096))
dir_in='/data/keyang/pretraining_txt'
n_token_code = nt_biallele_code['N']


output_txt_path = '/data/keyang/pretraining_data.txt' # Output pretraining dataset file
out_file = open(output_txt_path, 'wb')

for filename in os.listdir(dir_in):
    input_txt_path =os.path.join(dir_in, filename)  # Generated file from `sample_chromosome_matrix.py`
    if os.path.isfile(input_txt_path):
        m=filename[13:15] #the chromosome number
        seq = open(input_txt_path, 'rb').read().decode('utf-8')
        s_idx = -1
        for i in range(len(seq)):
            if seq[i] != n_token_code:
                s_idx = i
                break
        e_idx = -1
        for i in range(len(seq) - 1, 0, -1):
            if seq[i] != n_token_code:
                e_idx = i
                break

         # Trim sequence
        trim_seq = seq[s_idx:e_idx + 1]
        print(f'chromosome{m} | len(seq) {len(seq)} | s_idx {s_idx} | e_idx {e_idx} | len(trim_seq) {len(trim_seq)}')

        seq = trim_seq
        start_idxs = np.random.choice(possible_start_idxs, num_iterations, replace=False)
        ch_len_clean=len(seq)

        for start_idx in tqdm.tqdm(start_idxs):
            s_idx = start_idx
            while s_idx < ch_len_clean:
                segment_len = np.random.choice(possible_segment_lens)
                segment = seq[s_idx:s_idx+segment_len]

                out_file.write((segment + '\n').encode('utf8'))
                s_idx = s_idx + segment_len
out_file.close()

# # Build BPE tokenizer vocab

output_txt_path = '/data/keyang/pretraining_data.txt'
filesize = 153845916535                 #size of the really big file
f = open(output_txt_path, 'r',encoding='utf-8', errors='ignore')

seqs = []
max_num_sentences = 1000000


for i in range(max_num_sentences):
    offset = random.randrange(filesize)
    f.seek(offset)  # go to random position
    f.readline()  # discard - bound to be partial line
    random_line=f.readline()  # bingo!
    # extra to handle last/first line edge cases
    if len(random_line) == 0:  # we have hit the end
        f.seek(0)
        random_line = f.readline()  # so we'll grab the first line instead
    seqs.append(random_line)


nt_biallele_code = json.load(open('./resource/snp_vocab.json', 'r'))
unigram_tokens = list(nt_biallele_code.values())


char_bpe_tokenizer = CharBPETokenizer()

print('start training tokenizer')
char_bpe_tokenizer.train_from_iterator(seqs[:max_num_sentences], special_tokens=['[UNK]','[CLS]','[SEP]'], vocab_size=20000, initial_alphabet=unigram_tokens)
char_bpe_tokenizer.save_model(directory='/data/keyang/tokenizers/', prefix=f'chr_diploid')






