#!/usr/bin/env python
# coding: utf-8

# # Script for generating chromosome matrix from GRCh37 and dbSNP

# In[1]:


import os, sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import time
import tqdm


# In[ ]:





# # Load FASTA & SNP

# In[8]:


input_fasta_path = '/data/keyang/hg38.fa' # Reference Fasta File
input_csv_path = '/data/keyang/OmicsSomaticMutations.csv' # dbSNP csv File

output_path = '/data/keyang/down_stream_data' # Output Folder Path


# In[ ]:





# In[3]:


def parse_fasta(path, header_identifier):
    # Read file
    lines = open(path,'r').readlines()
    lines = list(map(lambda x: x[:-1], lines)) # remove \n
    
    # Concatenate sequence data per chromosome
    chromosome_dict = {}
    chromosome_name = None
    chromosome_sequence = ''
    for line in lines:
        if header_identifier in line:
            if chromosome_name:
                chromosome_dict[chromosome_name] = ''.join(chromosome_sequence)
            # Reinitialize name & sequence
            chromosome_name = line[:-1]        
            chromosome_sequence = []
        else:
            # Fill Sequence
            chromosome_sequence.append(line)
            
    # Add the last chromosome
    if chromosome_name:
        chromosome_dict[chromosome_name] = ''.join(chromosome_sequence)
        
    return chromosome_dict


# In[9]:


c38_dict = parse_fasta(input_fasta_path,'>')
c38_lens = [len(seq) for seq in c38_dict.values()]
print('Total Length GRch38', sum(c38_lens))
print('Min Length GRch38', min(c38_lens))
print('Max Length GRch38', max(c38_lens))


# In[18]:


csv = pd.read_csv(input_csv_path)
cell_ids=list(csv.DepMap_ID.drop_duplicates())
csv_depmap=csv[['Chrom', 'Pos', 'Ref', 'Alt','DepMap_ID']]
csv_depmap=csv_depmap.set_index('DepMap_ID')


# In[ ]:





# In[11]:


fasta_chr_keyword = 'dna:chromosome'
fasta_chr_keys = [k for k in c38_dict.keys() if fasta_chr_keyword in k]


# In[12]:


chr_order=[str(i+1) for i in range(22)]+['X','Y']
csv_chr_keys = ['chr'+i for i in chr_order]


# In[13]:


fasta_chr_keys_d={i.split(' ')[0].replace('>',''):i for i in fasta_chr_keys}
fasta_chr_keys=[fasta_chr_keys_d[i]for i in chr_order]


# In[14]:


chr_lens = []
for fasta_chr_key in fasta_chr_keys:
    chr_lens.append(len(c38_dict[fasta_chr_key]))


# In[ ]:





# In[16]:


def parse_cell_chr(cell_id,csv_chr,fasta_seq,csv_chr_key):
    mutations=csv_chr.loc[cell_id] #df
    cell_seq = list(fasta_seq)# Convert the string to a list

    for row in mutations.iloc:
        
        dict_row = dict(row)
        start_pos, ref, alt = int(dict_row['Pos']) - 1, dict_row['Ref'], dict_row['Ref'] + dict_row['Alt']
        end_pos= start_pos + len(ref) -1 # parse row
        cell_seq[start_pos:end_pos+1]=alt

    cell_seq_string=''.join(cell_seq)
    with open(f"down_stream_data/raw/{cell_id}_{csv_chr_key}.txt", mode="w") as file:
        file.write(cell_seq_string)


# In[19]:


get_ipython().run_cell_magic('time', '', "from joblib import Parallel, delayed\nimport multiprocessing\n\nnum_cores = multiprocessing.cpu_count()\n\n# Loop over Chromosome\nfor fasta_chr_key, csv_chr_key, chr_len in zip(fasta_chr_keys, csv_chr_keys, chr_lens):\n    start_time = time.time()    \n    \n    fasta_seq = c38_dict[fasta_chr_key]  # Fasta reference\n    csv_chr=csv_depmap[csv_depmap['Chrom']== csv_chr_key]\n    \n    results = Parallel(n_jobs=6)(\n    delayed(parse_cell_chr)(cell_id,csv_chr,fasta_seq,csv_chr_key) for cell_id in tqdm.tqdm(cell_ids, desc='Processing')\n)\n    for cell_id in tqdm.tqdm(cell_ids):\n        parse_cell_chr(cell_id,csv_chr,fasta_seq,csv_chr_key)\n    \n    print(f'Finish processing chromosome {csv_chr_key} | Elapsed time : {time.time() - start_time}s')")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




