#!/usr/bin/env python
# coding: utf-8

# # Script for generating chromosome matrix from GRCh37 and dbSNP

# In[ ]:


import os, sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import pysam
import time
import tqdm
import re
import scipy.sparse

# # Load FASTA & Tabix

# In[ ]:

relative_path='../../../data/keyang/'
input_fasta_path = relative_path+'hs37d5.fa' # Reference Fasta File
input_tabix_path = relative_path+'GCF_000001405.25.gz' # dbSNP Tabix File
output_path = relative_path+'pretraining_data' # Output Folder Path


# In[ ]:


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


# In[ ]:


#get_ipython().run_cell_magic('time', '', "
c37_dict = parse_fasta(input_fasta_path,'>')
c37_lens = [len(seq) for seq in c37_dict.values()]
print('Total Length GRch37', sum(c37_lens))
print('Min Length GRch37', min(c37_lens))
print('Max Length GRch37', max(c37_lens))


# In[ ]:


tb_file = pysam.TabixFile(input_tabix_path)


# # Filter only chromosome contigs for FASTA and tabix

# In[ ]:


fasta_chr_keyword = 'dna:chromosome'
fasta_chr_keys = [k for k in c37_dict.keys() if fasta_chr_keyword in k]


# In[ ]:


tb_chr_keyword = 'NC_0000'
tabix_chr_keys = [k for k in tb_file.contigs if tb_chr_keyword in k]


# In[ ]:


chr_lens = []
for fasta_chr_key in fasta_chr_keys:
    chr_lens.append(len(c37_dict[fasta_chr_key]))


# # Construct the chromosome matrix for each chromosome

# In[ ]:


def parse_freq(freq_str):
    proba_list = []
    for src_proba_str in freq_str.split('|'):
        probas = src_proba_str.split(':')[-1].split(',')
        proba_list.append([0.0 if proba == '.' else float(proba) for proba in probas])
        
    # Calculate proba
    probas = np.array(proba_list).mean(axis=0)    
    return probas
    
def parse_vcf_row(vcf_str):
    # parse vcf row
    cols = vcf_str.split('\t')
    snv_pos, ref, alts = int(cols[1]) - 1, cols[3], np.array([cols[3]] + cols[4].split(','))
    
    # retrieve probas from freq info
    freq_str = cols[-1].split('FREQ=')[-1].split(';')[0]
    probas = parse_freq(freq_str)
    
    # filter ref_alt and probas
    alts, probas = alts[probas > 0], probas[probas > 0]
    
    return snv_pos, snv_pos + len(ref) - 1, ref, alts, probas

# Nucleotide to index (https://en.wikipedia.org/wiki/FASTA_format#Sequence_representation)
nt_to_index = { 
    'A': [0], 'G': [1], 'T': [2], 'C': [3], 'N': [4], #'DEL': [5], 
    'AI': [6], 'GI': [7], 'TI': [8], 'CI': [9], 'NI': [10],
    'AD': [11], 'GD': [12], 'TD': [13], 'CD': [14], 'ND': [5],
    'U': [2], 'R': [0, 1], 'Y': [2, 3], 'K': [1, 2], 'M': [0, 3], 'S': [1, 3], 'W': [0, 2],
    'UI': [8], 'RI': [6, 7], 'YI': [8, 9], 'KI': [7, 8], 'MI': [6, 9], 'SI': [7, 9], 'WI': [6, 8],
    'UD': [13], 'RD': [11, 12], 'YD': [13, 14], 'KD': [12, 13], 'MD': [11, 14], 'SD': [12, 14], 'WD': [11, 13],
    'B': [1, 2, 3], 'H': [0, 2, 3], 'V': [0, 1, 3], 'D': [0, 1, 2],
    'BI': [7, 8, 9], 'HI': [6, 8, 9], 'VI': [6, 7, 9], 'DI': [6, 7, 8],
    'BD': [12, 13, 14], 'HD': [11, 13, 14], 'VD': [11, 12, 14], 'DD': [11, 12, 13],
}


# In[ ]:


#get_ipython().run_cell_magic('time', '', "# Loop over Chromosome\nbatch_len = 100000\nfor fasta_chr_key, tabix_chr_key, chr_len in zip(fasta_chr_keys, tabix_chr_keys, chr_lens):\n    start_time = time.time()    \n    \n    fasta_seq = c37_dict[fasta_chr_key]  # Fasta reference\n    data_block = np.zeros((chr_len, 11))  # Target block\n    \n    num_batch = (chr_len // batch_len) + 1\n    last_next_pos = 0\n    for nb in tqdm.tqdm(range(num_batch)):\n        start_query_pos, end_query_pos = nb * batch_len, (nb + 1) * batch_len\n        for row in tb_file.fetch(tabix_chr_key, start_query_pos, end_query_pos):\n            if ';COMMON' in row:\n                # parse vcf row\n                start_pos, end_pos, ref, alts, probas = parse_vcf_row(row)\n\n                # handle SNV and INDEL\n                for alt, proba in zip(alts, probas):\n                    if len(alt) == len(ref):\n                        # SNV / MUTATION - assign proba to mutated nucleotide\n                        for i, nt in enumerate(alt):\n                            if nt is not 'N':\n                                data_block[start_pos + i, nt_to_index[nt]] += proba / len(nt_to_index[nt])\n                    elif len(alt) < len(ref):\n                        # DELETION\n                        # assign proba to non-deleted prefix\n                        for i, nt in enumerate(alt):\n                            data_block[start_pos + i, nt_to_index[nt]] += proba / len(nt_to_index[nt])\n\n                        # assign proba to deleted suffix\n                        for i in range(len(alt), len(ref)):\n                            data_block[start_pos + i, nt_to_index['DEL']] += proba / len(nt_to_index['DEL'])\n                            \n                    else: # if len(alt) > len(ref):\n                        # INSERTION\n                        # assign proba to the prefix nucleotide\n                        for i in range(len(ref) - 1):\n                            data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(nt_to_index[alt[i]])\n                            \n                        # assign insertion proba to the last nucleotide\n                        data_block[start_pos+len(ref)-1, nt_to_index[f'{alt[len(ref)-1]}I']] += proba / len(nt_to_index[f'{alt[len(ref)-1]}I'])\n\n                # assign reference to region between SNV\n                for i in range(last_next_pos, start_pos):\n                    fasta_nt = fasta_seq[i]\n                    data_block[i, nt_to_index[fasta_nt]] = 1 / len(nt_to_index[fasta_nt])\n\n                # assign new last next position\n                last_next_pos = start_pos + len(ref)\n\n    # assign from last next position onward with nucleotide from reference\n    for i in range(last_next_pos, chr_len):\n        fasta_nt = fasta_seq[i]\n        data_block[i, nt_to_index[fasta_nt]] = 1 / len(nt_to_index[fasta_nt])\n    \n    # Normalize over 11D\n    data_block = data_block / data_block.sum(axis=1, keepdims=True)\n    \n    # Dump chromosome data    \n    np.save(f'{output_path}/{tabix_chr_key}.npy', data_block)\n    print(f'Finish processing chromosome {tabix_chr_key} | Elapsed time : {time.time() - start_time}s')")


# In[ ]:

#% % time
# Loop over Chromosome
batch_len = 100000
for fasta_chr_key, tabix_chr_key, chr_len in zip(fasta_chr_keys, tabix_chr_keys, chr_lens):
    start_time = time.time()

    fasta_seq = c37_dict[fasta_chr_key]  # Fasta reference
    data_block = np.zeros((chr_len, 15))  # Target block

    num_batch = (chr_len // batch_len) + 1
    last_next_pos = 0
    for nb in tqdm.tqdm(range(num_batch)):
        start_query_pos, end_query_pos = nb * batch_len, (nb + 1) * batch_len
        try:
            for row in tb_file.fetch(tabix_chr_key, start_query_pos, end_query_pos):
                if ';COMMON' in row:
                    # parse vcf row
                    start_pos, end_pos, ref, alts, probas = parse_vcf_row(row)

                    # handle SNV and INDEL
                    for alt, proba in zip(alts, probas):
                        if len(alt) == len(ref):
                            # SNV / MUTATION - assign proba to mutated nucleotide
                            for i, nt in enumerate(alt):
                                if nt != 'N':
                                    data_block[start_pos + i, nt_to_index[nt]] += proba / len(nt_to_index[nt])
                        elif len(alt) < len(ref):
                            # DELETION
                            # assign proba to non-deleted prefix
                            for i, nt in enumerate(alt):
                                data_block[start_pos + i, nt_to_index[nt]] += proba / len(nt_to_index[nt])

                            # assign proba to deleted suffix
                            for i in range(len(alt), len(ref)):
                                fasta_nt = fasta_seq[i] #ref_nt
                                data_block[start_pos + i, nt_to_index[fasta_nt]] += proba / len(nt_to_index[fasta_nt])

                        else:  # if len(alt) > len(ref):
                            # INSERTION
                            if len(alt)<=20: #to save time
                                m=0
                                ref_list=[c for c in ref]
                                ref_1='[ATCG]*'.join(ref_list)
                                search_obj=re.search(ref_1,alt)
                                if search_obj:
                                    unchanged=[]
                                    for i in range(len(ref)):
                                        if ref[i]!=alt[i+m]:
                                            # assign insertion proba to the every nucleotide insertion
                                            #the first nb in alt every time inerstion occurs
                                            data_block[start_pos + i, nt_to_index[f'{alt[i+m]}I']] += proba / len(nt_to_index[f'{alt[i+m]}I'])
                                            for j in range(1,len(alt)-i-m):
                                                if ref[i]==alt[i+m+j]:
                                                    m+=j
                                                    break
                                        else:
                                            unchanged.append(i)
                                    for i in unchanged:
                                        data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(nt_to_index[alt[i]])
                                else: #ref seq incomplete
                                    for i in range(len(ref) - 1):
                                        data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(nt_to_index[alt[i]])
                                    # assign insertion proba to the last nucleotide
                                    data_block[start_pos + len(ref) - 1, nt_to_index[f'{alt[len(ref) - 1]}I']] += proba / len(nt_to_index[f'{alt[len(ref) - 1]}I'])
                            else: #too long to process
                                for i in range(len(ref) - 1):
                                    data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(nt_to_index[alt[i]])
                                # assign insertion proba to the last nucleotide
                                data_block[start_pos + len(ref) - 1, nt_to_index[f'{alt[len(ref) - 1]}I']] += proba / len(nt_to_index[f'{alt[len(ref) - 1]}I'])

                    # assign reference to region between SNV
                    for i in range(last_next_pos, start_pos):
                        fasta_nt = fasta_seq[i]
                        data_block[i, nt_to_index[fasta_nt]] = 1 / len(nt_to_index[fasta_nt])

                    # assign new last next position
                    last_next_pos = start_pos + len(ref)
        except UnicodeDecodeError as e:
            try:
                for row in tb_file.fetch(tabix_chr_key, start_query_pos+e.start+1, end_query_pos):
                    if ';COMMON' in row:
                        # parse vcf row
                        start_pos, end_pos, ref, alts, probas = parse_vcf_row(row)

                        # handle SNV and INDEL
                        for alt, proba in zip(alts, probas):
                            if len(alt) == len(ref):
                                # SNV / MUTATION - assign proba to mutated nucleotide
                                for i, nt in enumerate(alt):
                                    if nt != 'N':
                                        data_block[start_pos + i, nt_to_index[nt]] += proba / len(nt_to_index[nt])
                            elif len(alt) < len(ref):
                                # DELETION
                                # assign proba to non-deleted prefix
                                for i, nt in enumerate(alt):
                                    data_block[start_pos + i, nt_to_index[nt]] += proba / len(nt_to_index[nt])

                                # assign proba to deleted suffix
                                for i in range(len(alt), len(ref)):
                                    fasta_nt = fasta_seq[i]  # ref_nt
                                    data_block[start_pos + i, nt_to_index[fasta_nt]] += proba / len(nt_to_index[fasta_nt])

                            else:  # if len(alt) > len(ref):
                                # INSERTION
                                if len(alt) <= 20:  # to save time
                                    m = 0
                                    ref_list = [c for c in ref]
                                    ref_1 = '[ATCG]*'.join(ref_list)
                                    search_obj = re.search(ref_1, alt)
                                    if search_obj:
                                        unchanged = []
                                        for i in range(len(ref)):
                                            if ref[i] != alt[i + m]:
                                                # assign insertion proba to the every nucleotide insertion
                                                # the first nb in alt every time inerstion occurs
                                                data_block[start_pos + i, nt_to_index[f'{alt[i + m]}I']] += proba / len(
                                                    nt_to_index[f'{alt[i + m]}I'])
                                                for j in range(1, len(alt) - i - m):
                                                    if ref[i] == alt[i + m + j]:
                                                        m += j
                                                        break
                                            else:
                                                unchanged.append(i)
                                        for i in unchanged:
                                            data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(
                                                nt_to_index[alt[i]])
                                    else:  # ref seq incomplete
                                        for i in range(len(ref) - 1):
                                            data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(
                                                nt_to_index[alt[i]])
                                        # assign insertion proba to the last nucleotide
                                        data_block[
                                            start_pos + len(ref) - 1, nt_to_index[f'{alt[len(ref) - 1]}I']] += proba / len(
                                            nt_to_index[f'{alt[len(ref) - 1]}I'])
                                else:  # too long to process
                                    for i in range(len(ref) - 1):
                                        data_block[start_pos + i, nt_to_index[alt[i]]] += proba / len(nt_to_index[alt[i]])
                                    # assign insertion proba to the last nucleotide
                                    data_block[
                                        start_pos + len(ref) - 1, nt_to_index[f'{alt[len(ref) - 1]}I']] += proba / len(
                                        nt_to_index[f'{alt[len(ref) - 1]}I'])

                        # assign reference to region between SNV
                        for i in range(last_next_pos, start_pos):
                            fasta_nt = fasta_seq[i]
                            data_block[i, nt_to_index[fasta_nt]] = 1 / len(nt_to_index[fasta_nt])

                        # assign new last next position
                        last_next_pos = start_pos + len(ref)
            except UnicodeDecodeError as e:
                continue


    # assign from last next position onward with nucleotide from reference
    for i in range(last_next_pos, chr_len):
        fasta_nt = fasta_seq[i]
        data_block[i, nt_to_index[fasta_nt]] = 1 / len(nt_to_index[fasta_nt])

    # Normalize over 15D
    data_block = data_block / data_block.sum(axis=1, keepdims=True)
    print('normalized!')

    # Dump chromosome data
    # np.save(f'{output_path}/{tabix_chr_key}.npy', data_block) #scipy稀疏 -》data
    sparse_matrix = scipy.sparse.csr_matrix(data_block)
    scipy.sparse.save_npz(f'{output_path}/{tabix_chr_key}.npz', sparse_matrix)
    print(f'Finish processing chromosome {tabix_chr_key} | Elapsed time : {time.time() - start_time}s')


