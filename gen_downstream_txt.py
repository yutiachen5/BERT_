#!/usr/bin/env python
# coding: utf-8

# # Script for generating chromosome matrix from GRCh37 and dbSNP

# In[1]:

import pandas as pd
import time
import tqdm
from joblib import Parallel, delayed
import re
import json

#双侧


# # Load FASTA & SNP
input_fasta_path = '/data/keyang/hg38.fa' # Reference Fasta File
input_csv_path = '/data/keyang/OmicsSomaticMutations.csv' # dbSNP csv File
output_path = '/data/keyang/down_stream_data' # Output Folder Path

nt_biallele_code =json.load(open('./resource/snp_vocab.json', "r", encoding="utf-8"))

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


c38_dict = parse_fasta(input_fasta_path,'>')
c38_lens = [len(seq) for seq in c38_dict.values()]
print('Total Length GRch38', sum(c38_lens),'\nMin Length GRch38', min(c38_lens),'\nMax Length GRch38', max(c38_lens))


csv = pd.read_csv(input_csv_path)
cell_ids=list(pd.read_csv('common_cell_line_ids.csv').DepMap_ID)
csv_depmap=csv[['Chrom', 'Pos', 'Ref', 'Alt','DepMap_ID','GT']]
csv_depmap=csv_depmap.set_index('DepMap_ID')



fasta_chr_keyword = 'dna:chromosome'
fasta_chr_keys = [k for k in c38_dict.keys() if fasta_chr_keyword in k]
chr_order=[str(i+1) for i in range(22)]+['X','Y']
csv_chr_keys = ['chr'+i for i in chr_order]
fasta_chr_keys_d={i.split(' ')[0].replace('>',''):i for i in fasta_chr_keys}
fasta_chr_keys=[fasta_chr_keys_d[i]for i in chr_order]



chr_lens = []
for fasta_chr_key in fasta_chr_keys:
    chr_lens.append(len(c38_dict[fasta_chr_key]))





def parse_cell_chr(cell_id,csv_chr,fasta_seq,csv_chr_key):
    try:
        mutations=csv_chr.loc[cell_id] #df
        cell_seq = list(fasta_seq)  # Convert the string to a list

        for row in mutations.iloc:
            dict_row = dict(row)
            start_pos, ref, alt, gt = int(dict_row['Pos']) - 1, dict_row['Ref'], dict_row['Alt'], dict_row['GT']
            end_pos = start_pos + len(ref) - 1  # parse row

            # handle SNV and INDEL
            if len(alt) == len(ref):
                # SNV / MUTATION - mutated nucleotide
                for i, nt in enumerate(alt):
                    if nt != 'N':
                        if gt == '1|1':
                            #mutation in both chromatids if indicated
                            cell_seq[start_pos + i] = nt_biallele_code['_'.join([nt,nt])]
                        else:
                            cell_seq[start_pos + i] = nt_biallele_code['_'.join(sorted([nt, ref[i]]))]
            elif len(alt) < len(ref):
                # DELETION - non-deleted prefix
                for i, nt in enumerate(alt):
                    cell_seq[start_pos + i] = nt
                # assign proba to deleted suffix
                if gt == '1|1':
                    # mutation in both chromatids if indicated
                    for i in range(len(alt), len(ref)):
                        fasta_nt = ref[i]  # ref_nt
                        cell_seq[start_pos + i] = nt_biallele_code['_'.join([fasta_nt + 'D',fasta_nt + 'D'])]
                else:
                    for i in range(len(alt), len(ref)):
                        fasta_nt = ref[i]  # ref_nt
                        cell_seq[start_pos + i] = nt_biallele_code['_'.join(sorted([fasta_nt + 'D',ref[i]]))]
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
                                if gt == '1|1':
                                    cell_seq[start_pos + i] = nt_biallele_code[
                                        '_'.join([f'{alt[i + m]}I', f'{alt[i + m]}I'])]
                                else:
                                    cell_seq[start_pos + i] = nt_biallele_code[
                                        '_'.join([f'{alt[i + m]}I', ref[i]])]
                                for j in range(1, len(alt) - i - m):
                                    if ref[i] == alt[i + m + j]:
                                        m += j
                                        break
                            else:
                                unchanged.append(i)
                        for i in unchanged:
                            cell_seq[start_pos + i] = ref[i]
                    else:  # ref seq incomplete
                        if gt =='1|1':
                            for i in range(len(ref) - 1):
                                cell_seq[start_pos + i] = nt_biallele_code['_'.join([alt[i],alt[i]])]
                            # assign insertion proba to the last nucleotide
                            cell_seq[start_pos + len(ref) - 1] = nt_biallele_code['_'.join([f'{alt[len(ref) - 1]}I',f'{alt[len(ref) - 1]}I'])]
                        else:
                            for i in range(len(ref) - 1):
                                cell_seq[start_pos + i] = nt_biallele_code['_'.join([alt[i],ref[i]])]
                            cell_seq[start_pos + len(ref) - 1] = nt_biallele_code[
                                '_'.join([f'{alt[len(ref) - 1]}I', ref[len(ref) - 1]])]
                else:  # too long to process
                    if gt == '1|1':
                        for i in range(len(ref) - 1):
                            cell_seq[start_pos + i] = nt_biallele_code['_'.join([alt[i], alt[i]])]
                        # assign insertion proba to the last nucleotide
                        cell_seq[start_pos + len(ref) - 1] = nt_biallele_code[
                            '_'.join([f'{alt[len(ref) - 1]}I', f'{alt[len(ref) - 1]}I'])]
                    else:
                        for i in range(len(ref) - 1):
                            cell_seq[start_pos + i] = nt_biallele_code['_'.join([alt[i], ref[i]])]
                        cell_seq[start_pos + len(ref) - 1] = nt_biallele_code[
                            '_'.join([f'{alt[len(ref) - 1]}I', ref[len(ref) - 1]])]
        cell_seq_string = ''.join(cell_seq)
        with open(f"{output_path}/{cell_id}_{csv_chr_key}.txt", mode="w") as file:
            file.write(cell_seq_string)
    except KeyError:
        print(cell_id)





# Loop over Chromosome
for fasta_chr_key, csv_chr_key, chr_len in zip(fasta_chr_keys, csv_chr_keys, chr_lens):
    start_time = time.time()    
    
    fasta_seq = c38_dict[fasta_chr_key]  # Fasta reference
    csv_chr=csv_depmap[csv_depmap['Chrom']== csv_chr_key]
    
    results = Parallel(n_jobs=4)(
    delayed(parse_cell_chr)(cell_id,csv_chr,fasta_seq,csv_chr_key) for cell_id in tqdm.tqdm(cell_ids, desc='Processing')
)
    for cell_id in tqdm.tqdm(cell_ids):
        parse_cell_chr(cell_id,csv_chr,fasta_seq,csv_chr_key)
    
    print(f'Finish processing chromosome {csv_chr_key} | Elapsed time : {time.time() - start_time}s')







