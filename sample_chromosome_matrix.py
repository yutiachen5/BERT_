import numpy as np
import tqdm
import json
import torch
import json
import argparse
import itertools
import os
import scipy.sparse as sp

parser = argparse.ArgumentParser(description='Generate tokens from profile matrix')
parser.add_argument('--in_path', default='/data/keyang/pretraining_matrix/NC_000001.11.npz', type=str)
parser.add_argument('--out_path', default='/data/keyang/pretraining_txt/token_NC_000001.11.txt', type=str)
#parser.add_argument('--vocab_path', default='./resource/snp_vocab.json', type=str)
parser.add_argument('--nt_to_idx_path', default='./resource/nucleotide_to_index.json', type=str)
parser.add_argument('--batch_len', default=100000, type=int)
args = vars(parser.parse_args())

torch.set_grad_enabled(False)


nt_to_index = json.load(open(args['nt_to_idx_path'], 'r'))
index_to_nt = {v:k for k,v in nt_to_index.items()}

# Load Vocab
biallele=[i[0]+'_'+i[1]  for i in itertools.combinations(sorted(nt_to_index.keys()), 2) ]
first_chr=0x4E00
vocabs=[chr(first_chr+i) for i in range(len(biallele))]
nt_biallele_code={biallele[i]:vocabs[i] for i in range(len(biallele))}
nt_biallele_code.update({"A": "A", "G": "G", "T": "T", "C": "C", "N": "N",
                         "AI": "B", "GI": "H", "TI": "U", "CI": "D", "NI": "O",
                         "AD": "F", "GD": "I", "TD": "V", "CD": "E", "ND":"P"})


with open('./resource/snp_vocab.json', "w", encoding="utf-8") as outfile:
    json.dump(nt_biallele_code, outfile)
#nt_biallele_code = json.load(open(args['vocab_path'], 'r'))

dir_in = '/data/keyang/pretraining_matrix'
dir_out= '/data/keyang/pretraining_txt'

for filename in os.listdir(dir_in): # Loop over chromosome
    if os.path.isfile(os.path.join(dir_in, filename)):
        args['in_path'] =os.path.join(dir_in, filename)
        args['out_path'] = os.path.join(dir_out, 'token_'+filename.replace('npz','txt'))
        # Prepare input & output
        out_file = open(args['out_path'], 'wb')
        data = torch.from_numpy(sp.load_npz(args['in_path']).toarray())

        num_batch = (data.shape[0] // args['batch_len']) + 1
        for nb in tqdm.tqdm(range(num_batch)):
            # Build biallelic sequence
            s_idx, e_idx = nb * args['batch_len'], (nb + 1) * args['batch_len']
            values, indices = data[s_idx:e_idx,:].topk(2, dim=-1)
        #torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
            tokens = []
            for i, (vals, idxs) in enumerate(zip(values, indices)):
                token = []
                for val, idx in zip(vals, idxs):
                    if val > 0:
                        token.append(index_to_nt[int(idx)])
                tokens.append(nt_biallele_code['_'.join(sorted(token))])

            # Dump string to file
            str_tokens = ''.join(tokens)
            out_file.write(str_tokens.encode('utf8'))
            out_file.flush()

        # Close file
        out_file.close()