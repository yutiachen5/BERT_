import json

nt_biallele_code =json.load(open('./resource/snp_vocab.json', "r", encoding="utf-8"))

nts = json.load(open('./resource/nucleotide_to_index.json', 'r')).keys()
nt_nts=['_'.join([nt,nt]) for nt in nts]

first_chr=0x5E00
vocabs=[chr(first_chr+i) for i in range(len(nt_nts))]
nt_biallele_code1={nt_nts[i]:vocabs[i] for i in range(len(nt_nts))}

nt_biallele_code.update(nt_biallele_code1)

with open('./resource/snp_vocab1.json', "w", encoding="utf-8") as outfile:
    json.dump(nt_biallele_code, outfile, ensure_ascii=False)