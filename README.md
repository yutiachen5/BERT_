# BERT_

### procedures for DNA sequence processing
1. gen_chromosome_matrix.py
- input1 (reference): /data/keyang/hg38.fa
- input2 (snp): /data/keyang/GCF_000001405.40.gz
- output (as input of next step): /data/keyang/pretraining_matrix/{tabix_chr_key}.npz

2. sample_chromosome_matrix.py [running on server now]
- input: ./resource/nucleotide_to_index.json
- output (data, as input of next step): /data/keyang/pretraining_txt/token_{tabix_chr_key}.txt
- output (vocabulary, as input of next step): ./resource/snp_vocab.json

3. gen_dataset_tokenizer.py
- output (a large file of DNA sequences): /data/keyang/pretraining_data.txt
- output (tokenizer model): /data/keyang/tokenizers/

4. update_vocab_json.py
- There is no 'A_A', 'C_C' terms in SNP inputs.
- But the items are possible to exist in cell line sequences,
- representing both of the 2 chromatids are mutated.
- output (complete vocabulary, as input of next step): ./resource/snp_vocab1.json

5. gen_downstream_txt.py [editing]


