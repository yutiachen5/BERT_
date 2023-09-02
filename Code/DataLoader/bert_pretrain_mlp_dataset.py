# -*- coding: utf-8 -*-

import math
from tokenize import Token
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE
# from bert_data_prepare.tokenizer import get_tokenizer
from os.path import join
from transformers import PreTrainedTokenizerFast
import json


def min_power_greater_than(value, base=2):
    """
    Return the lowest power of the base that exceeds the given value
    >>> min_power_greater_than(3, 4)
    4.0
    >>> min_power_greater_than(48, 2)
    64.0
    """
    p = math.ceil(math.log(value, base))
    return math.pow(base, p)


class SelfSupervisedDataset(Dataset):
    """
    Mostly for compatibility with transformers library
    LineByLineTextDataset returns a dict of "input_ids" -> input_ids
    """

    # Reference: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/data/datasets/language_modeling.py
    def __init__(self, seqs,
                 # split_fun,
                 tokenizer,
                 max_len,
                 logger,
                 round_len=True):
        self.seqs = seqs
        # self.split_fun = split_fun
        self.logger = logger
        self.tokenizer = tokenizer

        self.logger.info(
            f"Creating self supervised dataset with {len(self.seqs)} sequences")

        self.max_len = max_len
        self.logger.info(f"Maximum sequence length: {self.max_len}")

        if round_len:
            self.max_len = int(min_power_greater_than(self.max_len, 2))
            self.logger.info(f"Rounded maximum length to {self.max_len}")
        self._has_logged_example = False

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        #retval = self.tokenizer.encode(self._insert_whitespace(self.split_fun(seq)),
         #                              truncation=True, max_length=self.max_len)
        retval = self.tokenizer.encode(seq,truncation=True, max_length=self.max_len)
        if not self._has_logged_example:
            self.logger.info(f"Example of tokenized input: {seq} -> {retval}")
            self._has_logged_example = True
        return {"input_ids": torch.tensor(retval, dtype=torch.long)}

    def merge(self, other):
        """Merge this dataset with the other dataset"""
        all_seqs = self.seqs + other.seqs
        self.logger.info(
            f"Merged two self-supervised datasets of sizes {len(self)} {len(other)} for dataset of {len(all_seqs)}")
        return SelfSupervisedDataset(all_seqs)

    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)


class MLPDataset(object):
    def __init__(self,
                 config,
                 logger,
                 seed,
                 seq_dir,
                 vocab_dir,
                 prefix,
                 max_len=None,
                 test_split=0.1):
        self.token_with_special_list = None
        self.vocab_dict = None
        self.config = config
        self.seq_dir = seq_dir
        self.logger = logger
        self.seed = seed
        self.test_split = test_split

        self.seq_list = self._load_seq()

        self.logger.info('Start creating tokenizer...')
        # self.tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
        #                               add_hyphen=False,
        #                               logger=self.logger,
        #                               vocab_dir=vocab_dir,
        #                               token_length_list=token_length_list)

        self.vocab_filename = join(vocab_dir, "{}-vocab.json".format(prefix))
        self.merges_filename = join(vocab_dir, "{}-merges.txt".format(prefix))

        # self.tokenizer = CharBPETokenizer(vocab_filename,merges_filename,)

        vocab, merges = BPE.read_file(self.vocab_filename, self.merges_filename)
        bpe = BPE(vocab, merges)
        self.tokenizer = Tokenizer(bpe)

        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"
        self.tokenizer.add_tokens([self.PAD,self.MASK,self.UNK,self.SEP,self.CLS])

        # self.split_fun = self.tokenizer.split

        if max_len is None:
            self.max_len = max([len(self.split_fun(s)) for s in self.seq_list])
        else:
            self.max_len = max_len
        #max_len_rounded = min_power_greater_than(self.max_len, base=2)

        self.bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer,
                                        model_max_length=max_len,
                                        pad_token=self.PAD,
                                        mask_token=self.MASK,
                                        unk_token=self.UNK,
                                        sep_token=self.SEP,
                                        cls_token=self.CLS,
                                        padding_side="right",
                                        add_special_tokens=True )
        self.bert_tokenizer.save_pretrained(config._save_dir)

    def get_token_list(self):
        return self.token_with_special_list

    def get_vocab_size(self):
        with open(self.vocab_filename, 'r', encoding='utf-8') as file:
            vocab_dict0 = json.load(file)

        special_token = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]
        token_list = list(vocab_dict0.keys())
        self.token_with_special_list = token_list+special_token

        self.vocab_dict = {t: i for i, t in enumerate(self.token_with_special_list)}
        return len(self.vocab_dict)

    def get_pad_token_id(self):
        return self.vocab_dict[self.PAD]

    def get_tokenizer(self):
        return self.bert_tokenizer

    def _load_seq(self):
        # seq_df = pd.read_csv(self.seq_dir)
        with open(self.seq_dir, 'r', encoding='utf-8') as file:
            seq_list = []
            for line in file:
                line1 = line.strip()
                if line1 != '':
                    seq_list.append(line1)
        self.logger.info(f'Load {len(seq_list)} form {self.seq_dir}.')
        return seq_list

    def _split(self):
        train, test = train_test_split(self.seq_list, test_size=self.test_split, random_state=self.seed)
        return train, test

    def get_dataset(self):
        self_supvervised_dataset = SelfSupervisedDataset(seqs=self.seq_list,
                                                         # split_fun=self.split_fun,
                                                         tokenizer=self.bert_tokenizer,
                                                         max_len=self.max_len,
                                                         logger=self.logger,
                                                         round_len=True)
        return self_supvervised_dataset
