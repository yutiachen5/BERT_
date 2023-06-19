
import argparse
import collections
import torch
import numpy as np

import DataLoader.dataloader as module_data
# from data.utility import DatasetSplit
# from model.bert_pretrain import get_bert_model
# from model.metric import metrics
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed = config['dataset']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    holdout = config['dataset']['args']['test_split']
    config['dataset']['args']['config'] = config
    config['dataset']['args']['logger'] = logger
    # snp_embedding = snp_emb.main(config['dataset']['args']['snp_dir'])
    dataset = config.init_obj('dataset', module_data)
    train_dataset = dataset.get_dataset()
    test_dataset = None

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)




