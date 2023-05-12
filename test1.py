'''
Commented with UTF-8
Modified & commented by Wenjie Luo
'''

import argparse
import os
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from data.dataset import PoisonLabelDataset, MixMatchDataset
from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from model.model import LinearModel
from model.utils import (
    get_criterion,
    get_network,
    get_optimizer,
    get_scheduler,
    load_state,
)
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.log import result2csv
from utils.trainer.semi import mixmatch_train, linear_test, poison_linear_record


print("===Setup running===")
# 获取参数
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="./config/baseline_asd.yaml")
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    help="checkpoint name (empty string means the latest checkpoint)\
        or False (means training from scratch).",
)
parser.add_argument("--amp", default=False, action="store_true")
parser.add_argument(
    "--world-size",
    default=1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument("--rank",
                    default=0,
                    type=int,
                    help="node rank for distributed training")
parser.add_argument(
    "--dist-port",
    default="23456",
    type=str,
    help="port used to set up distributed training",
)
args = parser.parse_args()
config, inner_dir, config_name = load_config(args.config)
# 准备用于存储数据和日志的文件夹（默认为 ./saved_data/baseline_asd.yaml）
args.saved_dir, args.log_dir = get_saved_dir(config, inner_dir,
                                             config_name, args.resume)
# 将数据存储到相应文件夹下
shutil.copy2(args.config, args.saved_dir)
args.storage_dir, args.ckpt_dir, _ = get_storage_dir(
    config, inner_dir, config_name, args.resume)
shutil.copy2(args.config, args.storage_dir)
