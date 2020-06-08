# -*- coding: utf-8 -*-
import os
import json

import torch
from torch.utils.data import DataLoader
from transformers import cached_path

from od.inputters.dataset_wb import WBDataset, WBdistDataset

LCCC_URL = "https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB.zip"
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]


def get_data(tokenizer, dataset_path, dataset_cache, logger):
    """ Get tokenized dataset from COTK or cache."""
    dataset_path = dataset_path or LCCC_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
        samples = None
    else:
        logger.info("Download dataset from %s", dataset_path)
        cache_file = cached_path(dataset_path)
        with open(cache_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
            samples = [{k: v[:5]} for k, v in dataset.items()]

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset, samples


def build_dataloaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    datasets, raw_samples = get_data(tokenizer, args.data_path, args.dataset_cache, logger)
    train_dataset, valid_dataset = WBDataset(datasets["train"], tokenizer), WBDataset(datasets["valid"], tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)

    return train_loader, valid_loader, train_sampler, valid_sampler


def build_dist_loaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    train_dataset = WBdistDataset(tokenizer, data_path=args.train_path)
    valid_dataset = WBdistDataset(tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler
