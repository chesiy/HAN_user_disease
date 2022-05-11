from collections import defaultdict
from copyreg import pickle
import imp
import os
import re
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from scipy.io import loadmat
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict
import pickle
from tqdm import tqdm
import random

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]
disease2id = {disease:id for id,disease in enumerate(id2disease)}

class FlatDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train"):
        assert split in {"train", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split)
        for fname in os.listdir(input_dir2):
            label = float(fname[-5])   # "xxx_0.txt"/"xxx_1.txt"
            sample = {}
            sample["text"] = open(os.path.join(input_dir2, fname), encoding="utf-8").read()
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_flat(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(labels)
    return processed_batch, labels

class FlatDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        elif stage == "test":
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_flat, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)

class HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", max_posts=64):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        input_dir2 = os.path.join(input_dir, split+'.pkl')
        with open(input_dir2, 'rb') as f:
            raw_data = pickle.load(f)
            # raw_data = raw_data[:int(len(raw_data)/10)]
            raw_data = random.sample(raw_data, int(len(raw_data)/10))
        for record in tqdm(raw_data):
            label = np.zeros(len(id2disease))
            for disease in record['diseases']:
                if disease not in ['autism', 'schizophrenia']:
                    label[disease2id[disease]] = 1
            sample = {}
            posts = record['selected_posts'][:max_posts]
            tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_hier(data):
    labels = []
    processed_batch = []
    for item, label in data:
        user_feats = {}
        for k, v in item.items():
            user_feats[k] = torch.LongTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    return processed_batch, labels

class HierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.val_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "val")
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test")
        elif stage == "test":
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, shuffle=True, pin_memory=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_hier, pin_memory=False, num_workers=4)

if __name__ == "__main__":
    model_id = 'prajjwal1/bert-tiny'
    # note that we need to specify the number of classes for this task
    # we can directly use the metadata (num_classes) stored in the dataset
    # model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                # num_labels=train_dataset.features["label"].num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # data_module = FlatDataModule(4, "./processed/kmeans8", tokenizer, max_len=510)
    # data_module.setup("fit")
    # train_loader = data_module.train_dataloader()
    # batch0, labels0 = next(iter(train_loader))
    # print(batch0)
    # print(labels0)
    # import pdb; pdb.set_trace()

    data_module = HierDataModule(4, "./processed/description_sim16/", tokenizer, max_len=128)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    batch0, labels0 = next(iter(train_loader))
    print(batch0)
    print(labels0)
    # import pdb; pdb.set_trace()