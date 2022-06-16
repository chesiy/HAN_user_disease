from collections import defaultdict
from copyreg import pickle
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
from torch.utils.data import Sampler

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

class BalanceSampler(Sampler):
    def __init__(self, data_source, control_ratio=0.75) -> None:
        self.data_source = data_source
        self.control_ratio = control_ratio
        self.indexes_control = np.where(data_source.is_control == 1)[0]
        self.indexes_mental = np.where(data_source.is_control == 0)[0]
        self.len_control = len(self.indexes_control)
        self.len_mental = len(self.indexes_mental)
        np.random.shuffle(self.indexes_control)
        np.random.shuffle(self.indexes_mental)

        self.pointer_control = 0
        self.pointer_mental = 0

    def __iter__(self):
        for i in range(len(self.data_source)):
            if np.random.rand() < self.control_ratio:
                id0 = np.random.randint(self.pointer_control, self.len_control)
                sel_id = self.indexes_control[id0]
                self.indexes_control[id0], self.indexes_control[self.pointer_control] = self.indexes_control[self.pointer_control], self.indexes_control[id0]
                self.pointer_control += 1
                if self.pointer_control >= self.len_control:
                    self.pointer_control = 0
                    np.random.shuffle(self.indexes_control)
            else:
                id0 = np.random.randint(self.pointer_mental, self.len_mental)
                sel_id = self.indexes_mental[id0]
                self.indexes_mental[id0], self.indexes_mental[self.pointer_mental] = self.indexes_mental[self.pointer_mental], self.indexes_mental[id0]
                self.pointer_mental += 1
                if self.pointer_mental >= self.len_mental:
                    self.pointer_mental = 0
                    np.random.shuffle(self.indexes_mental)
            
            yield sel_id

    def __len__(self) -> int:
        return len(self.data_source)

class HierDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train", disease='None', use_symp=False, max_posts=64):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_posts = max_posts
        self.data = []
        self.labels = []
        self.is_control = []
        input_dir2 = os.path.join(input_dir, split+'.pkl')
        with open(input_dir2, 'rb') as f:
            raw_data = pickle.load(f)
            # raw_data = raw_data[:int(len(raw_data)/10)]
            # raw_data = random.sample(raw_data, int(len(raw_data)/100))
        # print(len(raw_data))
        for record in tqdm(raw_data):
            if len(record['diseases']) == 0:
                # control are all 0
                label = np.zeros((len(id2disease),))
                self.is_control.append(1)
            else:
                # treat other not diagnosed diseases as -1 instead of 0
                label = np.array([1 if dis in record['diseases'] and (dis == disease or disease == 'None') else -1 for dis in id2disease])
                if 1 in label:
                    self.is_control.append(0)
                else:
                    self.is_control.append(1)
            sample = {}
            posts = record['selected_posts'][:max_posts]
            tokenized = tokenizer(posts, truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            if use_symp:
                if len(record['symp_probs']) < 16:
                    record['symp_probs'] = np.concatenate([record['symp_probs'], np.zeros((16-record['symp_probs'].shape[0], record['symp_probs'].shape[1]))])
                sample['symp'] = record['symp_probs']
            self.data.append(sample)
            self.labels.append(label)
        self.is_control = np.array(self.is_control).astype(int)

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
            if k != 'symp':
                user_feats[k] = torch.LongTensor(v)
            else:
                user_feats[k] = torch.FloatTensor(v)
        processed_batch.append(user_feats)
        labels.append(label)
    labels = torch.FloatTensor(np.array(labels))
    label_masks = torch.not_equal(labels, -1)
    return processed_batch, labels, label_masks


class HierDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len, disease='None', use_symp=False, bal_sample=False):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.disease = disease
        self.control_ratio = 0.95
        self.bal_sample = bal_sample
        self.use_symp = use_symp
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "train", self.disease, self.use_symp)
            self.val_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "val", self.disease, self.use_symp)
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.disease, self.use_symp)
        elif stage == "test":
            self.test_set = HierDataset(self.input_dir, self.tokenizer, self.max_len, "test", self.disease, self.use_symp)

    def train_dataloader(self):
        if self.bal_sample:
            sampler = BalanceSampler(self.train_set, self.control_ratio)
            return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_hier, sampler=sampler, pin_memory=False, num_workers=4)
        else:
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

    data_module = HierDataModule(4, "./processed/symptom_top16/", tokenizer, max_len=128)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    print(next(iter(train_loader)))
    # print(batch0)
    # print(labels0)s
    # import pdb; pdb.set_trace()