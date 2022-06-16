import os
import yaml
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from os.path import dirname
import argparse
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from data import HierDataset, my_collate_hier
from model import HierClassifier
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

with open("../data/parsed_kg_info_after_anno.json", "r") as f:
    kg_content = json.load(f)

id2symp = kg_content['id2symp']

def get_avg_metrics(all_labels, all_probs, threshold, disease='None', class_names=id2disease):
    labels_by_class = []
    probs_by_class = []
    if disease != 'None':
        dis_id = id2disease.index(disease)
        sel_indices = np.where(all_labels[:, dis_id] != -1)
        labels = all_labels[:, dis_id][sel_indices]
        probs = all_probs[:, dis_id][sel_indices]
        ret = {}
        preds = (probs > threshold).astype(float)
        all_unequal = np.argwhere(labels != preds).squeeze()
        ret["macro_acc"]=np.mean(labels == preds)
        ret["macro_p"]=precision_score(labels, preds)
        ret["macro_r"]=recall_score(labels, preds)
        ret["macro_f1"]=f1_score(labels, preds)
        try:
            ret["macro_auc"]=roc_auc_score(labels, probs)
        except:
            ret["macro_auc"]=0.5
    else:
        for i in range(all_labels.shape[1]):
            sel_indices = np.where(all_labels[:, i] != -1)
            labels_by_class.append(all_labels[:, i][sel_indices])
            probs_by_class.append(all_probs[:, i][sel_indices])
        # macro avg metrics
        ret = {}
        all_unequal = []
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            ret[k] = []
        for labels, probs in zip(labels_by_class, probs_by_class):
            preds = (probs > threshold).astype(float)
            print(preds, labels)
            unequal = np.argwhere(labels != preds).squeeze()
            all_unequal.append(unequal)
            ret["macro_acc"].append(np.mean(labels == preds))
            ret["macro_p"].append(precision_score(labels, preds))
            ret["macro_r"].append(recall_score(labels, preds))
            ret["macro_f1"].append(f1_score(labels, preds))
            try:
                ret["macro_auc"].append(roc_auc_score(labels, probs))
            except:
                ret["macro_auc"].append(0.5)
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            # list of diseases
            for class_name, v in zip(class_names, ret[k]):
                ret[class_name+"_"+k[6:]] = v
            ret[k] = np.mean(ret[k])

    return ret, all_unequal

def main(args):
    ckpt_dir = args.ckpt_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    dataset = HierDataset(args.infer_input_dir, tokenizer, args.max_len, args.infer_split, args.disease)
    dataloader = DataLoader(dataset, batch_size=args.bs, collate_fn=my_collate_hier, pin_memory=True, num_workers=4)
    clf = HierClassifier.load_from_checkpoint(ckpt_dir)
    clf.eval()
    all_probs = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Inference: "):
        x, y, masks = batch
        with torch.no_grad():
            y_hat = clf(x)
            probs = torch.sigmoid(y_hat[0])
        del x, y_hat, masks
        all_probs.extend(probs)
        all_labels.extend(y)
    all_probs = np.stack(all_probs, 0)
    all_labels = np.stack(all_labels, 0)
    ret, unequal = get_avg_metrics(all_labels, all_probs, 0.5, args.disease)
    print(ret)
    input_dir2 = os.path.join(args.infer_input_dir, args.infer_split+'.pkl')
    with open(input_dir2, 'rb') as f:
        raw_data = pickle.load(f)
    if args.disease == 'None':
        for dis_id, disease_res in enumerate(unequal):
            unequal_res = []
            for it in disease_res:
                record = raw_data[it]
                posts = []
                for i in range(len(record['selected_posts'])):
                    post = record['selected_posts'][i]
                    prob = record['symp_probs'][i]
                    symp_rank = np.argsort(prob).tolist()[-5:]
                    posts.append({'post':post, 'prob':np.sort(prob).tolist()[-5:], 'symp_rank': symp_rank})
                unequal_res.append({'id':record['id'], 'diseases':record['diseases'], 'posts':posts})
            with open(f'./error_analyze/simple/{id2disease[dis_id]}_error.json', 'w') as f:
                json.dump(unequal_res, f, indent=4)
    else:
        unequal_res = []
        for it in unequal:
            record = raw_data[it]
            posts = []
            for i in range(len(record['selected_posts'])):
                post = record['selected_posts'][i]
                prob = record['symp_probs'][i]
                symp_rank = np.argsort(prob).tolist()[-5:]
                posts.append({'post':post, 'prob':np.sort(prob).tolist()[-5:], 'symp_rank': symp_rank})
            unequal_res.append({'id':record['id'], 'diseases':record['diseases'], 'posts':posts})
        with open(f'./error_analyze/diff_posts/{args.disease}_error.json', 'w') as f:
            json.dump(unequal_res, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--infer_input_dir", type=str, default="../data/symp_data")
    parser.add_argument("--infer_split", type=str, default="test")
    parser.add_argument("--infer_output_dir", type=str, default="./infer_output")
    parser.add_argument("--model_type", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--disease", type=str, default='None')
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--user_encoder", type=str, default="none")
    parser.add_argument("--pool_type", type=str, default="first")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_trans_layers", type=int, default=2)
    args = parser.parse_args()
    main(args)