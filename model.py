import enum
from cv2 import log
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import functional as F

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

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
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            ret[k] = []
        for labels, probs in zip(labels_by_class, probs_by_class):
            preds = (probs > threshold).astype(float)
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

    return ret

def masked_logits_loss(logits, labels, masks=None):
    # treat unlabeled samples(-1) as implict negative (0.)
    labels2 = torch.clamp_min(labels, 0.)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none')
    # print(losses, masks)
    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        return masked_losses.mean()
    else:
        return losses.mean()

class LightningInterface(pl.LightningModule):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        # print(kwargs)
        self.disease = kwargs['disease']
        # self.disease = 'None'
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = masked_logits_loss

    # def criterion(self, logits, labels):
    #     return logits_loss(logits, labels, pos_weight=None)

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat

        loss = self.criterion(y_hat, y, label_masks)
        tensorboard_logs = {'train_loss': loss}
        # import pdb; pdb.set_trace()
        return {'loss': loss, 'log': tensorboard_logs}

    # def training_epoch_end(self, output) -> None:
    #     self.log('lr', self.hparams.lr)
    
    def validation_step(self, batch, batch_nb):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'val_loss': self.criterion(y_hat, y, label_masks), "labels": yy, "probs": yy_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        # print(all_probs)
        ret = get_avg_metrics(all_labels, all_probs, self.threshold, self.disease)
        print('val res', ret)
        if self.current_epoch == 0:  # prevent the initial check modifying it
            self.best_f1 = 0
        self.best_f1 = max(self.best_f1, ret['macro_f1'])
        tensorboard_logs = {'val_loss': avg_loss, 'hp_metric': self.best_f1, 'val_f1': ret['macro_f1']}
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'test_loss': self.criterion(y_hat, y, label_masks), "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        ret = get_avg_metrics(all_labels, all_probs, self.threshold, self.disease)
        results = {'test_loss': avg_loss}
        for k, v in ret.items():
            results[f"test_{k}"] = v
        self.log_dict(results)
        return results

    def on_after_backward(self):
        pass
        # can check gradient
        # global_step = self.global_step
        # if int(global_step) % 100 == 0:
        #     for name, param in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class Classifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="prajjwal1/bert-tiny", **kwargs):
        super().__init__(threshold=threshold, **kwargs)

        self.model_type = model_type
        self.model = BERTFlatClassifier(model_type)
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(**x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        self.model_type = model_type
        # binary classification
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, 7)
    
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # import pdb; pdb.set_trace()
        x = outputs.last_hidden_state[:, 0, :]
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits

class BERTHierClassifierSimple(nn.Module):
    def __init__(self, model_type) -> None:
        super().__init__()
        self.model_type = model_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        self.attn_ff = nn.Linear(self.post_encoder.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.post_encoder.config.hidden_size, 1)
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, hidden_size]
            x = post_outputs.last_hidden_state[:, 0, :]
            # [num_posts, ]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(), -1)
            # weighted sum [hidden_size, ]
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        x = self.dropout(feats)

        logits = self.clf(x).squeeze()
        # [bs, num_posts]
        return logits, attn_scores

class BERTHierClassifierTrans(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        # batch_first = False
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            # [num_posts, ]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(), -1)
            # weighted sum [hidden_size, ]
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        x = self.dropout(feats)

        logits = self.clf(x).squeeze()

        # [bs, num_posts]
        return logits, attn_scores

class BERTHierClassifierTransAbs(nn.Module):
    '''with absolute learned positional embedding for post level'''
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim, 7)
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(), -1)
            # print(attn_score)
            # weighted sum [hidden_size, ]
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        x = self.dropout(feats)
        logits = self.clf(x)
        # [bs, num_posts]
        return logits, attn_scores

class BERTHierClassifierTransAbsMultiAtt(nn.Module):
    '''with absolute learned positional embedding for post level'''
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        # self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = [torch.softmax(attn_ff(x).squeeze(), -1) for attn_ff in self.attn_ff]
            # weighted sum [hidden_size, ]
            feat = [self.dropout(score @ x) for score in attn_score]
            feats.append(feat)
            attn_scores.append(attn_score)

        logits = []
        for i in range(len(id2disease)):
            tmp = [feats[j][i] for j in range(len(feats))]
            logit = self.clf[i](torch.stack(tmp))
            logits.append(logit)
        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        return logits, attn_scores

class SympGuidedMultiAtt(nn.Module):
    '''with absolute learned positional embedding for post level'''
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        # self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.num_symps = 38
        self.num_diseases = 7
        self.symp_attn_bias = nn.Parameter(torch.rand(1, self.num_symps))
        # [num_disease, num_symps]
        disease_symp_mask = np.load("disease_symp_mask.npy")
        self.disease_symp_mask = nn.Parameter(torch.FloatTensor(disease_symp_mask), requires_grad=False)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            # Symptom guided attention
            # Post with Disease-specific symptom will receive higher attention weight from the disease attention head
            attn_score = []
            for disease_id, attn_ff in enumerate(self.attn_ff):
                symp_mask = self.disease_symp_mask[disease_id]
                # achieved with the bias term
                symp_bias = user_feats["symp"] * (symp_bias * symp_mask)
                attn_score.append(torch.softmax(attn_ff(x + symp_bias).squeeze(), -1))
            # weighted sum [hidden_size, ]
            feat = [self.dropout(score @ x) for score in attn_score]
            feats.append(feat)
            attn_scores.append(attn_score)

        logits = []
        for i in range(len(id2disease)):
            tmp = [feats[j][i] for j in range(len(feats))]
            logit = self.clf[i](torch.stack(tmp))
            logits.append(logit)
        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        return logits, attn_scores

def kmax_pooling(x, k):
    return x.sort(dim = 2)[0][:, :, -k:]

class KMaxMeanCNN(nn.Module):
    def __init__(self, in_dim, filter_num=50, filter_sizes=(2,3,4,5,6), dropout=0.2, max_pooling_k=5):
        super(KMaxMeanCNN, self).__init__()
        self.in_dim = in_dim
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.max_pooling_k = max_pooling_k
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_dim, filter_num, size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seqs, seq_masks=None):
        # import ipdb; ipdb.set_trace()
        input_seqs = input_seqs.transpose(1, 2) # [bs, L, in_dim] -> [bs, in_dim, L]
        x = [F.relu(conv(input_seqs)) for conv in self.convs]
        x = [kmax_pooling(item, self.max_pooling_k).mean(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)

        return x

class PairWiseClassifier(nn.Module):
    def __init__(self, model_type, in_dim, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.CNN_encoder = KMaxMeanCNN(in_dim)
        self.clf = nn.Linear(self.hidden_dim + 250, 7)
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        symp_probs = []
        for user_feats in batch:
            symp_probs.append(user_feats['symp'])
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(), -1)
            feat = attn_score @ x
            feats.append(feat)
            attn_scores.append(attn_score)
        feats = torch.stack(feats)
        symp_probs = torch.stack(symp_probs)
        # print(symp_probs.shape)
        symp_emb = self.CNN_encoder(symp_probs)
        # print(symp_emb.shape)
        user_emb = torch.concat([feats, symp_emb], dim=1)
        user_emb = self.dropout(user_emb)
        logits = self.clf(user_emb)
        
        return logits, attn_scores


class PairWiseMultiAttn(nn.Module):
    def __init__(self, model_type, in_dim, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        # batch_first = False
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.CNN_encoder = KMaxMeanCNN(in_dim)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim + 250, 1) for disease in id2disease])
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        symp_probs = []
        for user_feats in batch:
            symp_probs.append(user_feats['symp'])
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = [torch.softmax(attn_ff(x).squeeze(), -1) for attn_ff in self.attn_ff]
            # weighted sum [hidden_size, ]
            feat = [self.dropout(score @ x) for score in attn_score]
            feats.append(feat)
            attn_scores.append(attn_score)
        symp_probs = torch.stack(symp_probs)
        # print(symp_probs.shape)
        symp_emb = self.CNN_encoder(symp_probs)
        # print(symp_emb.shape)
        logits = []
        for i in range(len(id2disease)):
            tmp = [feats[j][i] for j in range(len(feats))]
            # print(torch.stack(tmp).shape)
            logit = self.clf[i](torch.concat([torch.stack(tmp), symp_emb], dim=1))
            logits.append(logit)
        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        
        return logits, attn_scores


class HierClassifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="prajjwal1/bert-tiny", user_encoder="none", num_heads=8, num_trans_layers=2, freeze_word_level=False, pool_type="first", **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.model_type = model_type
        if user_encoder == "trans":
            self.model = BERTHierClassifierTrans(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "trans_abs":
            self.model = BERTHierClassifierTransAbs(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "trans_abs_multi_att":
            self.model = BERTHierClassifierTransAbsMultiAtt(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "symp_guide_multi_att":
            self.model = SympGuidedMultiAtt(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "pairwise":
            in_dim = 38
            self.model = PairWiseClassifier(model_type, in_dim, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "pairwise_multiattn":
            in_dim = 38
            self.model = PairWiseMultiAttn(model_type, in_dim, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        else:
            self.model = BERTHierClassifierSimple(model_type)
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        # parser.add_argument("--trans", action="store_true")
        parser.add_argument("--user_encoder", type=str, default="none")
        parser.add_argument("--pool_type", type=str, default="first")
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--num_trans_layers", type=int, default=2)
        parser.add_argument("--freeze_word_level", action="store_true")
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer