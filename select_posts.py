import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import string
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import re
import json
import blingfire

sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sbert.cuda(0)
kg_info_dir = "../data/parsed_kg_info_after_anno.json"
with open(kg_info_dir, 'r', encoding='utf-8') as f:
    kg_info = json.load(f)
id2desc = kg_info['id2desc']
description_embs = sbert.encode(id2desc)

cut_sentences = lambda x: blingfire.text_to_sentences(x.strip()).split("\n")

with open("../smhd_55/test_data.pkl", "rb") as f:
    data = pickle.load(f)
    
topK = 32
for group in ["description"]:
    os.makedirs(f"processed/{group}_sim{topK}_sentence", exist_ok=True)

selected_train = []
selected_val = []
selected_test = []
print(len(data))
for i, record in enumerate(tqdm(data)):
    user_posts = record['posts']
    sentences = []
    for post in user_posts:
        for sent in cut_sentences(post):
            sentences.append(sent)
    # print(len(sentences))
    user_embs = sbert.encode(sentences)
    # user_embs = sbert.encode(user_posts)
    pair_sim = cosine_similarity(user_embs, description_embs)
    # print(pair_sim.shape)
    sim_scores = pair_sim.max(1)
    # print(sim_scores.shape, np.sort(sim_scores)[-topK:])
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = [sentences[ii] for ii in top_ids]
    # sel_posts = [user_posts[ii] for ii in top_ids]
    selected_train.append({'id': record['id'], 'diseases': record['diseases'], 'selected_posts': sel_posts})

with open(f"processed/description_sim{topK}_sentence/test.pkl", "wb") as f:
    pickle.dump(selected_train, f)


# with open(f"processed/description_sim16/train.pkl", "rb") as f:
#     data = pickle.load(f)
#     for record in data:
#         print(record)