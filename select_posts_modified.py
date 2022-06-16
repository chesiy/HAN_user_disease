import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import string
from tqdm import tqdm
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
import blingfire

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

'''
# all + status

topK = 16
for group in ["symptom_status"]:
    os.makedirs(f"processed/{group}_top{topK}", exist_ok=True)

for split in ["train", "val", "test"]:
    with open(f"../smhd_55/{split}_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(f"../smhd_55/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
        symp_data = pickle.load(f)
    with open(f"../smhd_55/symp_dataset/{split}_status_feats.pkl", "rb") as f:
        other_feats = pickle.load(f)
    selected = []
    print(len(data))
    for i, record in enumerate(tqdm(data)):
        if len(record['diseases']):
            uid = 'P' + str(record['id'])
        else:
            uid = 'C' + str(record['id'])
        symp_probs = symp_data[uid]
        status_feats = other_feats[uid]
        symp_probs = symp_probs * (1 - status_feats['uncertain']).reshape(-1, 1)
        # print(symp_probs)
        user_posts = record['posts']
        post_scores = symp_probs.max(1)
        # print(post_scores.shape)
        top_ids = post_scores.argsort()[-topK:]
        top_ids = np.sort(top_ids)  # sort in time order
        sel_posts = [user_posts[ii] for ii in top_ids]
        sel_probs = symp_probs[top_ids]
        # print(sel_probs.shape)
        selected.append({'id': record['id'], 'diseases': record['diseases'], 
                        'selected_posts': sel_posts, 'symp_probs': sel_probs})
        # break
    # break
    with open(f"processed/symptom_status_top{topK}/{split}.pkl", "wb") as f:
        pickle.dump(selected, f)
'''
##################################################################################################

# all + MMR
topK = 16
alpha = 0.9
for group in ["symptom_MMR"]:
    os.makedirs(f"processed/{group}_top{topK}", exist_ok=True)

for split in ["train", "val", "test"]:
    with open(f"../smhd_55/{split}_data.pkl", "rb") as f:
        data = pickle.load(f)
    with open(f"../smhd_55/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
        symp_data = pickle.load(f)

    selected = []
    print(len(data))
    for i, record in enumerate(tqdm(data)):
        if len(record['diseases']):
            uid = 'P' + str(record['id'])
        else:
            uid = 'C' + str(record['id'])
        symp_probs = symp_data[uid]
        # print(symp_probs)
        user_posts = record['posts']
        sel_ids = []
        similarity_matrix = cosine_similarity(symp_probs, symp_probs)
        symp_score = symp_probs.max(1)
        for round_idx in range(topK):
            mmr_score = symp_score
            if round_idx != 0:
                mmr_score = alpha * symp_score - (1-alpha) * similarity_matrix[:, sel_ids].max(1)
            top_ids = mmr_score.argsort()
            for top_id in top_ids[::-1]:
                if top_id not in sel_ids:
                    sel_ids.append(top_id)
                    break

        sel_ids = np.sort(sel_ids)
        sel_posts = [user_posts[ii] for ii in sel_ids]
        sel_probs = symp_probs[sel_ids]
        # print(sel_probs.shape)
        selected.append({'id': record['id'], 'diseases': record['diseases'], 
                        'selected_posts': sel_posts, 'symp_probs': sel_probs})
        # break
    # break
    with open(f"processed/symptom_MMR_top{topK}/{split}.pkl", "wb") as f:
        pickle.dump(selected, f)


##################################################################################################
'''
# by disease + status
with open("../data/parsed_kg_info_after_anno.json", "r") as f:
    kg_content = json.load(f)

id2disease = kg_content['id2disease']
id2symp = kg_content['id2symp']
symp_id2disease_id = kg_content['symp_id2disease_ids']
disease_id2symp_id = []
for i in range(len(id2disease)):
    disease_id2symp_id.append([])
# print(disease_id2symp_id)
for symp_id, diseases in enumerate(symp_id2disease_id):
    # print(symp_id, diseases)
    for dis_id in diseases:
        disease_id2symp_id[dis_id].append(symp_id)

topK = 16
for group in ["symptom", "symptom_status"]:
    os.makedirs(f"processed/{group}_multi_disease_top{topK}", exist_ok=True)
    for disease in id2disease:
        os.makedirs(f"processed/{group}_multi_disease_top{topK}/{disease}", exist_ok=True)

    for split in ["train", "val", "test"]:
        with open(f"../smhd_55/{split}_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(f"../smhd_55/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
            symp_data = pickle.load(f)
        if group == "symptom_status":
            with open(f"../smhd_55/symp_dataset/{split}_status_feats.pkl", "rb") as f:
                other_feats = pickle.load(f)
    
        selected = {}
        for disease in id2disease:
            selected[disease] = []
        # print(len(data))
        for i, record in enumerate(tqdm(data)):
            if len(record['diseases']):
                uid = 'P' + str(record['id'])
            else:
                uid = 'C' + str(record['id'])
            symp_probs = symp_data[uid]
            if group == "symptom_status":
                status_feats = other_feats[uid]
                symp_probs = symp_probs * (1 - status_feats['uncertain']).reshape(-1, 1)
            user_posts = record['posts']
            # print(uid, record['diseases'], len(user_posts))
            for disease_id in range(len(id2disease)):
                disease = id2disease[disease_id]
                post_scores = symp_probs[:, disease_id2symp_id[disease_id]].max(1)
                top_ids = post_scores.argsort()[-topK:]
                top_ids = np.sort(top_ids)  # sort in time order
                sel_posts = [user_posts[ii] for ii in top_ids]
                sel_probs = symp_probs[top_ids]
                # print(disease, sel_posts)
                selected[disease].append({'id': record['id'], 'diseases': record['diseases'], 
                                'selected_posts': sel_posts, 'symp_probs': sel_probs})
                
        for disease, posts in selected.items():
            with open(f"processed/{group}_multi_disease_top{topK}/{disease}/{split}.pkl", "wb") as f:
                pickle.dump(posts, f)
'''
##################################################################################################

###################################################################################################
'''
# symptom + MMR
with open("../data/parsed_kg_info_after_anno.json", "r") as f:
    kg_content = json.load(f)

# id2disease = kg_content['id2disease']
id2symp = kg_content['id2symp']
symp_id2disease_id = kg_content['symp_id2disease_ids']
disease_id2symp_id = []
for i in range(len(id2disease)):
    disease_id2symp_id.append([])

for symp_id, diseases in enumerate(symp_id2disease_id):
    # print(symp_id, diseases)
    for dis_id in diseases:
        disease_id2symp_id[dis_id].append(symp_id)

topK = 16
alpha = 0.9
for group in ["symptom_MMR"]:
    os.makedirs(f"processed/{group}_multi_disease_top{topK}_{alpha}", exist_ok=True)
    for disease in id2disease:
        os.makedirs(f"processed/{group}_multi_disease_top{topK}_{alpha}/{disease}", exist_ok=True)

    for split in ["test", "train", "val"]:
        with open(f"../smhd_55/{split}_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(f"../smhd_55/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
            symp_data = pickle.load(f)
    
        selected = {}
        for disease in id2disease:
            selected[disease] = []
        # print(len(data))
        for i, record in enumerate(tqdm(data)):
            if len(record['diseases']):
                uid = 'P' + str(record['id'])
            else:
                uid = 'C' + str(record['id'])
            symp_probs = symp_data[uid]
            user_posts = record['posts']
            print(record['diseases'])
            for disease_id in range(len(id2disease)):
                sel_ids = []
                disease = id2disease[disease_id]
                post_symp_probs = symp_probs[:, disease_id2symp_id[disease_id]]
                similarity_matrix = cosine_similarity(post_symp_probs, post_symp_probs)
                symp_score = post_symp_probs.max(1)
                for round_idx in range(topK):
                    mmr_score = symp_score
                    if round_idx != 0:
                        mmr_score = alpha * symp_score - (1-alpha) * similarity_matrix[:, sel_ids].max(1)
                    top_ids = mmr_score.argsort()
                    for top_id in top_ids[::-1]:
                        if top_id not in sel_ids:
                            sel_ids.append(top_id)
                            break
                sel_ids = np.sort(sel_ids)
                sel_posts = [user_posts[ii] for ii in sel_ids]
                sel_probs = symp_probs[sel_ids]
                # print(disease, sel_ids)
                # print(sel_posts)
                # print(sel_probs)
                selected[disease].append({'id': record['id'], 'diseases': record['diseases'], 
                                'selected_posts': sel_posts, 'symp_probs': sel_probs})
                # break
            # break  
        # break              
        for disease, posts in selected.items():
            with open(f"processed/{group}_multi_disease_top{topK}_{alpha}/{disease}/{split}.pkl", "wb") as f:
                pickle.dump(posts, f)
'''


'''
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sbert.cuda(0)
kg_info_dir = "../data/parsed_kg_info_after_anno.json"
with open(kg_info_dir, 'r', encoding='utf-8') as f:
    kg_content = json.load(f)
id2desc = kg_content['id2desc']
description_embs = sbert.encode(id2desc)
id2symp = kg_content['id2symp']
symp_id2disease_id = kg_content['symp_id2disease_ids']
symp_id2desc_range = kg_content['symp_id2desc_range']
disease_id2symp_id = []
disease_id2desc_id = []
for i in range(len(id2disease)):
    disease_id2symp_id.append([])
    disease_id2desc_id.append([])

for symp_id, diseases in enumerate(symp_id2disease_id):
    # print(symp_id, diseases)
    for dis_id in diseases:
        disease_id2symp_id[dis_id].append(symp_id)
        for desc_id in range(symp_id2desc_range[symp_id][0], symp_id2desc_range[symp_id][1]):
            disease_id2desc_id[dis_id].append(desc_id)
# print(disease_id2desc_id)
cut_sentences = lambda x: blingfire.text_to_sentences(x.strip()).split("\n")

topK = 16
for group in ["desc_sim"]:
    os.makedirs(f"processed/{group}_multi_disease_top{topK}", exist_ok=True)
    for disease in id2disease:
        os.makedirs(f"processed/{group}_multi_disease_top{topK}/{disease}", exist_ok=True)

    for split in ["train", "val", "test"]:
        with open(f"../smhd_55/{split}_data.pkl", "rb") as f:
            data = pickle.load(f)
        with open(f"../smhd_55/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
            symp_data = pickle.load(f)
    
        selected = {}
        for disease in id2disease:
            selected[disease] = []
        # print(len(data))
        for i, record in enumerate(tqdm(data)):
            if len(record['diseases']):
                uid = 'P' + str(record['id'])
            else:
                uid = 'C' + str(record['id'])
            symp_probs = symp_data[uid]
            user_posts = record['posts']
            user_embs = sbert.encode(user_posts)
            pair_sim = cosine_similarity(user_embs, description_embs)
            # print(uid, record['diseases'], len(user_posts))
            for disease_id in range(len(id2disease)):
                disease = id2disease[disease_id]
                sim_scores = pair_sim[:, disease_id2desc_id[disease_id]].max(1)
                top_ids = sim_scores.argsort()[-topK:]
                top_ids = np.sort(top_ids)  # sort in time order
                sel_posts = [user_posts[ii] for ii in top_ids]
                # print(disease, sel_posts)
                selected[disease].append({'id': record['id'], 'diseases': record['diseases'], 
                                'selected_posts': sel_posts})
                
        for disease, posts in selected.items():
            with open(f"processed/{group}_multi_disease_top{topK}/{disease}/{split}.pkl", "wb") as f:
                pickle.dump(posts, f)
'''
