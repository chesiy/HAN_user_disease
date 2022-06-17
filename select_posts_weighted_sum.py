import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import numpy as np
from tqdm import tqdm
import json

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

kg_info_dir = "../data/parsed_kg_info_after_anno.json"
with open(kg_info_dir, 'r', encoding='utf-8') as f:
    kg_content = json.load(f)
kg_id2symp = kg_content['id2symp']
symp2kg_id = kg_content['symp2id']
kg_symp_id2disease_id = kg_content['symp_id2disease_ids']

# Caution! order of symptoms in model inference is different from KG
model_id2symp = ['Anxious_Mood',
 'Autonomic_symptoms',
 'Cardiovascular_symptoms',
 'Catatonic_behavior',
 'Decreased_energy_tiredness_fatigue',
 'Depressed_Mood',
 'Gastrointestinal_symptoms',
 'Genitourinary_symptoms',
 'Hyperactivity_agitation',
 'Impulsivity',
 'Inattention',
 'Indecisiveness',
 'Respiratory_symptoms',
 'Suicidal_ideas',
 'Worthlessness_and_guilty',
 'avoidance_of_stimuli',
 'compensatory_behaviors_to_prevent_weight_gain',
 'compulsions',
 'diminished_emotional_expression',
 'do_things_easily_get_painful_consequences',
 'drastical_shift_in_mood_and_energy',
 'fear_about_social_situations',
 'fear_of_gaining_weight',
 'fears_of_being_negatively_evaluated',
 'flight_of_ideas',
 'intrusion_symptoms',
 'loss_of_interest_or_motivation',
 'more_talktive',
 'obsession',
 'panic_fear',
 'pessimism',
 'poor_memory',
 'sleep_disturbance',
 'somatic_muscle',
 'somatic_symptoms_others',
 'somatic_symptoms_sensory',
 'weight_and_appetite_change',
 'Anger_Irritability']
model_symp_id2disease_id = [kg_symp_id2disease_id[symp2kg_id[symp]] for symp in model_id2symp]

# Inverse Disease Frequency (IDF)
# The fewer diseases a symptom related to, the higher weight
# lowest: 1 (all 7 diseases), highest: 3 (only 1 disease)
model_symp_id2disease_cnt = [len(diseases) for diseases in model_symp_id2disease_id]
idf = np.log2(1 + len(id2disease) / np.array([cnt for cnt in model_symp_id2disease_cnt])).reshape(-1, 1)

# Inverse Symptom Frequency (ISF)
# The less likely a symptom occurs, the higher weight
with open(f"../smhd_55/symp_dataset/train_rm_feats.pkl", "rb") as f:
    train_symp_data = pickle.load(f)
avg_symp_probs = np.mean(np.stack([symp_probs.mean(0) for symp_probs in train_symp_data], 0), 0)
isf = np.log2(1 + 1 / avg_symp_probs).reshape(-1, 1)

topK = 16
pool_types = ["sum", "sum_isf", "sum_idf", "sum_isf_idf"]
for pool_type in pool_types:
    group = f"symptom_{pool_type}"
    os.makedirs(f"processed/{group}_top{topK}", exist_ok=True)

for split in ["train", "val", "test"]:
    with open(f"../smhd_55/{split}_data.pkl", "rb") as f:
        data = pickle.load(f)
    if split == "train":
        symp_data = train_symp_data
    else:
        with open(f"../smhd_55/symp_dataset/{split}_rm_feats.pkl", "rb") as f:
            symp_data = pickle.load(f)

    selected = {pool_type: [] for pool_type in pool_types}
    print(len(data))
    for i, record in enumerate(tqdm(data)):
        if len(record['diseases']):
            uid = 'P' + str(record['id'])
        else:
            uid = 'C' + str(record['id'])
        symp_probs = symp_data[uid]
        user_posts = record['posts']
        for pool_type in pool_types:
            if pool_type == "sum":
                post_scores = symp_probs.sum(1)
            elif pool_type == "sum_isf":
                post_scores = (symp_probs * isf).squeeze()
            elif pool_type == "sum_idf":
                post_scores = (symp_probs * idf).squeeze()
            elif pool_type == "sum_isf_idf":
                post_scores = (symp_probs * (idf*isf)).squeeze()
            top_ids = post_scores.argsort()[-topK:]
            top_ids = np.sort(top_ids)  # sort in time order
            sel_posts = [user_posts[ii] for ii in top_ids]
            sel_probs = symp_probs[top_ids]
            selected[pool_type].append({'id': record['id'], 'diseases': record['diseases'], 'selected_posts': sel_posts, 'symp_probs': sel_probs})
        # break
    # break
    for pool_type in pool_types:
        group = f"symptom_{pool_type}"
        with open(f"processed/{group}_top{topK}/{split}.pkl", "wb") as f:
            pickle.dump(selected[pool_type], f)
