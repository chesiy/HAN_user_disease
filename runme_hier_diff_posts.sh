export CUDA_VISIBLE_DEVICES=0
# for sel_disease in adhd anxiety
# for sel_disease in ptsd
for sel_disease in adhd anxiety bipolar depression eating ocd ptsd
do
    nohup python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_multi_disease_top16/"${sel_disease} --bs 32 --user_encoder=pairwise --num_trans_layers=4 --disease=${sel_disease} > ./log/tiny_diff_posts_${sel_disease}_pairwise_bsamp_0.95.log 2>&1 &
    # python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_multi_disease_top16/"${sel_disease} --bs 32 --user_encoder=pairwise --num_trans_layers=4 --disease=${sel_disease}
    # nohup python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_multi_disease_top16_0.9/"${sel_disease} --bs 32 --user_encoder=trans_abs --num_trans_layers=4 --disease=${sel_disease} > ./log/MMR_log/tiny_MMR_diff_${sel_disease}_bsamp_0.95_0.9.log 2>&1 &
done 

# for sel_disease in adhd
# do
#     python -u main_hier_clf.py --lr=1e-5 --input_dir "./processed/symptom_multi_disease_top16/"${sel_disease} --bs 32 --user_encoder=pairwise --num_trans_layers=4 --disease=${sel_disease}
# done 
