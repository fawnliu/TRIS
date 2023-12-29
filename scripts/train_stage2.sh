now=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
    --batch_size 48 \
    --size 320 \
    --dataset refcocog \
    --splitBy umd \
    --test_split val \
    --bert_tokenizer clip \
    --backbone clip-RN50 \
    --max_query_len 20 \
    --epoch 15 \
    --output ./weights/stage2/pseudo_refcocog_umd \
    --pseudo_path output/refcocog_umd/ins_seg \
    2>&1 | tee logs/train_${now}_pseudo_refcocog_umd.txt
