now=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
    --batch_size 48 \
    --size 320 \
    --dataset refcocog \
    --splitBy umd \
    --test_split val \
    --epoch 15 \
    --bert_tokenizer clip \
    --backbone clip-RN50 \
    --max_query_len 20 \
    --negative_samples 6 \
    --output ./weights/stage1/refcocog_umd \
    --board_folder ./output/board \
    2>&1 | tee logs/train_${now}_stage1_refcocog_umd.txt 
