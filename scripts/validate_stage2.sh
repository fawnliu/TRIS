now=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=0  python validate.py \
    --batch_size 1 \
    --size 320 \
    --dataset refcoco \
    --splitBy unc \
    --test_split val \
    --max_query_len 20 \
    --output weights/stage2 \
    --resume --pretrain  stage2_refcoco.pth \
    --eval \
    2>&1 | tee logs/eval_${now}_stage2_pseudo_refcoco.txt


CUDA_VISIBLE_DEVICES=0  python validate.py \
    --batch_size 1 \
    --size 320 \
    --dataset refcoco+ \
    --splitBy unc \
    --test_split val \
    --max_query_len 20 \
    --output weights/stage2 \
    --resume --pretrain  stage2_refcoco+.pth \
    --eval \
    2>&1 | tee logs/eval_${now}_stage2_pseudo_refcocop.txt


CUDA_VISIBLE_DEVICES=0  python validate.py \
    --batch_size 1 \
    --size 320 \
    --dataset refcocog \
    --splitBy umd \
    --test_split val \
    --max_query_len 20 \
    --output weights/stage2 \
    --resume --pretrain  stage2_refcocog_umd.pth \
    --eval \
    2>&1 | tee logs/eval_${now}_stage2_pseudo_refcocog_umd.txt


CUDA_VISIBLE_DEVICES=0  python validate.py \
    --batch_size 1 \
    --size 320 \
    --dataset refcocog \
    --splitBy google \
    --test_split val \
    --max_query_len 20 \
    --output weights/stage2/ \
    --resume --pretrain  stage2_refcocog_google.pth \
    --eval \
    2>&1 | tee logs/eval_${now}_stage2_pseudo_refcocog_google.txt


CUDA_VISIBLE_DEVICES=0 python validate_referit.py \
    --batch_size 1 \
    --size 320 \
    --dataset referit \
    --test_split val \
    --backbone clip-RN50 \
    --max_query_len 20 \
    --refer_data_root ./data/referit/ \
    --output weights/stage2/ \
    --resume --pretrain stage2_referit.pth \
    --eval \
    2>&1 | tee logs/eval_${now}_referit_clip_RN50.txt 


