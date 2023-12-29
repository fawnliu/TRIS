now=$(date +"%Y%m%d_%H%M%S")


# CUDA_VISIBLE_DEVICES=0  python validate.py \
#     --batch_size 1 \
#     --size 320 \
#     --dataset refcoco \
#     --splitBy unc \
#     --test_split val \
#     --max_query_len 20 \
#     --output weights \
#     --resume --pretrain  stage1_refcoco.pth \
#     --eval \
#     2>&1 | tee logs/eval_${now}_stage1_refcoco.txt


# CUDA_VISIBLE_DEVICES=0  python validate.py \
#     --batch_size 1 \
#     --size 320 \
#     --dataset refcoco+ \
#     --splitBy unc \
#     --test_split val \
#     --max_query_len 20 \
#     --output weights \
#     --resume --pretrain  stage1_refcoco+.pth \
#     --eval \
#     2>&1 | tee logs/eval_${now}_stage1_refcocop.txt


CUDA_VISIBLE_DEVICES=0  python validate.py \
    --batch_size 1 \
    --size 320 \
    --dataset refcocog \
    --splitBy umd \
    --test_split val \
    --max_query_len 20 \
    --output weights \
    --resume --pretrain  stage1_refcocog_umd.pth \
    --eval \
    2>&1 | tee logs/eval_${now}_stage1_refcocog_umd.txt



# CUDA_VISIBLE_DEVICES=0  python validate.py \
#     --batch_size 1 \
#     --size 320 \
#     --dataset refcocog \
#     --splitBy google \
#     --test_split val \
#     --max_query_len 20 \
#     --output weights \
#     --resume --pretrain  stage1_refcocog_google.pth \
#     --eval \
#     2>&1 | tee logs/eval_${now}_stage1_refcocog_google.txt



# CUDA_VISIBLE_DEVICES=0 python validate_referit.py \
#     --batch_size 1 \
#     --size 320 \
#     --dataset referit \
#     --test_split test \
#     --max_query_len 20 \
#     --refer_data_root ./data/referit/ \
#     --output weights \
#     --resume --pretrain stage1_referit.pth \
#     --eval \
#     2>&1 | tee logs/eval_${now}_stage1_referit.txt

# dir=./output
# python validate.py \
#     --batch_size 1 \
#     --size 320 \
#     --dataset refcocog \
#     --splitBy umd \
#     --test_split val \
#     --max_query_len 20 \
#     --output weights/ \
#     --resume --pretrain  stage1_refcocog_umd.pth \
#     --cam_save_dir $dir/refcocog_umd/cam/ \
#     --name_save_dir $dir/refcocog_umd  \
#     --eval 
#     # --prms --save_cam 

