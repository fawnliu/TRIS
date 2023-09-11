dir=../output

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sample_refer.py \
#     --cam_out_dir ../output/refcoco/cam \
#     --ir_label_out_dir ../output/refcoco/ir_label \
#     --ins_seg_out_dir ../output/refcoco/ins_seg \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/refcoco \
#     --train_list ../output/refcoco/refcoco_train_names.json \
#     --num_workers 2 \
#     --irn_batch_size 96 \
#     --train_irn_pass True \
#     --make_ins_seg_pass True \

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sample_refer.py \
#     --cam_out_dir ../output/refcocop/cam \
#     --ir_label_out_dir ../output/refcocop/ir_label \
#     --ins_seg_out_dir ../output/refcocop/ins_seg \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/refcocop \
#     --train_list ../output/refcocop/refcoco+_train_names.json \
#     --num_workers 2 \
#     --irn_batch_size 96 \
#     --train_irn_pass True \
#     --make_ins_seg_pass True \


# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sample_refer.py \
#     --cam_out_dir ../output/referit/cam \
#     --ir_label_out_dir ../output/referit/ir_label \
#     --ins_seg_out_dir ../output/referit/ins_seg \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/referit \
#     --voc12_root ./data/referit/images/ \
#     --train_list ../output/referit/referit_train_names.json \
#     --num_workers 2 \
#     --irn_batch_size 96 \
#     --train_irn_pass True \
#     --make_ins_seg_pass True \


CUDA_VISIBLE_DEVICES=0 python run_sample_refer.py \
    --voc12_root ../data/train2014 \
    --cam_out_dir ../output/refcocog_umd/cam \
    --ir_label_out_dir ../output/refcocog_umd/ir_label \
    --ins_seg_out_dir ../output/refcocog_umd/ins_seg \
    --cam_eval_thres 0.15 \
    --work_space output_refer/refcocog_umd \
    --train_list ../output/refcocog_umd/refcocog_train_names.json \
    --num_workers 2 \
    --irn_batch_size 24 \
    --train_irn_pass True \
    --make_ins_seg_pass True \


# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_sample_refer.py \
#     --cam_out_dir ../output/refcocog_google/cam \
#     --ir_label_out_dir ../output/refcocog_google/ir_label \
#     --ins_seg_out_dir ../output/refcocog_google/ins_seg \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/refcocog_google \
#     --train_list ../output/refcocog_google/refcocog_train_names.json \
#     --num_workers 2 \
#     --irn_batch_size 96 \
#     --train_irn_pass True \
#     --make_ins_seg_pass True \

