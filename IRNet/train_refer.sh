dir=../output

# python run_sample_refer.py \
#     --cam_out_dir ../output/refcoco/cam \
#     --ir_label_out_dir ../output/refcoco/ir_label \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/refcoco \
#     --num_workers 2 \
#     --train_list ../output/refcoco/refcoco_train_names.json \
#     --cam_to_ir_label_pass True \

# python run_sample_refer.py \
#     --cam_out_dir ../output/refcocop/cam \
#     --ir_label_out_dir ../output/refcocop/ir_label \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/refcocop \
#     --num_workers 2 \
#     --train_list ../output/refcocop/refcoco+_train_names.json \
#     --cam_to_ir_label_pass True \

# python run_sample_refer.py \
#     --cam_out_dir ../output/refcocog_google/cam \
#     --ir_label_out_dir ../output/refcocog_google/ir_label \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/refcocog_google \
#     --num_workers 2 \
#     --train_list ../output/refcocog_google/refcocog_train_names.json \
#     --cam_to_ir_label_pass True \

python run_sample_refer.py \
    --voc12_root ../data/train2014 \
    --cam_out_dir ../output/refcocog_umd/cam \
    --ir_label_out_dir ../output/refcocog_umd/ir_label \
    --ir_label_out_dir ../output/refcocog_umd/ir_label \
    --ins_seg_out_dir ../output/refcocog_umd/ins_seg \
    --cam_eval_thres 0.15 \
    --work_space output_refer/refcocog_umd \
    --num_workers 2 \
    --train_list ../output/refcocog_umd/refcocog_train_names.json \
    --cam_to_ir_label_pass True \

# python run_sample_refer.py \
#     --cam_out_dir ../output/referit/cam \
#     --ir_label_out_dir ../output/referit/ir_label \
#     --cam_eval_thres 0.15 \
#     --work_space output_refer/referit \
#     --num_workers 2 \
#     --voc12_root ./data/referit/images/ \
#     --train_list ../output/referit/referit_train_names.json \
#     --cam_to_ir_label_pass True \


