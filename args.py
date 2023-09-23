import argparse



def get_parser():
    parser=argparse.ArgumentParser(
        description="Referring Segmentation codebase"
    )
    # Dataset
    parser.add_argument('--dataset', default='refcoco', 
                    help='choose one of the following datasets: refcoco, refcoco+, refcocog, refcocog_umd')
    # parser.add_argument('--dataset_root',default='./data',help='split to run test')
    parser.add_argument("--max_query_len",default=20,type=int)
    parser.add_argument('--negative_samples', default=0, type=int)
    parser.add_argument('--positive_samples', default=1, type=int)
    # BERT 
    parser.add_argument('--bert_tokenizer',  default='clip')
    # REFER
    parser.add_argument('--refer_data_root', default='./data/', help='REFER dataset root directory')
    parser.add_argument('--splitBy', default='unc', help='split By')
    parser.add_argument('--spilt',default='val',help='split to run test')

    parser.add_argument("--pretrained_checkpoint",default=None,type=str,help="name of checkpoint ")
    
    # optimizer set
    parser.add_argument("--lr",default=0.00005,type=float,help="initial learning rate")  # 1.5e-5  
    parser.add_argument("--weight-decay", "--weight_decay",default=0.01,type=float,help="weight-decay")
    parser.add_argument("--lr_multi",default=0.1,type=float)  
    parser.add_argument("--end_lr",default=1e-5,type=float,help="end_learning_rate") 
    parser.add_argument("--power",default=1.0,type=float,help="power of polynomial learning rate")
    parser.add_argument("--max_decay_steps",default=40,type=int,help="max_decay_steps for polynomial learning ")
    
    # training set
    parser.add_argument("--batch_size",default=1,type=int,help="batch size per GPU")
    parser.add_argument("--epoch",default=30,type=int,help="training epoch")  # default 45 
    parser.add_argument("--print-freq",default=100,type=int,help="the frequent of print")
    parser.add_argument("--size",default=384,type=int,help="the size of image")
    parser.add_argument("--resume",action="store_true",help="start from a check point")
    parser.add_argument("--start_epoch",default=0,type=int,help="start epoch")
    parser.add_argument("--gpu",default='0',type=str,help="")
    parser.add_argument("--pseudo_path",default=None,type=str)
    
    # Only evaluate 
    parser.add_argument("--pretrain",default=None,type=str,help="name of checkpoint ")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--test_split",default='val',type=str,help="['val', 'testA', 'testB', 'test]")
    parser.add_argument("--prms", action="store_true", default=False)
    
    # we provide two evaluate mode to better use all sentence to make predict
    parser.add_argument("--eval_mode",default='cat',type=str,help="['cat' or 'random']") 
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--dcrf", action="store_true", default=False)
    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--consistency_type", default='mse',type=str)
    parser.add_argument("--scales", default=None,type=str)

    # Save check point 
    parser.add_argument("--output",default=None,type=str,help="output dir for checkpoint")
    parser.add_argument("--board_folder",default=None,type=str,help="tensorboard")
    parser.add_argument("--vis_out", default=None, type=str)
    parser.add_argument("--eval_vis_out", default=None, type=str)

    parser.add_argument("--pooling", default="gmp_gap", type=str)

    # Distributed training parameters
    parser.add_argument("--distributed",action="store_true", default=False, help="start from a check point")
    parser.add_argument("--world-size", default=2, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--local_rank",default=0,help="local rank") 

    # * Loss coefficients
    parser.add_argument('--attn_multi', default=0.1, type=float)
    parser.add_argument('--attn_multi_vis', default=0.1, type=float)
    parser.add_argument('--attn_multi_text', default=0.1, type=float)
    parser.add_argument('--w1', default=2, type=float)
    parser.add_argument('--w2', default=1, type=float)
    parser.add_argument('--w3', default=5, type=float)

    parser.add_argument('--FOCAL_P', default=3, type=float)
    parser.add_argument('--FOCAL_LAMBDA', default=0.01, type=float)

    parser.add_argument('--wr', default=5e-4, type=float)

    parser.add_argument("--backbone", default="clip-RN50", type=str)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    parser.add_argument("--cam_save_dir", default=None, type=str)
    parser.add_argument("--name_save_dir", default=None, type=str)
    parser.add_argument("--save_cam",action="store_true", default=False)


    parser.add_argument("--mode", default='clip', type=str)
    
    # demo 
    parser.add_argument("--img",default=None,type=str)
    parser.add_argument("--text",default=None,type=str)

    return parser
    