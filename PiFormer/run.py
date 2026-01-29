import torch
import numpy as np
import random
from exp.exp_main import Exp_Main
import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


fix_seed = 1024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multivariate Time Series Forecasting')
    
    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str, default='PiFormer',
                        help='model name, options: [PiFormer]')
    parser.add_argument('--model_id', type=str, default="abalation")

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/remote-home/data', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='IndiaSea.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate')
    parser.add_argument('--target', type=str, default='target', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=32, help='prediction sequence length')

    # model
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=128)
    # parser.add_argument('--num_nodes', type=int, default=2751)
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--use_norm', type=int, default=0, help='whether to apply norm')
    parser.add_argument('--dropout', type=float, default=0.02, help='dropout ratio')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    
    # PiFormer3
    parser.add_argument('--patch_size', nargs='+', type=int, default=2)

    # PiFormer2
    # SPO
    # parser.add_argument('--data_path', type=str, default='SPO.csv', help='data file')
    # parser.add_argument('--num_nodes', type=int, default=2801)
    # parser.add_argument('--width', type=int, default=70)
    # parser.add_argument('--length', type=int, default=40)
    # parser.add_argument('--patch_width', type=int, default=14)
    # parser.add_argument('--patch_length', type=int, default=10)

    # NAO
    # parser.add_argument('--data_path', type=str, default='NAO.csv', help='data file')
    # parser.add_argument('--num_nodes', type=int, default=2451)
    # parser.add_argument('--width', type=int, default=70)
    # parser.add_argument('--length', type=int, default=35)
    # parser.add_argument('--patch_width', type=int, default=7)
    # parser.add_argument('--patch_length', type=int, default=7)

    # SAO
    # parser.add_argument('--data_path', type=str, default='SAO.csv', help='data file')
    # parser.add_argument('--num_nodes', type=int, default=3151)
    # parser.add_argument('--width', type=int, default=70)
    # parser.add_argument('--length', type=int, default=45)
    # parser.add_argument('--patch_width', type=int, default=10)
    # parser.add_argument('--patch_length', type=int, default=9)

    # IndiaSea
    # parser.add_argument('--data_path', type=str, default='IndiaSea.csv', help='data file')
    # parser.add_argument('--num_nodes', type=int, default=2751)
    # parser.add_argument('--width', type=int, default=55)
    # parser.add_argument('--length', type=int, default=50)
    # parser.add_argument('--patch_width', type=int, default=11)
    # parser.add_argument('--patch_length', type=int, default=10)

    # NPO
    parser.add_argument('--data_path', type=str, default='NPO.csv', help='data file')
    parser.add_argument('--num_nodes', type=int, default=2081)
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--length', type=int, default=26)
    parser.add_argument('--patch_width', type=int, default=10)
    parser.add_argument('--patch_length', type=int, default=13)

    
    # SouthSea
    # parser.add_argument('--data_path', type=str, default='SouthSea.csv', help='data file')
    # parser.add_argument('--num_nodes', type=int, default=481)
    # parser.add_argument('--width', type=int, default=20)
    # parser.add_argument('--length', type=int, default=24)
    # parser.add_argument('--patch_width', type=int, default=5)
    # parser.add_argument('--patch_length', type=int, default=6)


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=15, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=True, help='See utils/tools for usage')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            if args.model == "PiFormer2":
                setting = '{}_{}_ft{}_sl{}_pl{}_{}_d{}_lr{}_dp{}_bs{}_ps{}_dff{}_pw{}_pl{}'.format(
                    args.model_id,
                    args.model,
                    args.data_path[:-4],
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.learning_rate,
                    args.dropout,
                    args.batch_size,
                    args.patch_size, 
                    args.d_ff,
                    args.patch_width,
                    args.patch_length, ii)
            else:
                setting = '{}_{}_ft{}_sl{}_pl{}_{}_d{}_lr{}_dp{}_bs{}_ps{}_dff{}'.format(
                    args.model_id,
                    args.model,
                    args.data_path[:-4],
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.learning_rate,
                    args.dropout,
                    args.batch_size,
                    args.patch_size, 
                    args.d_ff, ii)

            exp = Exp(args)  # set experiments
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            time_now = time.time()
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            print('Inference time: ', time.time() - time_now)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        if args.model == "PiFormer2":
            setting = '{}_{}_ft{}_sl{}_pl{}_{}_d{}_lr{}_dp{}_bs{}_ps{}_dff{}_pw{}_pl{}'.format(
                args.model_id,
                args.model,
                args.data_path[:-4],
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.learning_rate,
                args.dropout,
                args.batch_size,
                args.patch_size, 
                args.d_ff,
                args.patch_width,
                args.patch_length, ii)
        else:
            setting = '{}_{}_ft{}_sl{}_pl{}_{}_d{}_lr{}_dp{}_bs{}_ps{}_dff{}'.format(
                args.model_id,
                args.model,
                args.data_path[:-4],
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.learning_rate,
                args.dropout,
                args.batch_size,
                args.patch_size, 
                args.d_ff, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
