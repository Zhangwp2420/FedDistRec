import argparse
import os

ROOT = os.path.abspath(os.path.join(os.path.join(__file__), '../../'))
LOG_ROOT = os.path.join(ROOT, 'logs')
DATA_ROOT = os.path.join(ROOT, 'data')


def get_parser():

    parser = argparse.ArgumentParser('Argument Parser')

    parser.add_argument('--model', type=str,required=True)

    parser.add_argument('--seed', '-seed', type=int, default=2025)

    parser.add_argument('--dataset', '-d', type=str, required=True)
    
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--patience', '-p', type=int, default=5)

    parser.add_argument('--warmup', '-warmup', type=int, default=0)
    
    parser.add_argument('--eval_interval', '-ei', type=int, default=1)
  
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)

    parser.add_argument('--num_round', type=int, default=100)

    parser.add_argument('--local_epoch', type=int, default=5)

    parser.add_argument('--batch_size', '-bs', type=int, default=256)
   
    parser.add_argument('--lr', '-lr', type=float, default=0.0001)

    parser.add_argument('--wd', '-wd', type=float, default=1e-5)

    parser.add_argument('--latent_dim','-dim',type=int, default=32)

    parser.add_argument('--num_negative', type=int, default=4)

    parser.add_argument('--scale', type=float, default=1000)

    parser.add_argument('--beta_start', type=float, default=0.0001)

    parser.add_argument('--beta_end', type=float, default=0.02)

    parser.add_argument('--beta_base', type=float, default=0.01)  

    parser.add_argument('--diffusion_steps', '-ds', type=int, default=32)

    parser.add_argument('--lambda_rec_loss', '-lrec', type=float, default=1.0)

    parser.add_argument('--lambda_mse_loss', '-lmse', type=float, default=1.0)
    
    parser.add_argument('--top_k', type=int, default=10)

    parser.add_argument('--nmc', '-nmc', type=int, default=5)
   
    return parser


def get_args():
    
    parser = get_parser()
    args = parser.parse_args()
    return args, parser

