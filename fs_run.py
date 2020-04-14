import sys
import argparse
import pickle as pkl

from main import train_main

parser = argparse.ArgumentParser()
parser.add_argument('--tr_file', type=str, help='')
parser.add_argument('--val_file', type=str, help='')
parser.add_argument('--out_file', type=str, help='')

parser.add_argument('--batch_size', type=int, help='')
parser.add_argument('--optimizer', type=str, help='')
parser.add_argument('--lr', type=float, help='')
parser.add_argument('--dim_hidden', type=int, help='')
parser.add_argument('--n_layers', type=int, help='')
parser.add_argument('--patience', type=int, default=5, help='')
parser.add_argument('--input_size', type=int, default=1, help='')
parser.add_argument('--output_size', type=int, default=2, help='')
parser.add_argument('--print_every', type=int, default=1000, help='')
parser.add_argument('--valid_every', type=int, default=5000, help='')

args = parser.parse_args()

pkl.dump(args, open(args.out_file+'.args.pkl', 'wb'), -1)
with open(args.out_file + '.args', 'w') as fp:
    for key in vars(args):
        fp.write(key + ': ' + str(getattr(args, key)) + '\n')        

tr_losses, val_losses = train_main(args)

''' 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot(val_losses)
plt.savefig(args.out_dir + ".png")
'''
