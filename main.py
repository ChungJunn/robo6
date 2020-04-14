'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np

import math
import sys
import time

import argparse

from data import FSIterator

parser = argparse.ArgumentParser()
parser.add_argument('--tr_file', type=str, help='')
parser.add_argument('--val_file', type=str, help='')
parser.add_argument('--out_dir', type=str, help='')

parser.add_argument('--batch_size', type=int, help='')
parser.add_argument('--optimizer', type=str, help='')
parser.add_argument('--lr', type=float, help='')
parser.add_argument('--dim_hidden', type=int, help='')
parser.add_argument('--n_layers', type=int, help='')
parser.add_argument('--patience', type=int, default=5, help='')
parser.add_argument('--input_size', type=int, default=1, help='')
parser.add_argument('--output_size', type=int, default=2, help='')

args = parser.parse_args()

def train(model, input, mask, target, optimizer, criterion):
    model.train()

    loss_matrix = []
    
    optimizer.zero_grad()

    output, hidden = model(input, None)
   
    # Old Code applying mask and obtaining mean loss
    '''
    for t in range(input.size(0)):
        loss = criterion(output[t], target[t].view(-1))
        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)
    mask = mask[:(input.size(0)), :]
    
    masked = loss_matrix * mask
    
    loss = torch.sum(masked) / torch.sum(mask)
    '''    
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss * mask.view(-1)
    loss = torch.sum(loss) / torch.sum(mask)

    loss.backward()
    
    optimizer.step()

    return output, loss.item()

def evaluate(model, input, target, mask, criterion):
    loss_matrix = []
    
    output, hidden = model(input, None)
    
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    loss = loss * mask.view(-1)
    loss = torch.sum(loss) / torch.sum(mask)

    return output, loss.item()

def validate(model, validiter):
    current_loss = 0
    model.eval()
    with torch.no_grad(): 
        for i, (tr_x, tr_y, xm, end_of_file) in enumerate(validiter): 
            tr_x, tr_y, xm = torch.FloatTensor(tr_x), torch.LongTensor(tr_y), torch.FloatTensor(xm)
            tr_x, tr_y, xm = Variable(tr_x).to(device), Variable(tr_y).to(device), Variable(xm).to(device)
            
            if (tr_x.size(0)-1)==0: continue
            
            output, loss = evaluate(model, tr_x, tr_y, xm, criterion)
            current_loss += loss
            
            if end_of_file == 1:
                break
    
    return current_loss / i

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    trainiter = FSIterator(args.tr_file, batch_size = args.batch_size)
    validiter = FSIterator(args.val_file, batch_size = args.batch_size, just_epoch=True) # batchd_size 1 is recommended, since remainder is discard
 
    device = torch.device("cuda")     

    # setup model
    from model import FS_MODEL1, FS_MODEL2
    model = FS_MODEL2(args.input_size, args.dim_hidden, args.output_size, args.batch_size, args.n_layers).to(device)

    # define loss
    mystring = "optim." + args.optimizer
    optimizer = eval(mystring)(model.parameters(), args.lr)
    criterion = nn.NLLLoss(reduction='none')
    
    print_every = 1000
    valid_every = 5000

    start = time.time()
 
    def train_main(model, trainiter, validiter, optimizer, device, print_every, valid_every):
        tr_losses=[]
        val_losses=[]
        current_loss =0
        valid_loss = 0.0
        bad_counter = 0
        best_loss = -1

        for i, (tr_x, tr_y, xm, end_of_file) in enumerate(trainiter):
            tr_x, tr_y, xm = torch.FloatTensor(tr_x), torch.LongTensor(tr_y), torch.FloatTensor(xm)
            tr_x, tr_y, xm = Variable(tr_x).to(device), Variable(tr_y).to(device), Variable(xm).to(device)

            output, loss = train(model, tr_x, xm, tr_y, optimizer, criterion)
            current_loss += loss

            # print iter number, loss, prediction, and target
            if (i+1) % print_every == 0:
                
                top_n, top_i = output.topk(1)
                print("%d (%s) %.4f" % (i+1,timeSince(start), current_loss/print_every))
                tr_losses.append(current_loss / print_every)

                current_loss=0
        
            if (i+1) % valid_every == 0:
                valid_loss = validate(model, validiter)
                print("val : {:.4f}".format(valid_loss))
        
                if valid_loss < best_loss or best_loss < 0:
                    bad_counter = 0
                    torch.save(model, args.out_dir)
                    val_losses.append(valid_loss)                
                    best_loss = valid_loss

                else:
                    bad_counter += 1

                if bad_counter > args.patience:
                    print('Early Stopping')
                    break
   
        return tr_losses, val_losses
    
    tr_losses, val_losses = train_main(model, trainiter, validiter, optimizer, device, print_every, valid_every)
    
    ''' 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(val_losses)
    plt.savefig(args.out_dir + ".png")
    '''
