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

parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--hidden_size', type=int, default=8, help='')
parser.add_argument('--savePath', type=str, required=True, help='')
parser.add_argument('--max_epochs', type=int, default=1, help='')

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
    
    return current_loss / (tr_x.size(0) * i)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    batch_size = args.batch_size #TODO: batchsize and seq_len is the issue to be addressed
    n_epoches = args.max_epochs 

    trainiter = FSIterator("./data/train.csv", batch_size = batch_size)
    validiter = FSIterator("./data/valid.csv", batch_size = batch_size, just_epoch=True) # batchd_size 1 is recommended, since remainder is discard
 
    device = torch.device("cuda")     

    #TODO variables need to be args
    # setup model
    from model import FS_MODEL1, FS_MODEL2
    input_size = 1
    hidden_size = args.hidden_size
    output_size = 2
    
    model = FS_MODEL2(input_size, hidden_size, output_size, batch_size).to(device)
    #model = FS_MODEL1(input_size, hidden_size, output_size, batch_size).to(device)

    # define loss
    criterion = nn.NLLLoss(reduction='none')
    optimizer = optim.RMSprop(model.parameters())
    
    print_every = 100
    valid_every = 200

    start = time.time()

    patience = 5    
    savePath = args.savePath
   
 
    def train_main(model, trainiter, validiter, optimizer, device, print_every, valid_every):
        all_losses=[]
        current_loss =0
        valid_loss = 0.0
        bad_counter = 0
        best_loss = -1

        for i, (tr_x, tr_y, xm, end_of_file) in enumerate(trainiter):
            tr_x, tr_y, xm = torch.FloatTensor(tr_x), torch.LongTensor(tr_y), torch.FloatTensor(xm)
            tr_x, tr_y, xm = Variable(tr_x).to(device), Variable(tr_y).to(device), Variable(xm).to(device)

            if tr_x.size(0) - 1 == 0: # single-day data
                continue

            output, loss = train(model, tr_x, xm, tr_y, optimizer, criterion)
            current_loss += loss

            # print iter number, loss, prediction, and target
            if (i+1) % print_every == 0:
                top_n, top_i = output.topk(1)
                #correct = 'correct' if top_i[0].item() == target[0].item() else 'wrong'
                print("%d (%s) %.4f" % (i+1,timeSince(start), current_loss/print_every))
                all_losses.append(current_loss / print_every)

                current_loss=0
        
            if (i+1) % valid_every == 0:  
                valid_loss = validate(model, validiter)
                print("valid loss : {}".format(valid_loss))
        
            if valid_loss < best_loss or best_loss < 0:
                bad_counter = 0
                torch.save(model, savePath)

            else:
                bad_counter += 1

            if bad_counter > patience:
                print('Early Stopping')
                break
    
    train_main(model, trainiter, validiter, optimizer, device, print_every, valid_every)
     
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(all_losses)
    plt.savefig(args.savePath + ".png")

