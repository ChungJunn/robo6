'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

def getSeq_len(row):
    '''
    returns : count of non-nans (integer)
    adopted from: M4rtni's answer in stackexchange
    '''
    return np.count_nonzero(~np.isnan(row))

def trimBatch(batch):
    '''
    args: npndarray of a batch (bsz, n_features)
    returns: trimmed npndarray of a batch.
    '''
    max_seq_len = 0
    for n in range(batch.shape[0]):
        max_seq_len = max(max_seq_len, getSeq_len(batch[n]))

    if max_seq_len == 0:
        print("error in trimBatch()")
        sys.exit(-1)

    batch = batch[:,:max_seq_len]
    return batch

def getMask(batch):
    '''
    returns: boolean array indicating whether nans
    '''
    return (~np.isnan(batch)).astype(np.int32)

def pp_row(row):
    xlist = []
    ylist = []
    xmlist = []

    for i in range(len(row)):
        if i >= len(row)-1 or np.isnan(row[i+1]):
            break

        xlist.append(row[i])
        lbl = int((row[i+1] - row[i]) > 0)
        ylist.append(lbl)
        xmlist.append(1)

    for j in range(i+1, len(row)):
        xlist.append(0)
        ylist.append(0)
        xmlist.append(0)

    x = np.array(xlist)
    y = np.array(ylist).astype(np.int32)
    xm = np.array(xmlist).astype(np.int32)

    return x, y, xm

def pp_batch(batch):
    xlist = []
    ylist = []
    xmlist = []

    for n in range(batch.shape[0]):
        x, y, xm = pp_row(batch[n])
        xlist.append(x)
        ylist.append(y)
        xmlist.append(xm)

    xs = np.vstack(xlist)
    ys = np.vstack(ylist)
    xms = np.vstack(xmlist)

    return xs, ys, xms

class FSIterator:
    def __init__(self, filename, batch_size=32, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch
        self.fp = open(filename, 'r')
    
    def __iter__(self):
        return self

    def reset(self):
        self.fp.seek(0)

    def __next__(self):
        bat_seq = []

        end_of_data = 0
        for i in range(self.batch_size):
            seq = self.fp.readline()
            if seq == "":
                if self.just_epoch:
                    end_of_data = 1
                    if self.batch_size==1:
                        raise StopIteration
                    else:
                        self.reset()
                        break
                self.reset()
                seq = self.fp.readline()

            seq_f = [float(s) for s in seq.split(',')]

            bat_seq.append(seq_f)
        
        bat_seq = trimBatch(np.array(bat_seq))
        x_data, y_data, mask = pp_batch(bat_seq)

        x_data = np.expand_dims(x_data.transpose(1,0), axis=-1)
        y_data, mask = y_data.transpose(1,0), mask.transpose(1,0)
        

        return x_data, y_data, mask, end_of_data

class FSIterator2:
    def __init__(self, filename, batch_size=32, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch
        self.fp_end = open(filename + "/END.csv", 'r')
        self.fp_start = open(filename + "/START.csv", 'r')
        self.fp_low = open(filename + "/LOW.csv", 'r')
        self.fp_high = open(filename + "/HIGH.csv", 'r')
        self.fp_trade = open(filename + "/TRADE.csv", 'r')
        self.fps = [self.fp_end, self.fp_start, self.fp_low, self.fp_high, self.fp_trade]

    def __iter__(self):
        return self

    def reset(self):
        for fp in self.fps:
            fp.seek(0)

    def __next__(self):

        bat_seq = []
        touch_end = 0

        while(len(bat_seq)< self.batch_size):
            seq_end = self.fps[0].readline()
            seq_start = self.fps[1].readline()
            seq_low = self.fps[2].readline()
            seq_high = self.fps[3].readline()
            seq_trade = self.fps[4].readline()

            if touch_end:
                raise StopIteration

            if seq_end == "":
                print("touch end")
                touch_end = 1
                
                self.reset()
                # read the first line
                seq_end = self.fps[0].readline()
                seq_start = self.fps[1].readline()
                seq_low = self.fps[2].readline()
                seq_high = self.fps[3].readline()
                seq_trade = self.fps[4].readline()

            seq_end = [float(s) for s in seq_end.split(',')]
            seq_start = [float(s) for s in seq_start.split(',')]
            seq_low = [float(s) for s in seq_low.split(',')]
            seq_high = [float(s) for s in seq_high.split(',')]
            seq_trade = [float(s) for s in seq_trade.split(',')]

            #if(np.count_nonzero(~np.isnan(seq_end))>7 and seq_end[-1] == 1):
            #if(np.count_nonzero(~np.isnan(seq_end)) >= 10 and np.count_nonzero(~np.isnan(seq_end)) < 21): # short data
            #if(np.count_nonzero(~np.isnan(seq_end)) >= 21): # long data
                #if(np.count_nonzero(~np.isnan(seq_f))>4):
            if(sum(~np.isnan(seq_end)) == sum(~np.isnan(seq_start)) == sum(~np.isnan(seq_low))== sum(~np.isnan(seq_high)) == sum(~np.isnan(seq_trade))):
                if(max(seq_end)<=2):
                    seqs = [seq_end, seq_start, seq_low, seq_high, seq_trade]
                    bat_seq.append(seqs)

        x_data, y_data, mask_data = self.prepare_data(np.array(bat_seq)) # B x [[E*daylen],[S*daylen],[L*daylen],[H*daylen]]

        device = torch.device("cuda")
        x_data = torch.tensor(x_data).type(torch.float32).to(device)
        y_data = torch.tensor(y_data).type(torch.LongTensor).to(device)
        mask_data = torch.tensor(mask_data).type(torch.float32).to(device)

        return x_data, y_data, mask_data

    def prepare_data(self, seq):
        PRE_STEP = 1 # this is for delta
        #import pdb; pdb.set_trace()
        seq_end_x = seq[:,0,:-1]
        seq_start_x = seq[:,1,:-1]
        seq_low_x = seq[:,2,:-1]
        seq_high_x = seq[:,3,:-1]
        seq_trade_x = seq[:,4,:-1]        

        seq_y = seq[:,0,-1]
        
        # resize into the longest day length
        seq_end_x = trimBatch(seq_end_x)
        seq_start_x = trimBatch(seq_start_x)
        seq_low_x = trimBatch(seq_low_x)
        seq_high_x = trimBatch(seq_high_x)
        seq_trade_x = trimBatch(seq_trade_x)       

        seq_mask = getMask(seq_end_x[:,1:-PRE_STEP])
        
        seq_end_x = np.nan_to_num(seq_end_x)
        seq_start_x = np.nan_to_num(seq_start_x)
        seq_low_x = np.nan_to_num(seq_low_x)
        seq_high_x = np.nan_to_num(seq_high_x)
        seq_trade_x = np.nan_to_num(seq_trade_x)        

        seq_end_x_delta = seq_end_x[:,1:] - seq_end_x[:,:-1]
        seq_start_x_delta = seq_start_x[:,1:] - seq_start_x[:,:-1]
        seq_low_x_delta = seq_low_x[:,1:] - seq_low_x[:,:-1]
        seq_high_x_delta = seq_high_x[:,1:] - seq_high_x[:,:-1]
        seq_trade_x_delta = seq_trade_x[:,1:] - seq_trade_x[:,:-1]        

        try: 
            x_data = np.stack([seq_end_x[:,1:-PRE_STEP], seq_end_x_delta[:,:-PRE_STEP],
                           seq_start_x[:,1:-PRE_STEP], seq_start_x_delta[:,:-PRE_STEP],
                           seq_low_x[:,1:-PRE_STEP], seq_low_x_delta[:,:-PRE_STEP],
                           seq_high_x[:,1:-PRE_STEP], seq_high_x_delta[:,:-PRE_STEP], 
                           seq_trade_x[:,1:-PRE_STEP], seq_trade_x_delta[:,:-PRE_STEP]], axis=2) #batch * daylen * inputdim(2)
        except:
            import pdb; pdb.set_trace()

        x_data = x_data.transpose(1,0,2) # daylen * batch * inputdim
        
        y_data = seq_y.reshape(1,-1) # batch * 1
        y_data = np.stack([y_data.transpose(1,0)]) # 1*batch*1

        #y_data = (seq_delta[:,1:] > 0)*1.0 # the diff
        
        mask_data = np.stack(seq_mask.transpose(1,0))
        '''
        x_data : daymaxlen-2, batch, inputdim(=2)
        y_data : 1 * batch * 1
        mask_data : 1*daymaxlen-2, batch
        '''
        return x_data, y_data, mask_data

if __name__ == "__main__":
    myiter = FSIterator2("./data/0412/train")

    for x, y, xm in myiter:
        import pdb; pdb.set_trace()

