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

if __name__ == "__main__":
     
    iterator = FSIterator("./data/train.csv")

    for item in iterator:
        print(item)
