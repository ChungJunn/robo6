import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import argparse

from model import FS_MODEL1, FS_MODEL2
from data import FSIterator

parser = argparse.ArgumentParser()
parser.add_argument('--loadPath', type=str, required=True, help='')
parser.add_argument('--horizon', type=int, default=30, help='')
parser.add_argument('--out_file', type=str, help='')

args = parser.parse_args()

loadPath = args.loadPath
batch_size = 1 # this is fixed to 1 at testing

device = torch.device("cpu")
model = torch.load(loadPath).to(device)

testiter = FSIterator("./data/test.csv", batch_size = batch_size, just_epoch=True)

n_totals = np.zeros(args.horizon)
n_targets = np.zeros((2, args.horizon))
n_corrects = np.zeros((2, args.horizon))

for i, (x, y, xm, end_of_file) in enumerate(testiter):
    x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.int32)    
    output, hidden = model(x, None)
    logit, pred = output.topk(1)
    
    for t in range(args.horizon): 
        if t >= x.shape[0]: break

        n_totals[t] += 1
        n_targets[y[t].item(),t] += 1
        
        if pred[t].item() == y[t].item():
            n_corrects[pred[t].item(),t] += 1

    if i == 10: break #TODO for temporaray testing

accs = []
for i in range(args.horizon):
    if n_totals[i] != 0:
        accs.append(str((np.sum(n_corrects[:,i]) / n_totals[i])))
 
mystring = loadPath
accString = ','.join(accs)

with open(args.out_file, "a") as fp:
    fp.write(accString)
print(accString)
