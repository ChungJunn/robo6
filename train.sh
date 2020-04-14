GPU_ID=$1
EXP_NAME='adam'

DATA_DIR=$HOME'/chl/robo6/data/'
SAVE_DIR=$HOME'/chl/robo6/'

OUT_FILE=$SAVE_DIR$EXP_NAME'.pth'

TR_FILE=$DATA_DIR'train.csv'
VAL_FILE=$DATA_DIR'valid.csv'

BATCH_SIZE=32
OPTIMIZER='Adam'
LR=0.001
DIM_HIDDEN=20
N_LAYERS=2

PRINT_EVERY=1000
VALID_EVERY=5000

export CUDA_VISIBLE_DEVICES=$GPU_ID 
python3 fs_run.py --tr_file=$TR_FILE --val_file=$VAL_FILE --out_file=$OUT_FILE \
    --batch_size=$BATCH_SIZE --optimizer=$OPTIMIZER --lr=$LR \
    --dim_hidden=$DIM_HIDDEN --n_layer=$N_LAYERS \
    --print_every=$PRINT_EVERY --valid_every=$VALID_EVERY
