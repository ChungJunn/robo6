GPU_ID=$1
EXP_NAME='SGD5'

DATA_DIR=$HOME'/chl/robo6/data/'
SAVE_DIR=$HOME'/chl/robo6/'

MODEL_OUT_FILE=$SAVE_DIR$EXP_NAME'.pth'
TEST_FILE=$DATA_DIR'test.csv'
RESULT_OUT_FILE=$SAVE_DIR$EXP_NAME'.result'

TR_FILE=$DATA_DIR'train.csv'
VAL_FILE=$DATA_DIR'valid.csv'

BATCH_SIZE=32
OPTIMIZER='SGD'
LR=0.1
DIM_HIDDEN=20
N_LAYERS=2

PRINT_EVERY=1000
VALID_EVERY=10000

HORIZON=30

export CUDA_VISIBLE_DEVICES=$GPU_ID 
python3 fs_run.py --tr_file=$TR_FILE --val_file=$VAL_FILE --model_out_file=$MODEL_OUT_FILE \
    --batch_size=$BATCH_SIZE --optimizer=$OPTIMIZER --lr=$LR \
    --dim_hidden=$DIM_HIDDEN --n_layer=$N_LAYERS \
    --print_every=$PRINT_EVERY --valid_every=$VALID_EVERY \
    --loadPath=$MODEL_OUT_FILE --result_out_file=$RESULT_OUT_FILE --test_file=$TEST_FILE \
    --horizon=$HORIZON --train --test
