DATA_DIR=$HOME'/chl/robo6/data'
DATA_FILE='robo_train.csv'

TR_FILE='train.csv'
VAL_FILE='valid.csv'
TEST_FILE='test.csv'

SEED=$1

python data_prepare.py --data_dir=$DATA_DIR --data_file=$DATA_FILE \
        --tr_out_file=$TR_FILE --val_out_file=$VAL_FILE --test_out_file=$TEST_FILE \
        --seed=$SEED

