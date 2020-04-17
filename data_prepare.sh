DATA_DIR=$HOME'/chl/robo6/data/initial/'
DATA_FILE=$DATA_DIR'robo_train.csv'

TR_FILE=$DATA_DIR'train.csv'
VAL_FILE=$DATA_DIR'valid.csv'
TEST_FILE=$DATA_DIR'test.csv'

SEED=3224

python data_prepare.py --data_file=$DATA_FILE --tr_out_file=$TR_FILE \
                       --val_out_file=$VAL_FILE --test_out_file=$TEST_FILE \
                       --seed=$SEED

