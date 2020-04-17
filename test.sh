EXP_NAME='hello'

DATA_DIR=$HOME'/chl/robo6/data/initial/'
SAVE_DIR=$HOME'/chl/robo6/'

TEST_FILE=$DATA_DIR'test.csv'
MODEL_FILE=$SAVE_DIR$EXP_NAME'.pth'
OUT_FILE=$SAVE_DIR$EXP_NAME'.result'

python3 test.py --loadPath=$MODEL_FILE --out_file=$OUT_FILE --test_file=$TEST_FILE
