EXP_NAME='hello'

DATA_DIR=$HOME'/chl/robo6/data'
SAVE_DIR=$HOME'/chl/robo6/'

MODEL_FILE=$SAVE_DIR$EXP_NAME'.pth'
OUT_FILE=$SAVE_DIR$EXP_NAME'.result'

python3 test.py --loadPath=$MODEL_FILE  --out_file=$OUT_FILE
