MODEL_NAME_OR_PATH=$1
TASK=$2
device=$3

DATA_ROOT='data/'
MAX_LENGTH=256
TRAIN_BATCH_SIZE=8
MAX_STEPS=100000

if [ $TASK = "cola" ]; then
  DATA_DIR=${DATA_ROOT}cola
  MAX_LENGTH=64
elif [ $TASK = "mnli" ] || [ $TASK = "mnli-mm" ]; then
  DATA_DIR=${DATA_ROOT}mnli
  MAX_LENGTH=256
elif [ $TASK = "mrpc" ]; then
  DATA_DIR=${DATA_ROOT}mrpc
  MAX_LENGTH=128
elif [ $TASK = "sst-2" ]; then
  DATA_DIR=${DATA_ROOT}sst-2
  MAX_LENGTH=64
elif [ $TASK = "sts-b" ]; then
  DATA_DIR=${DATA_ROOT}sts-b
  MAX_LENGTH=128
elif [ $TASK = "qqp" ]; then
  DATA_DIR=${DATA_ROOT}qqp
  MAX_LENGTH=256
elif [ $TASK = "qnli" ]; then
  DATA_DIR=${DATA_ROOT}qnli
  MAX_LENGTH=256
elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}rte
  MAX_LENGTH=256
elif [ $TASK = "wnli" ]; then
  DATA_DIR=${DATA_ROOT}wnli
  MAX_LENGTH=128
fi


CUDA_VISIBLE_DEVICES=$device python3 domain_adapt.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--task_name $TASK \
--max_length $MAX_LENGTH \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--max_steps $MAX_STEPS \

