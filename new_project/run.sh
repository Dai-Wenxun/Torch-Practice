MODEL_NAME_OR_PATH=$1
TASK=$2
device=$3

DATA_ROOT='data/'
SEQ_LENGTH=256
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=16
ACCU=1

if [ $TASK = "cola" ]; then
  DATA_DIR=${DATA_ROOT}cola
  SEQ_LENGTH=64
  MAX_STEPS=3000
elif [ $TASK = "mrpc" ]; then
  DATA_DIR=${DATA_ROOT}mrpc
  SEQ_LENGTH=128
  MAX_STEPS=3000
elif [ $TASK = "sst-2" ]; then
  DATA_DIR=${DATA_ROOT}sst-2
  SEQ_LENGTH=64
  MAX_STEPS=10000
elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}rte
  SEQ_LENGTH=256
  MAX_STEPS=3000
fi


CUDA_VISIBLE_DEVICES=$device python3 fire.py \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--task_name $TASK \
--max_length $SEQ_LENGTH \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps $ACCU \
--max_steps $MAX_STEPS \

