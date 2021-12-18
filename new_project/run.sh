METHOD=$1
ADAPT_METHOD=$2
MODEL_NAME_OR_PATH=$3
TASK=$4
TRAIN_EXAMPLES=$5

DATA_ROOT='data/'
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=32
ACCU=1
MAX_STEPS=2500
LOGGING_STEPS=50

if [ $TASK = "cola" ]; then
  DATA_DIR=${DATA_ROOT}cola
  SEQ_LENGTH=64
elif [ $TASK = "mnli" ] || [ $TASK = "mnli-mm" ]; then
  DATA_DIR=${DATA_ROOT}mnli
  SEQ_LENGTH=256
elif [ $TASK = "mrpc" ]; then
  DATA_DIR=${DATA_ROOT}mrpc
  SEQ_LENGTH=128
elif [ $TASK = "sst-2" ]; then
  DATA_DIR=${DATA_ROOT}sst-2
  SEQ_LENGTH=64
elif [ $TASK = "sts-b" ]; then
  DATA_DIR=${DATA_ROOT}sts-b
  SEQ_LENGTH=64
elif [ $TASK = "qqp" ]; then
  DATA_DIR=${DATA_ROOT}qqp
  SEQ_LENGTH=256
  LOGGING_STEPS=100
elif [ $TASK = "qnli" ]; then
  DATA_DIR=${DATA_ROOT}qnli
  SEQ_LENGTH=128
elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}rte
  SEQ_LENGTH=128
elif [ $TASK = "wnli" ]; then
  DATA_DIR=${DATA_ROOT}wnli
  SEQ_LENGTH=64
fi


#CUDA_VISIBLE_DEVICES=$device
python3 fire.py \
--method $METHOD \
--adapt_method $ADAPT_METHOD \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--task_name $TASK \
--max_length $SEQ_LENGTH \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--gradient_accumulation_steps $ACCU \
--max_steps $MAX_STEPS \
--train_examples $TRAIN_EXAMPLES \
--logging_steps $LOGGING_STEPS