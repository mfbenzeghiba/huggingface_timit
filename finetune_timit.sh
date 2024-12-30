#!/usr/bin/env bash

# model and dataset
model_name_or_path="facebook/wav2vec2-base"
dataset_name='Timit'

# path to train and valid csv files
# these files are generate by the prepare_data.py script.
train_csv='*/processed_train.csv'
valid_csv='*/processed_valid.csv'
phone_mapping_file='*/phones.60-48-39.map.txt'

exp_dir='Your_exp_dir'

# path to the vocab file if you have already created one.
# otherwise the file will be created by the script.
vocab_file='path_to_the_vocab_file'

# define the phone set 61to39, 61to48 or 61to61(no reduction)
phone_mapping_key="61to39"
audio_column_name='audio'
modeling_unit='char' # 'phoneme' or 'char'

overwrite_output_dir=true
overwrite_cache=true
  
# training hyper-params
preprocessing_num_workers=1
num_epochs=50
per_device_train_batch_size=8
per_device_eval_batch_size=8
gradient_acc=1
learning_rate="5e-4"
warmup_steps=1000

# eval params
eval_on_start=true
eval_strategy="epoch"
save_strategy="epoch"
metric_for_best_model="eval_ter"
greater_is_better=false
load_best_model_at_end=true

logging_steps=100
logging_strategy='epoch'
logging_first_step=true
freeze_feature_encoder=true

# Data augmentation details
layerdrop="0.0"
activation_dropout="0.2"
attention_dropout="0.2"
hidden_dropout="0.2"
feat_proj_dropout="0.0"
mask_time_prob="0.0"
mask_time_length="10"
mask_feature_prob="0.0"
mask_feature_length="64"

if [ "$modeling_unit" = "char" ]; then
  text_column_name="text"
elif [ "$modeling_unit"="phoneme" ]; then
  text_column_name='phonetic'
else
  echo "modeling unit can be either char or phoneme, you give: $modeling_unit"
  exit 0
fi


if [ "$freeze_feature_encoder" = "true" ]; then
  optional_args=" --freeze_feature_encoder"
fi
if [ "$overwrite_output_dir" = "true" ]; then
  optional_args="$optional_args --overwrite_output_dir"
fi

base_dir="${modeling_unit}_v2"
output_dir="${exp_dir}\\${base_dir}"
echo "$base_dir $output_dir"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir";
fi

python run_wav2vec2_timit.py \
    --audio_column_name=$audio_column_name \
    --activation_dropout=$activation_dropout \
    --attention_dropout=$attention_dropout \
    --dataset_name=$dataset_name \
    --do_train \
    --do_eval \
    --evaluation_strategy=$eval_strategy \
    --feat_proj_dropout=$feat_proj_dropout \
    --fp16 \
    --greater_is_better=$greater_is_better \
    --gradient_accumulation_steps=$gradient_acc \
    --group_by_length \
    --hidden_dropout=$hidden_dropout \
    --layerdrop=$layerdrop \
    --learning_rate=$learning_rate \
    --load_best_model_at_end=$load_best_model_at_end \
    --logging_dir=$logging_dir \
    --logging_first_step=$logging_first_step \
    --logging_strategy=$logging_strategy \
    --mask_feature_prob=$mask_feature_prob \
    --mask_feature_length=$mask_feature_length \
    --mask_time_length=$mask_time_length \
    --mask_time_prob=$mask_time_prob \
    --metric_for_best_model=$metric_for_best_model \
    --model_name_or_path=$model_name_or_path \
    --modeling_unit=$modeling_unit \
    --num_train_epochs=$num_epochs \
    --output_dir=$output_dir \
    --overwrite_cache=$overwrite_cache \
    --per_device_eval_batch_size=$per_device_eval_batch_size \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --phone_mapping_file=$phone_mapping_file \
    --preprocessing_num_workers="$preprocessing_num_workers" \
    --report_to="none" \
    --run_name="$output_dir" \
    --save_strategy=$save_strategy \
    --text_column_name=$text_column_name \
    --train_csv=$train_csv \
    --valid_csv=$valid_csv \
    --warmup_steps=$warmup_steps \
    --weight_decay=0.001 \
    $optional_args

echo "Done finetuning of $model_name_or_path on $dataset_name in $output_dir"
exit 0
