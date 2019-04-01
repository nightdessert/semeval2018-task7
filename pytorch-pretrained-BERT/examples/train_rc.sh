#!/usr/bin/env bash

base_path=/home/v-yinguo/Amcute/repos/semeval2018-task7
subtask=1.1
train_file=train_$subtask.json
test_file=test_$subtask.json

 python3 $base_path/pytorch-pretrained-BERT/examples/run_rc.py \
--train_file $base_path/bert_data/$train_file \
--predict_file $base_path/bert_data/$test_file \
--output_dir $base_path/bert_prediction_result \
--do_train \
--do_predict \
--bert_model bert-base-uncased \
--verbose_logging \
--do_train \
--num_train_epochs 15 \
--train_batch_size 256 \
--gradient_accumulation_steps 4 \
--do_lower_case \

