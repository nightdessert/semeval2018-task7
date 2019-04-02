#!/usr/bin/env bash

base_path=/home/v-yinguo/Amcute/repos/semeval2018-task7
subtask=1.1
train_file=train_$subtask.json
test_file=test_$subtask.json
ans_file=$base_path/test-data/$subtask/relations.txt
predict_file=$base_path/bert_prediction_result/$subtask 

 python3 $base_path/pytorch-pretrained-BERT/examples/eval_rc.py \
--train_file $base_path/bert_data/$train_file \
--predict_file $base_path/bert_data/$test_file \
--output_dir $base_path/bert_prediction_result/model_$subtask \
--do_predict \
--bert_model bert-base-uncased \
--verbose_logging \
--checkpoint_index 13 \
--num_train_epochs 30 \
--predict_batch_size 32 \
--gradient_accumulation_steps 16 \
--do_lower_case \
> $predict_file


perl $base_path/data/semeval2018_task7_scorer-v1.2.pl $predict_file $ans_file

