#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --version sagri \
    --model SpanAttModelV3 \
    --bert_name_or_path dmis-lab/biobert-base-cased-v1.1 \
    --learning_rate 3e-5 \
    --batch_size 1 \
    --gradient_accumulation_steps 128 \
    --train_epoch 41 \
    --score tri_affine \
    --truncate_length 192 \
    --word \
    --word_dp 0.2 \
    --char \
    --pos \
    --use_context \
    --warmup_ratio 0.0 \
    --att_dim 200 \
    --bert_before_lstm \
    --lstm_dim 1024 \
    --lstm_layer 2 \
    --encoder_learning_rate 5e-4 \
    --max_span_count 30 \
    --share_parser \
    --subword_aggr max \
    --init_std 1e-2 \
    --dp 0.2 \
    --continue_from /workspace/nested_ner/output/sagri-SpanAttModelV3-biobert-base-cased-v1.1-1.0-1.0-2-30-200-0.01-max-context-bbl-word_0.2-char_50_0.2-pos_50_0.2-lstm_1024_0.2_2-sps-warm0.0-0.0-len_192-epoch_40-lr_3e-05_0.0005_0.0-bsz_128-tti-mlpdp_0.2-/epoch30.pth
    #--continue_from /workspace/nested_ner/output/sagri-SpanAttModelV3-biobert-base-cased-v1.1-1.0-1.0-2-30-200-0.01-max-context-bbl-word_0.2-char_50_0.2-pos_50_0.2-lstm_1024_0.2_2-sps-warm0.0-0.0-len_192-epoch_15-lr_3e-05_0.0005_0.0-bsz_128-tti-mlpdp_0.2-/epoch7.pth