export CUDA_VISIBLE_DEVICES=3

model_name=TimeAlign

seq_len=720

w_align=0.1
patch_num=1
dropout=0.1
local_margin=0.5
global_margin=0.0


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --global_margin $global_margin \
  --itr 1 


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --global_margin $global_margin \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.8 \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --global_margin $global_margin \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.9 \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --global_margin $global_margin \
  --itr 1
