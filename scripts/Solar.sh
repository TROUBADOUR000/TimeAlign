export CUDA_VISIBLE_DEVICES=6

model_name=TimeAlign

seq_len=720

w_align=0.2
dropout=0.3
patch_num=1
local_margin=0.0


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0005 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --batch_size 32 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0005 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0005 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --batch_size 32 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.txt \
  --model_id solar_$seq_len'_'720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0005 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --batch_size 32 \
  --itr 1
