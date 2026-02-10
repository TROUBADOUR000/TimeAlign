export CUDA_VISIBLE_DEVICES=4

model_name=TimeAlign

seq_len=720

w_align=0.1
dropout=0.1
patch_num=1
local_margin=0.5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --train_epochs 30 \
  --learning_rate 0.001 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --pos 0 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --train_epochs 30 \
  --learning_rate 0.001 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --pos 0 \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --train_epochs 30 \
  --learning_rate 0.001 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --pos 0 \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 16 \
  --train_epochs 30 \
  --learning_rate 0.001 \
  --dropout $dropout \
  --w_align $w_align \
  --patch_num $patch_num \
  --local_margin $local_margin \
  --pos 0 \
  --itr 1
