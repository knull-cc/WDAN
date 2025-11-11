exp_id=final
seq_len=720
fix_seed=2025
itr=3
machine=local
server_name=Local
precision=full
gpu_id=0
export CUDA_VISIBLE_DEVICES=0

# region ETTh1
python -u run.py \
  --model WDAN_SimpleTM \
  --data ETTh1 \
  --fix_seed $fix_seed \
  --itr 3 \
  --machine $machine \
  --server_name $server_name \
  --exp_id $exp_id \
  --seq_len $seq_len\
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --use_norm 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 128 \
  --stats_dwt_levels 2 \
  --stats_window_len 12 \
  --stats_d_model 128 \
  --stats_d_ff 128 \
  --stats_ffn_layers 0 \
  --twice_epoch 0 \
  --base_stats_lr 0.0001 \
  --stats_strategy stats_bb_union \
  --loss_type mse \
  --gpu_id $gpu_id \
  --m 3 \
  --alpha 0.0 \
  --wv "db1" \
  --base_lr 0.0001