CUDA_VISIBLE_DEVICES=0 python main.py\
 --model_name corediff\
 --run_name dose25_mayo_2016\
 --batch_size 4\
 --max_iter 150000\
 --test_dataset mayo_2016\
 --test_id 9\
 --context\
 --only_adjust_two_step\
 --dose 25\
 --save_freq 2500\
 --train_dataset mayo_2016\
 --wandb

