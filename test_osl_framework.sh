CUDA_VISIBLE_DEVICES=gpu_id python main.py\
 --model_name corediff\
 --run_name dose5_mayo2016_sim\
 --test_batch_size 1\
 --test_dataset mayo_2016_sim\
 --test_id 9\
 --context\
 --only_adjust_two_step\
 --dose 50\
 --test_iter 150000\
 --mode test_osl_framework\

