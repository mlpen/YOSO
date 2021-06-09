
CUDA_VISIBLE_DEVICES=0 python3 run_glue.py --model $1 --batch_size 32 --lr 2e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --model $1 --batch_size 32 --lr 3e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=2 python3 run_glue.py --model $1 --batch_size 32 --lr 4e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=3 python3 run_glue.py --model $1 --batch_size 32 --lr 5e-5 --task $3 --checkpoint $2 &
wait
CUDA_VISIBLE_DEVICES=0 python3 run_glue.py --model $1 --batch_size 16 --lr 2e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --model $1 --batch_size 16 --lr 3e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=2 python3 run_glue.py --model $1 --batch_size 16 --lr 4e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=3 python3 run_glue.py --model $1 --batch_size 16 --lr 5e-5 --task $3 --checkpoint $2 &
wait
CUDA_VISIBLE_DEVICES=0 python3 run_glue.py --model $1 --batch_size 8 --lr 2e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=1 python3 run_glue.py --model $1 --batch_size 8 --lr 3e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=2 python3 run_glue.py --model $1 --batch_size 8 --lr 4e-5 --task $3 --checkpoint $2 &
CUDA_VISIBLE_DEVICES=3 python3 run_glue.py --model $1 --batch_size 8 --lr 5e-5 --task $3 --checkpoint $2 &
wait
