export CUDA_VISIBLE_DEVICES=0
python -W ignore rl_quantize.py     \
 --arch qresnet50                   \
 --dataset imagenet                 \
 --dataset_root data/imagenet       \
 --suffix sigma0.5_3bit             \
 --preserve_ratio 0.015             \
 --float_bit 32                     \
 --max_bit 8                        \
 --min_bit 2                        \
 --train_size 60000                 \
 --n_worker 32                      \
 --data_bsize 128                   \
 --linear_quantization              \
 --sigma 0.5                        \
 --n1 500                           \
 --n2 500                           \