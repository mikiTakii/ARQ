export CUDA_VISIBLE_DEVICES=0
python -W ignore rl_quantize.py     \
 --arch qresnet20                   \
 --dataset cifar10                  \
 --dataset_root data/cifar10        \
 --suffix sigma0.5_3bit             \
 --preserve_ratio 0.0084            \
 --float_bit 32                     \
 --max_bit 8                        \
 --min_bit 2                        \
 --n_worker 16                      \
 --data_bsize 256                   \
 --train_size 10000                 \
 --val_size 500                     \
 --linear_quantization              \
 --sigma 0.5                        \
 --n1 500                           \
 --n2 1000                          \
