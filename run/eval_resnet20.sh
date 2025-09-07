export CUDA_VISIBLE_DEVICES=0
python -W ignore rl_quantize.py     \
 --arch qresnet20                   \
 --dataset cifar10                  \
 --dataset_root data/cifar10        \
 --suffix sigma0.5_3bit_E           \
 --float_bit 32                     \
 --max_bit 8                        \
 --min_bit 2                        \
 --n_worker 16                      \
 --linear_quantization              \
 --sigma 0.5                        \
 --evaluate                         \
