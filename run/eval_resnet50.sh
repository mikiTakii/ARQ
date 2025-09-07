export CUDA_VISIBLE_DEVICES=0
python -W ignore rl_quantize.py     \
 --arch qresnet50                   \
 --dataset imagenet                 \
 --dataset_root data/imagenet       \
 --suffix sigma0.5_3bit_E           \
 --preserve_ratio 1                 \
 --float_bit 32                     \
 --max_bit 8                        \
 --min_bit 2                        \
 --n_worker 32                      \
 --linear_quantization              \
 --sigma 0.5                        \
 --evaluate                         \