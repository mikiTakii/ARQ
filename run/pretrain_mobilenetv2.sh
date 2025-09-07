export CUDA_VISIBLE_DEVICES=0
python -W ignore pretrain.py                    \
 -a mobilenetv2                                 \
 --data_name imagenet                           \
 -c checkpoint/mobilenetv2                      \
 -d data/imagenet                               \
 --epochs 150                                   \
 --lr 0.05                                      \
 --wd 0.00004                                   \
 --train_batch 256                              \
 --workers 16                                   \
 --noise 0.5                                    \