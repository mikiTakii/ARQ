export CUDA_VISIBLE_DEVICES=0
python -W ignore pretrain.py                    \
 -a resnet20                                    \
 --data_name cifar10                            \
 -c checkpoint/resnet20                         \
 -d data/cifar10                                \
 --epochs 90                                    \
 --lr 0.1                                       \
 --wd 1e-4                                      \
 --lr_type step3                                \
 --train_batch 256                              \
 --workers 16                                   \
 --noise 0.5                                    \