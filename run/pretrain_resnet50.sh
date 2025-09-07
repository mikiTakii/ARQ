export CUDA_VISIBLE_DEVICES=0
python -W ignore pretrain.py                    \
 -a resnet50                                    \
 --data_name imagenet                           \
 -c checkpoint/resnet50                         \
 -d data/imagenet                               \
 --epochs 90                                    \
 --lr 0.1                                       \
 --wd 1e-4                                      \
 --lr_type step3                                \
 --train_batch 128                              \
 --workers 16                                   \
 --noise 0.5                                    \