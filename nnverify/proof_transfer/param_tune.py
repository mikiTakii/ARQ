import argparse
import sys, os

sys.path.append('.')

import nnverify.proof_transfer.proof_transfer as pt
import nnverify.proof_transfer.approximate as ap
import csv
import optuna

from nnverify import config, common
from nnverify.bnb import Split
from nnverify.common import Domain
from nnverify.proof_transfer.pt_types import ProofTransferMethod, IVAN, REORDERING
from nnverify.common.dataset import Dataset

def tune_mnist_mlp2_prune(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    thr = trial.suggest_float("thr", 1e-5, 1e-1, log=True)

    args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                           dataset=Dataset.MNIST, eps=0.02,
                           split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(alpha, thr),
                           timeout=100)
    sp = pt.proof_transfer(args)

    write_result(alpha, sp, thr)
    return -sp

def tune_mnist_mlp2_reorder(trial):
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    thr = trial.suggest_float("thr", 1e-5, 1e-1, log=True)

    args = pt.TransferArgs(net=config.MNIST_FFN_L2, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                           dataset=Dataset.MNIST, eps=0.02,
                           split=Split.RELU_ESIP_SCORE, count=50, pt_method=REORDERING(alpha, thr),
                           timeout=100)
    sp = pt.proof_transfer(args)

    write_result(alpha, sp, thr)
    return -sp


def write_result(alpha, sp, thr):
    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'proof_transfer.csv'
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['thr=' + str(thr), 'alpha=' + str(alpha), 'speedup=', sp])


def tune_mnist_conv_prune(trial):
    thr = trial.suggest_float("thr", 1e-5, 1e-1, log=True)

    args = pt.TransferArgs(net=config.MNIST_FFN_01, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                           dataset=Dataset.MNIST, eps=0.1,
                           split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(thr),
                           timeout=100)
    sp = pt.proof_transfer(args)

    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'proof_transfer.csv'
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['thr=' + str(thr), 'speedup=', sp])
    return -sp

def tune_cifar_conv1_prune(trial):
    thr = trial.suggest_float("thr", 1e-5, 1e-1, log=True)

    args = pt.TransferArgs(net=config.CIFAR_CONV_SMALL, domain=Domain.LP, approx=ap.Quantize(ap.QuantizationType.INT16),
                           dataset=Dataset.CIFAR10, eps=2 / 255,
                           split=Split.RELU_ESIP_SCORE, count=50, pt_method=IVAN(thr), timeout=400)
    sp = pt.proof_transfer(args)

    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'proof_transfer.csv'
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['thr=' + str(thr), 'speedup=', sp])
    return -sp


def tune_acas_prune(trial):
    thr = trial.suggest_float("thr", 1e-5, 1e-1, log=True)

    args = pt.TransferArgs(domain=Domain.DEEPZ, approx=ap.Quantize(ap.QuantizationType.INT8),
                           dataset=Dataset.ACAS, split=Split.INPUT_SB, pt_method=IVAN(thr), timeout=100)
    sp = pt.proof_transfer_acas(args)

    os.makedirs(common.RESULT_DIR, exist_ok=True)
    file_name = common.RESULT_DIR + 'proof_transfer.csv'
    with open(file_name, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(['thr=' + str(thr), 'speedup=', sp])
    return -sp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='prune', help='prune or reorder')
    parser.add_argument('--task', type=int, default=0, help='task number')

    args = parser.parse_args()

    study = optuna.create_study()
    print(f"Sampler is {study.sampler.__class__.__name__}")

    if args.algo == 'prune':
        if args.task == 0:
            study.optimize(tune_mnist_mlp2_prune, n_trials=10)
        elif args.task == 1:
            study.optimize(tune_mnist_conv_prune, n_trials=10)
        elif args.task == 2:
            study.optimize(tune_cifar_conv1_prune, n_trials=10)
        elif args.task == 3:
            study.optimize(tune_acas_prune, n_trials=10)
    elif args.algo == 'reorder':
        if args.task == 0:
            study.optimize(tune_mnist_mlp2_reorder, n_trials=20)
    print(study.best_params)
