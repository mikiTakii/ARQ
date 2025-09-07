import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

from nnverify.common.dataset import Dataset

from torch.utils.data import DataLoader
from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from progress.bar import Bar
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import torch

import os


from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, get_dataset_full
from torch.optim import SGD, Optimizer
from time import time
import time as time1
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

class RSCache():
    def __init__(self):
        self.pA_arr = []
        self.radii = []
        self.seeds = []
        self.predictions = []
        self.chi_arr = []
        self.cAHat_arr = []
        self.time_arr = []

        self.correct = []
        

    def add(self, pABar, radius, seed, predictions, chi, cAHat, correct):
        self.pA_arr.append(pABar)
        self.radii.append(radius)
        self.seeds.append(seed)
        self.predictions.append(predictions)
        self.chi_arr.append(chi)
        self.cAHat_arr.append(cAHat)

        self.correct.append(correct)

    def add_time_elapsed(self, time_elapsed):
        self.time_arr.append(time_elapsed)
        


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, dataset, num_classes: int, sigma: float, device=None, dataset_name = Dataset.CIFAR10):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print("Initiating smooth network on device: ", self.device) 
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.dataset = dataset
        self.rs_cache = RSCache()
        self.second = False
        self.dataset_name = dataset_name

    
    
    def certify(self, count: int, n: int, alpha: float, batch_size: int, evaluate, quantizable_idx, centroid_label_dict, kmeans, data_root):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.

        """
        n0 = n//100
        if (count < 0 or count > len(self.dataset)):
            count = len(self.dataset)

        #####Finetune part########
        if quantizable_idx != None:
            if evaluate == True:
                train_dataset = get_dataset_full(self.dataset_name, 'train', data_root)
                dataset1 = get_dataset_full(self.dataset_name, 'test', data_root)
                test_loader = DataLoader(dataset1, batch_size=128, shuffle=False)
                pin_memory = True     #false for cifar
                train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128,
                                    num_workers=16, pin_memory=pin_memory)

                criterion = CrossEntropyLoss().cuda()
                if self.dataset_name == Dataset.CIFAR10:
                    optimizer = SGD(self.base_classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
                    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                    total_epochs = 30

                    for epoch in range(total_epochs):                
                        print('Epoch:', epoch, 'LR:', optimizer.param_groups[0]['lr'])
                        before = time()
                        train_loss, train_acc = self.train(train_loader, self.base_classifier, criterion, optimizer, epoch, self.sigma, quantizable_idx, centroid_label_dict, kmeans)
                        after = time()
                        
                        val_loss = self.validate(test_loader, self.base_classifier, self.sigma)
                        scheduler.step()
                    
                else:
                    optimizer = SGD(self.base_classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.01, threshold_mode='rel')
                    total_epochs = 10

                    for epoch in range(total_epochs):
                    
                        print('Epoch:', epoch, 'LR:', optimizer.param_groups[0]['lr'])
                        before = time()
                        train_loss, train_acc = self.train(train_loader, self.base_classifier, criterion, optimizer, epoch, self.sigma, quantizable_idx, centroid_label_dict, kmeans)
                        after = time()
                        val_loss = self.validate(test_loader, self.base_classifier, self.sigma)
                        scheduler.step(val_loss)


        self.base_classifier.eval()

        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)

        for i in range(len(self.dataset)):
            if i == count:
                break

            (x, label) = self.dataset[i]
            x = x.to(self.device)

            before_time = time()
            cAHat, radius, correct = self.compute_radius(x, n, n0, batch_size, alpha, label)

            time_elapsed = datetime.timedelta(seconds=(time() - before_time))
            self.rs_cache.add_time_elapsed(time_elapsed.total_seconds())
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, cAHat, radius, correct, str(time_elapsed)), flush=True)
    
  
    
    def compute_radius(self, x, n, n0, batch_size, alpha, label):
        # draw samples of f(x+ epsilon)
        counts_selection, _, _ = self._sample_noise(x, n0, batch_size, self.sigma)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()

        correct = int(cAHat == label)
        if (correct != 1):
            self.rs_cache.add(0.0, 0.0, None, None, 0.0, cAHat, 0.0)
            return cAHat, 0.0, correct
        # draw more samples of f(x + epsilon)
        counts_estimation, seeds, predictions = self._sample_noise(x, n, batch_size, self.sigma)     #modify
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        print('pA: ', pABar)
        
        
        if pABar < 0.5:
            cAHat = Smooth.ABSTAIN
            radius = 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
        
        self.rs_cache.add(pABar, radius, seeds, predictions, None, cAHat, 1.0)

        return cAHat, radius, correct

    def compute_radius_with_chi(self, x, n_chi, batch_size, alpha_chi, label, seeds, predictions, original_pA):

        #estimate chi with n_chi samples
        chi, counts_selection = self._estimate_chi(x, n_chi, batch_size, alpha_chi, seeds, predictions)
        print('chi: ', chi)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()

        correct = int(cAHat == label)
        print('pA: ', original_pA - chi)
        if (original_pA - chi) < 0.5:
            cAHat = Smooth.ABSTAIN
            radius = 0.0
        else:
            radius = self.sigma/2 * (norm.ppf(original_pA - chi) - norm.ppf(1 - original_pA + chi)) 
        
        self.rs_cache.add(original_pA - chi, radius, None, None, chi, cAHat, correct)
        return cAHat, radius, correct

    def certify_with_cache(self, count: int, n2: int, n_chi: int, alpha1, alpha_chi, batch_size: int, original_rs_cache: np.ndarray, gamma: int, quantizable_idx, centroid_label_dict, kmeans):
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
        n0 = n2//10
        if (count < 0 or count > len(self.dataset)):
            count = len(self.dataset)

        self.base_classifier.eval()
       

        for i in range(len(self.dataset)):
            if i == count:
                break

            (x, label) = self.dataset[i]
            x = x.to(self.device)
            
            original_cAHat = original_rs_cache.cAHat_arr[i]
            original_correct = int(original_cAHat == label)
            original_pA = original_rs_cache.pA_arr[i]
            before_time = time()
            if (original_correct != 1):
                time_elapsed = datetime.timedelta(seconds=(time() - before_time))
                self.rs_cache.add(original_pA, 0.0, None, None, 0.0, original_cAHat, 0)
                self.rs_cache.add_time_elapsed(time_elapsed.total_seconds())
                print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, original_cAHat,0.0, original_correct, str(time_elapsed)), flush=True)
                continue
            
            
            if original_pA < gamma:
                predictions = original_rs_cache.predictions[i]
                seeds = original_rs_cache.seeds[i]
                cAHat, radius, correct = self.compute_radius_with_chi(x, n_chi, batch_size, alpha_chi, label, seeds, predictions, original_pA)
            else:
                cAHat, radius, correct = self.compute_radius(x, n2, n0, batch_size, alpha1+alpha_chi, label)

            time_elapsed = datetime.timedelta(seconds=(time() - before_time))
            self.rs_cache.add_time_elapsed(time_elapsed.total_seconds())

            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, cAHat, radius, correct, str(time_elapsed)), flush=True)
            

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts, _, _ = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size: int, sigma: float) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        
        with torch.no_grad():
            counts = torch.zeros(self.num_classes, dtype=torch.int32, device=self.device)  # Keep counts on GPU
            seeds = []
            preds = []

            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                seeds.append(torch.seed())
                noise = torch.randn(batch.shape, device=self.device) * sigma
                predictions = self.base_classifier(batch + noise).argmax(1)

                preds.append(predictions)
                counts += torch.bincount(predictions, minlength=self.num_classes)  # Use bincount to keep on GPU

            counts = counts.cpu().numpy()  # Move final counts to CPU at the end
            return counts, seeds, preds
            
    def _estimate_chi(self, x: torch.tensor, num: int, batch_size: int, alpha_chi: float, seeds, predictions):
        with torch.no_grad():
            different = 0
            total = num
            counts = torch.zeros(self.num_classes, dtype=torch.int32, device=self.device)  # Keep counts on GPU

            # Precompute the repeated batch outside the loop if x does not change
            repeated_batches = [x.repeat((min(batch_size, num - i * batch_size), 1, 1, 1)) for i in range(ceil(num / batch_size))]

            for i, batch in enumerate(repeated_batches):
                torch.manual_seed(seeds[i])
                noise = torch.randn_like(batch) * self.sigma  # Use torch.randn_like to match batch shape directly

                # Compute perturbed predictions
                perturbed_predictions = self.base_classifier(batch + noise).argmax(1)

                # Calculate differences using PyTorch operations to avoid moving data to CPU
                different += torch.sum(predictions[i] != perturbed_predictions).item()  # Only convert the final result to CPU

                # Update counts using PyTorch's bincount
                counts += torch.bincount(perturbed_predictions, minlength=self.num_classes)

            # Convert counts to numpy array only at the end if necessary
            counts = counts.cpu().numpy()
            return self._upper_confidence_bound(different, total, alpha_chi), counts


        
    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float):
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]

    def train(self, loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, quantizable_idx, centroid_label_dict, kmeans):
        scaler = GradScaler() 
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time()

        # switch to train mode
        model.train()
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time() - end)

            optimizer.zero_grad()

            inputs = inputs.cuda()
            targets = targets.cuda()
            
            with autocast():
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            if kmeans == True:
                kmeans_update_model(model, quantizable_idx, centroid_label_dict, free_high_bit=False)

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if i % 200 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)
    
    def validate(self, val_loader, model, noise_sd, verbose=False):
        scaler = GradScaler() 
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t1 = time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time()
            bar = Bar('valid:', max=len(val_loader))
            for i, (inputs, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time() - end)

                inputs, targets = inputs.cuda(), targets.cuda()

                # ##########original###########
                with autocast():
                    inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

                # compute output
                    output = model(inputs)
                    criterion = nn.CrossEntropyLoss().cuda()
                    loss = criterion(output, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()
                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()
        t2 = time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return losses.avg
