import os
from time import time
import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from progress.bar import Bar

from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from lib.utils.data_utils import get_split_train_dataset
from lib.utils.quantize_utils import QConv2d, QLinear, calibrate


from nnverify.smoothing.code.incremental_rs import *
import nnverify.parse as parse
from nnverify.smoothing.code.datasets import get_normalize_layer

from torch.cuda.amp import GradScaler, autocast


import cProfile
import pstats


class LinearQuantizeEnv:
    def __init__(self, model, pretrained_model, data, data_root, compress_ratio, args, n_data_worker=16,
                 batch_size=256, float_bit=8, is_model_pruned=False, evaluate=False, sigma=0.5, n1=500, n2=1000, policy=[]):
        # default setting
        self.quantizable_layer_types = [QConv2d, QLinear]

        # save options
        self.model = model
        self.sigma = sigma
        self.n1 = n1
        self.n2 = n2
        self.evaluate = evaluate

        self.model_for_measure = deepcopy(model)
        self.model_name = args.arch
        self.cur_ind = 0
        if self.evaluate:
            self.policy = policy
        else:
            self.policy = []  # quantization strategy

        self.finetune_lr = args.finetune_lr
        self.optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=1e-5)
        #self.criterion = nn.CrossEntropyLoss().cuda()

        self.device = next(model.parameters()).device
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.pretrained_model = pretrained_model
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.data_root = data_root
        self.compress_ratio = compress_ratio
        self.is_model_pruned = is_model_pruned
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.finetune_gamma = args.finetune_gamma
        self.finetune_lr = args.finetune_lr
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch

        # options from args
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.float_bit = float_bit * 1.
        self.last_weight_action = self.max_bit
        self.last_activation_action = self.max_bit
        self.action_radio_button = True

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('imagenet' in data)
        self.use_top5 = args.use_top5

        self.acc = 0
        self.rs_cache = None
        self.original_radii = None
        self.original_pA = None
        self.original_time = None

        # init reward
        self.best_reward = -math.inf

        self.best_strategy = []
        self.best_certified_acc = 0
        self.best_ACR = 0
        self.best_acc = 0
        self.best_val_acc = 0
        self.best_cost_ratio = 0
        self.counter = 0

        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)



        model = self.model
        model_quantized = None
        net = None
        dataset = self.data_type
        origin = True
        if self.evaluate:
            N = 100000
        else:
            N = 10000

        self.org_certified_acc, self.org_acc, self.org_ACR= self.nnverify(model, net, dataset, origin, N, self.sigma, self.sigma, None, None, self.n1, self.n2)


        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # mode
        self.cost_mode = 'Bit#s'
        self.simulator_batch = 16
        self.cost_lookuptable = self._get_lookuptable()

        # sanity check #MARK
        assert self.compress_ratio > self._min_cost() / self._org_cost(), \
            'Error! You can make achieve compress_ratio smaller than min_bit!'

        # restore weight
        self.reset()
        print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_certified_acc,
                                                                                        self.train_size, self.val_size))
        print('=> original cost: {:.4f}'.format(self._org_cost()/1e9))

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        action = self._action_wall(action)  # percentage to preserve

        if self.action_radio_button:
            self.last_weight_action = action
        else:
            self.last_activation_action = action
            self.strategy.append([self.last_weight_action, self.last_activation_action])  # save action to strategy

        # all the actions are made
        if self._is_final_layer() and (not self.action_radio_button):
            if self.evaluate == True:
                # policy here
                self.strategy = self.policy
            self._final_action_wall()
            assert len(self.strategy) == len(self.quantizable_idx)
            cost = self._cur_cost()
            cost_ratio = cost / self._org_cost()


            self._set_mixed_precision(quantizable_idx=self.quantizable_idx, strategy=self.strategy)


            start_time = time()
            self.model = calibrate(self.model, self.train_loader)
            end_time = time()

            elapsed_time = end_time - start_time
            print(f"Calibration took {elapsed_time} seconds.")


            if self.finetune_flag and not self.evaluate:
                acc = self._finetune(self.train_loader, self.model, epochs=self.finetune_epoch, verbose=True)
            

            model_quantized = self.model
            net = None
            dataset = self.data_type
            N = 100000
            if self.evaluate:                
                origin = True
            else:
                origin = False

            centroid_label_dict = None

           
            start_time = time()

            certified_acc, acc, ACR = self.nnverify(model_quantized, net, dataset, origin, N, self.sigma, self.sigma, self.quantizable_idx, centroid_label_dict, self.n1, self.n2)
            
            end_time = time()

            elapsed_time = end_time - start_time
            print(f"Certification took {elapsed_time} seconds.")
            
            reward = self.reward(certified_acc, ACR)
            print('reward value', reward)

            info_set = {'cost_ratio': cost_ratio, 'accuracy': certified_acc, 'cost': cost}
            print('Current best policy: {}, reward: {:.3f}, ACR: {:.3f}, acc: {:.3f}, cost_ratio: {:.3f}'.format(
                    self.best_strategy, self.best_reward, self.best_ACR, self.best_acc,  self.best_cost_ratio))
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_strategy = self.strategy
                self.best_ACR = ACR
                self.best_acc = acc
                self.best_cost_ratio = cost_ratio
                prGreen('New best policy: {}, reward: {:.3f}, ACR: {:.3f}, acc: {:.3f}, cost_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, ACR, acc, cost_ratio))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            self.action_radio_button = not self.action_radio_button
            return obs, reward, done, info_set

        cost = self._cur_cost()
        info_set = {'cost': cost}
        reward = 0
        done = False

        if self.action_radio_button:
            self.layer_embedding[self.cur_ind][-1] = 0.0
        else:
            self.cur_ind += 1  # the index of next layer
            self.layer_embedding[self.cur_ind][-1] = 1.0
        self.layer_embedding[self.cur_ind][-2] = float(action) / float(self.max_bit)
        self.layer_embedding[self.cur_ind][-1] = float(self.action_radio_button)
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        self.action_radio_button = not self.action_radio_button
        return obs, reward, done, info_set

    # for quantization
    def reward(self, certified_acc, ACR):
        return (ACR-self.org_ACR)

    def reset(self):
        # restore env by loading the pretrained model
        self.model.load_state_dict(self.pretrained_model, strict=False)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1

    def _final_action_wall(self):
        target = self.compress_ratio * self._org_cost()
        min_cost = 0
       
        for i, n_bit in enumerate(self.strategy):
            min_cost += self.cost_lookuptable[i] * int(self.min_bit) * int(self.min_bit)

        print('before action_wall: ', self.strategy, min_cost/1e9, self._cur_cost()/1e9)
        while min_cost < self._cur_cost() and target < self._cur_cost():
            # print('current: ', self.strategy, min_cost, self._cur_cost())
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit[1] > self.min_bit:
                    self.strategy[-(i+1)][1] -= 1
                self._keep_first_last_layer()
                if target >= self._cur_cost():
                    break
                if n_bit[0] > self.min_bit:
                    self.strategy[-(i+1)][0] -= 1
                self._keep_first_last_layer()
                if target >= self._cur_cost():
                    break
        print('after action_wall: ', self.strategy, min_cost/1e9, self._cur_cost()/1e9)  #GBitOPs

    def _keep_first_last_layer(self):
        self.strategy[0][0] = 8
        self.strategy[0][1] = -1
        self.strategy[-1][0] = 8
        self.strategy[-1][1] = 8

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        #action = float(action)
        action = float(np.asarray(action).reshape(-1)[0])
        min_bit, max_bit = self.bound_list[self.cur_ind]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
        return action  # not constrained here

    def _set_mixed_precision(self, quantizable_idx, strategy):
        assert len(quantizable_idx) == len(strategy), \
            'You should provide the same number of bit setting as layer list for weight quantization!'
        quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_idx, strategy)}
        for i, layer in enumerate(self.model.modules()):
            if i not in quantizable_idx:
                continue
            else:
                layer.w_bit = quantize_layer_bit_dict[i][0]
                layer.a_bit = quantize_layer_bit_dict[i][1]

    def _cur_cost(self):
        cur_cost = 0.
        for i, n_bit in enumerate(self.strategy):
            cur_cost += self.cost_lookuptable[i] * n_bit[0] * n_bit[1]      #MARK
        return cur_cost

    def _org_cost(self):
        org_cost = 0
        for i in range(len(self.cost_lookuptable)):
            org_cost += self.cost_lookuptable[i] * 32 * 32
        return org_cost

    def _min_cost(self):
        min_cost = 0
        for i in range(len(self.cost_lookuptable)):
            if i == 0 or i == (len(self.cost_lookuptable) - 1):
                min_cost += self.cost_lookuptable[i]*8*8
            else:
                min_cost += self.cost_lookuptable[i] * int(self.min_bit) * int(self.min_bit)
        return min_cost

    def _init_data(self):
        self.train_loader, self.val_loader, n_class, self.small_valset, self.small_testset = get_split_train_dataset(
            self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
            val_size=self.val_size, train_size=self.train_size, for_inception=self.is_inception)
        if self.evaluate:
            self.small_valset = self.small_testset

    def _build_index(self):
        self.quantizable_idx = []
        self.bound_list = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                self.quantizable_idx.append(i)
                self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final bound list: {}'.format(self.bound_list))



    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        if self.is_imagenet:
            measure_model(self.model_for_measure, 224, 224)
        else:
            measure_model(self.model_for_measure, 32, 32)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d or type(m) == QConv2d:
                this_state.append([int(m.in_channels == m.groups)])
                this_state.append([m.in_channels])
                this_state.append([m.out_channels])
                this_state.append([m.stride[0]])
                this_state.append([m.kernel_size[0]])
                this_state.append([np.prod(m.weight.size())])
                this_state.append([m.in_w*m.in_h])
            elif type(m) == nn.Linear or type(m) == QLinear:
                this_state.append([0.])
                this_state.append([m.in_features])
                this_state.append([m.out_features])
                this_state.append([0.])
                this_state.append([1.])
                this_state.append([np.prod(m.weight.size())])
                this_state.append([m.in_w*m.in_h])

            this_state.append([i])
            this_state.append([1.])
            this_state.append([1.])
            layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _get_lookuptable(self):
        if self.data_type == 'cifar10':
            latency_list = [442368, 2359296, 2359296, 2359296, 2359296, 2359296, 2359296, 1253376, 2359296, 139264, 2359296, 
                            2359296, 2359296, 2359296, 1327104, 2359296, 147456, 2359296, 2359296, 2359296, 2359296, 640]
        else:
            if self.model_name == 'qresnet50':
                latency_list = [119067648, 12845056, 115605504, 51380224, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 
                            51380224, 102760448, 119734272, 51380224, 106430464, 51380224, 115605504, 51380224, 51380224, 115605504, 
                            51380224, 51380224, 115605504, 51380224, 102760448, 123863040, 51380224, 110100480, 51380224, 115605504, 
                            51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224, 51380224, 115605504, 51380224, 
                            51380224, 115605504, 51380224, 102760448, 132120576, 51380224, 117440512, 51380224, 115605504, 51380224, 
                            51380224, 115605504, 51380224, 2048000]
            elif self.model_name == 'qmobilenetv2':
                latency_list = [10934784, 3612672, 6422528, 19267584, 2757888, 7225344, 10838016, 4064256, 10838016, 10838016, 
                                1052352, 3612672, 4816896, 1354752, 4816896, 4816896, 1354752, 4816896, 4816896, 362880, 
                                2408448, 4816896, 677376, 4816896, 4816896, 677376, 4816896, 4816896, 677376, 4816896, 
                                4816896, 677376, 7225344, 10838016, 1016064, 10838016, 10838016, 1016064, 10838016, 10838016, 
                                290304, 4515840, 7526400, 423360, 7526400, 7526400, 423360, 7526400, 7526400, 423360, 
                                15052800, 20070400, 1280000]
            elif self.model_name == 'qalexnet':
                latency_list = [70276800, 223948800, 112140288, 149520384, 99680256, 37748736, 16777216, 40960]
        return latency_list.copy()

    def _finetune(self, train_loader, model, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.
        if self.data_type != 'cifar10':
            scaler = GradScaler()

        # switch to train mode
        model.train()
        end = time()
        t1 = time()
        bar = Bar('train:', max=len(train_loader))
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                #inputs, targets = inputs.cuda(), targets.cuda()
                #inputs = inputs.cuda() + torch.randn_like(inputs, device='cuda') * self.sigma

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs + torch.randn_like(inputs) * self.sigma
                # measure data loading time
                data_time.update(time() - end)
                if self.data_type != 'cifar10':
                    with autocast():
                        output = model(inputs)
                        loss = self.criterion(output, targets)

                else:
                    # compute output
                    output = model(inputs)
                    loss = self.criterion(output, targets)

                if self.data_type != 'cifar10':
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                if self.data_type == 'cifar10':
                    # compute gradient
                    self.optimizer.zero_grad()
                    loss.backward()

                    # do SGD step
                    self.optimizer.step()

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()
        t2 = time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return best_acc

    def _validate(self, val_loader, model, verbose=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if self.data_type != 'cifar10':
            scaler = GradScaler()

        t1 = time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time()
            bar = Bar('valid:', max=len(val_loader))
            for i, (inputs, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time() - end)

                #inputs, targets = inputs.cuda(), targets.cuda()
                #inputs = inputs.cuda() + torch.randn_like(inputs, device='cuda') * self.sigma

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs + torch.randn_like(inputs) * self.sigma

                # compute output                
                if self.data_type != 'cifar10':
                    with autocast():
                        output = model(inputs)
                        loss = self.criterion(output, targets)
                else:
                    output = model(inputs)
                    loss = self.criterion(output, targets)

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
        if self.use_top5:
            return top5.avg
        else:
            return top1.avg





    def nnverify(self, model, net, dataset, origin, N, sigma, radius, quantizable_idx, centroid_label_dict, n1, n2):
        print("Value of n1:", n1, "Type of n1:", type(n1))
        print("Value of n2:", n2, "Type of n2:", type(n2))


        count=n1
        


        if dataset == 'cifar10':
            gamma = 0.99
            dataset = Dataset.CIFAR10
        elif dataset == 'imagenet':
            gamma = 0.995
            dataset = Dataset.IMAGENET
        elif dataset == 'imagenet10':
            gamma = 0.995
            dataset = Dataset.IMAGENET10
        elif dataset == 'imagenet100':
            gamma = 0.995
            dataset = Dataset.IMAGENET100


        if quantizable_idx == None and self.evaluate == True:
            evaluate = False
        else:
            evaluate = self.evaluate

        if origin:
            model1=model

        
            original_args = IncrementalRSArgs(net=model1, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=n2, sigma=sigma, N=N, N2=None, N_chi=None, approx=None, radius=radius, small_valset=self.small_valset, evaluate=self.evaluate, data_root=self.data_root)
            rs_cache, original_radii, original_pA, original_time, result, original_correct, r0, r25, r5, r75, r1, r125, r15, r175, r2, r250, r3, r35, ACR = IncrementalRS(original_args).certify_original(model1, evaluate, quantizable_idx, centroid_label_dict, kmeans=False)
            self.rs_cache = rs_cache
            self.original_radii = original_radii
            self.original_pA = original_pA
            self.original_time = original_time
            self.original_correct = original_correct
            print("Certified acc list ", r0, r25, r5, r75, r1, r125, r15, r175, r2, r250, r3, r35, ACR)
            return result, r0, ACR

        else:
            model_quantized1 = model
            perturbed_base_classifier = model_quantized1
            sample = n2
            if self.evaluate:
                args = IncrementalRSArgs(net=model, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=n2, sigma=sigma, N=N, N2=sample, N_chi=sample, approx=None, radius=radius, small_valset=self.small_valset, evaluate=self.evaluate, data_root=self.data_root)
            else: 
                args = IncrementalRSArgs(net=model, dataset=dataset, count=count, alpha1=0.01, alpha_chi=0.01, batch=n2, sigma=sigma, N=N, N2=sample, N_chi=sample, approx=None, radius=radius, small_valset=self.small_valset, evaluate=self.evaluate, data_root=self.data_root)
            rs_cache = self.rs_cache
            original_radii = self.original_radii
            original_pA = self.original_pA
            original_time = self.original_time
            original_correct = self.original_correct
            res, r0, r25, r5, r75, r1, r125, r15, r175, r2, r250, r3, r35, ACR= IncrementalRS(args).certify_perturbed(rs_cache, perturbed_base_classifier, original_radii, original_pA, original_time, original_correct, quantizable_idx, centroid_label_dict, kmeans = False, gamma=gamma)
            print("Certified acc list ", r0, r25, r5, r75, r1, r125, r15, r175, r2, r250, r3, r35, ACR)
            return res, r0, ACR