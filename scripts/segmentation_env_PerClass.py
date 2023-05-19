
import sys
import random
sys.path.append(".")
from guided_diffusion.utils import staple

import numpy
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import math
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.utils import staple
import argparse

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz
from collections import OrderedDict
from prettytable import PrettyTable
#from mmcv.utils import print_log
# from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
# from mmseg.utils import get_root_logger

def eval(pre_eval_results):
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4
    total_area_intersect = sum(pre_eval_results[0]) # total_area_intersect.shape = (num_classes, )
    total_area_union = sum(pre_eval_results[1])     # total_area_union.shape = (num_classes, )
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label,)
    
    return ret_metrics


def total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label,nan_to_num=None,
                          beta=1):
    
    

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})

    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label
    dice = 2 * total_area_intersect / (
        total_area_pred_label + total_area_label)
    ret_metrics['IoU'] = iou
    #ret_metrics['Acc'] = acc
    #ret_metrics['Dice'] = dice

    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
    ret_metrics['Fscore'] = f_value
    ret_metrics['Precision'] = precision
    ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }

    return ret_metrics 


def pre_eval(pred, seg_map):
    pre_eval_results = []
    pre_eval_results.append(
        intersect_and_union(
            pred,
            seg_map,
            2))
    return pre_eval_results


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ):

    mask = (label != 255)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def f_score(precision, recall, beta=1):
    
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--inp_pth", default='')
    argParser.add_argument("--out_pth", default='')
    args = argParser.parse_args()
    mix_res = (0,0)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth
    results = []
    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:              # 4.30 files=['area14_0_1792_256_2048.jpg',....]
            #if 'ens' in name: 4.23 注释
            num += 1
            ind = name.split('_')[0]    # 4.30 ind 'area14'
            pred = Image.open(os.path.join(root, name)).convert('L')
            gt_name = os.path.splitext(name)[0] + ".tif"
            #gt_name = "beijing_" + ind + ".tif"
            gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
            pred = torchvision.transforms.PILToTensor()(pred)
            pred = torch.unsqueeze(pred,0).float() 
            pred = pred / pred.max()
            # if args.debug:
            #     print('pred max is', pred.max())
            #     vutils.save_image(pred, fp = os.path.join('./results/' + str(ind)+'pred.jpg'), nrow = 1, padding = 10)
            gt = torchvision.transforms.PILToTensor()(gt)
            gt = torchvision.transforms.Resize((256,256))(gt)
            gt = torch.unsqueeze(gt,0).float() / 255.0
            # if args.debug:
            #     vutils.save_image(gt, fp = os.path.join('./results/' + str(ind)+'gt.jpg'), nrow = 1, padding = 10)
            #temp = eval_seg(pred, gt)
            result = pre_eval(pred, gt)
            results.extend(result)

    ret_metrics = eval(results)
    class_names = ('class1', 'class2')

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print('per class results:')
    print('\n' + class_table_data.get_string())
    print('Summary:')
    print('\n' + summary_table_data.get_string())   
    

if __name__ == "__main__":
    main()
