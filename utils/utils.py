import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn import metrics

def cont_grad(x, rate=1):
    return rate * x + (1 - rate) * x.detach()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def get_test_metrics(y_pred:np.ndarray, y_true:np.ndarray, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
            # 分割字符串，获取'a'和'b'的值
            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            # 如果'a'的值还没有在字典中，添加一个新的键值对
            if a not in result_dict:
                result_dict[a] = []

            # 将'b'的值添加到'a'的列表中
            result_dict[a].append(item)
        image_arr = list(result_dict.values())
        # 将字典的值转换为一个列表，得到二维数组

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if img_names is not None and type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc = auc
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    res = EasyDict({'ACC': acc, 'AUC': auc, 'EER': eer, 'AP': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true})
    return res

def find_best_threshold(y_trues, y_preds):
    '''
        This function is utilized to find the threshold corresponding to the best ACER
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
    '''
    print("Finding best threshold...")
    best_thre = 0.5
    best_metrics = None
    candidate_thres = list(np.unique(np.sort(y_preds)))
    for thre in candidate_thres:
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None:
            best_metrics = metrics
            best_thre = thre
        elif metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    print(f"Best threshold is {best_thre}")
    return best_thre, best_metrics


def cal_metrics(y_trues, y_preds, threshold=0.5):
    '''
        This function is utilized to calculate the performance of the methods
        Args:
            y_trues (list): the list of the ground-truth labels, which contains the int data
            y_preds (list): the list of the predicted results, which contains the float data
            threshold (float, optional):
                'best': calculate the best results
                'auto': calculate the results corresponding to the thresholds of EER
                float: calculate the results of the specific thresholds
    '''

    metrics = EasyDict()

    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)

    metrics.EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))

    if threshold == 'best':
        _, best_metrics = find_best_threshold(y_trues, y_preds)
        return best_metrics

    elif threshold == 'auto':
        threshold = metrics.Thre
        # print('Auto threshold is:',threshold)

    prediction = (np.array(y_preds) > threshold).astype(int)

    res = confusion_matrix(y_trues, prediction, labels=[0, 1])
    TP, FN = res[0, :]
    FP, TN = res[1, :]
    metrics.ACC = (TP + TN) / len(y_trues)

    TP_rate = float(TP / (TP + FN))
    TN_rate = float(TN / (TN + FP))

    metrics.APCER = float(FP / (TN + FP))
    metrics.BPCER = float(FN / (FN + TP))
    metrics.ACER = (metrics.APCER + metrics.BPCER) / 2

    return metrics


