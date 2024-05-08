import os
import argparse
from detectors import DETECTOR
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
from torchvision import datasets
import torchvision
from dataset.datasets_train import *
import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import multiprocessing
import pickle
import numpy as np
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score


torch.cuda.set_device(0)
print()

def eff_pretrained_model(numClasses):
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(1536, numClasses)
    return model

def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    acc = accuracy_score(label, prediction >= 0.5)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    return auc, TPR, FPR, acc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str,
                        default="/home/ubuntu/shahur/dfdc/test.csv")
    parser.add_argument("--input_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int,
                        default=256, help="size of the batches")
    parser.add_argument("--num_out_classes", type=int, default=1)
    parser.add_argument("--checkpoints", type=str,
                        default="/home/ubuntu/shahur/Misleading/checkpoints/old/XceptionMis_Newthird_nofuse/ff++_XceptionMis_Newthird_nofuse_allfrozen_seed(3407)/lamda1_0.1_lamda2_0.01_lr0.001/XceptionMis_Newthird_nofuse0.pth")
    parser.add_argument("--test_data_name", type=str,
                        default='dfdc')
    parser.add_argument("--model_structure", type=str, default='XceptionMis_Newthird_final',
                        help="efficient,ucf_daw")

    opt = parser.parse_args()
    print(opt, '!!!!!!!!!!!')

    cuda = True if torch.cuda.is_available() else False

    # prepare the model (detector)
    if opt.model_structure == 'XceptionMis_Newthird_final':
        model_class = DETECTOR['XceptionMis_Newthird_final']
        # print(model_class)

    if opt.model_structure == 'XceptionMis_Newthird_final':
        from transform import xception_default_data_transforms as data_transforms
        model = model_class()
    if cuda:
        # model.cuda()
        model.to(device)

    ckpt = torch.load(opt.checkpoints,map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print('loading from: ', opt.checkpoints)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    test_dataset = ImageDataset_Test(
        opt.test_path, data_transforms['test'])
       

    test_dataloader = DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=False)

    print("%s" % opt.test_path)
    print('-' * 10)
    print('%d batches int total' % len(test_dataloader))

    corrects = 0.0

    pred_label_list = []
    pred_probs_list = []
    label_list = []
    total = 0
    running_corrects = 0

    for i, data_dict in enumerate(test_dataloader):
        bSTime = time.time()
        model.eval()
        data, label = data_dict['image'], data_dict["label"]
        data_dict['image'], data_dict["label"] = data.to(
            device), label.to(device)
        data = data.to(device)
        labels = torch.where(data_dict['label'] != 0, 1, 0)
        with torch.no_grad():

            # preds = model(data_dict = data_dict, rgb_img = data, srm_img = data,flag = 0, inference=True)
            preds = model(data_dict = data_dict,inference=True)

            _, preds_label = torch.max(preds['cls'], 1)
            pred_probs = torch.softmax(
                        preds['cls'], dim=1)[:, 1]
            total += data_dict['label'].size(0)
            running_corrects += (preds_label == data_dict['label']).sum().item()
            
            r = (preds_label == data_dict['label']).sum().item()
            print(f'-------------------------------:{r}')

            preds_label = preds_label.cpu().data.numpy().tolist()
            pred_probs = pred_probs.cpu().data.numpy().tolist()
                # print(pred)
        pred_label_list += preds_label
        pred_probs_list += pred_probs
        label_list += label.cpu().data.numpy().tolist()
        # if i % 50 == 0:
        #     batch_metrics = model.get_test_metrics()

        # print('#{} batch_metric{{"acc": {}, "auc": {}, "eer": {}, "ap": {}}}'.format(i,
        #                                                                             batch_metrics['acc'],
        #                                                                             batch_metrics['auc'],
        #                                                                             batch_metrics['eer'],
        #                                                                             batch_metrics['ap']))
                                                                                            
          
        bETime = time.time()
        print('#{} batch finished, eclipse time: {}'.format(i, bETime-bSTime))
    pred_label_list = np.array(pred_label_list)

    pred_probs_list = np.array(pred_probs_list)
    label_list = np.array(label_list)

    epoch_acc = running_corrects / total

            
    auc, TPR, FPR, _ = classification_metrics(
                label_list, pred_probs_list)

    print('Acc: {:.4f}  auc: {}, tpr: {}, fpr: {}'.format(
                 epoch_acc, auc, TPR, FPR))
       
    print()
    print('-' * 10)