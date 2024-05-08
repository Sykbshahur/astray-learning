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
from dataset.datasets_train_test import *
import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import multiprocessing
import pickle
import numpy as np
from efficientnet_pytorch import EfficientNet


torch.cuda.set_device(1)
print()

def eff_pretrained_model(numClasses):
    model = EfficientNet.from_pretrained('efficientnet-b3')
    model._fc = nn.Linear(1536, numClasses)
    return model


def dspfwa(numClasses):
    model = SPPNet(backbone=50, num_class=numClasses, pretrained=True)
    model.classifier = nn.Linear(83968, numClasses)
    return model


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str,
                        default="/home/ubuntu/shahur/celebdf/test.csv")
    parser.add_argument("--input_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int,
                        default=256, help="size of the batches")
    parser.add_argument("--num_out_classes", type=int, default=1)
    parser.add_argument("--checkpoints", type=str,
                        default="D:/RA_work/Fairness_YanJu_Code/4012Code/training/checkpoints/ucf_oneStage/ff++_ucf_oneStage_sam0.5_full/lamda1_0.1_lamda2_0.01_lr0.0005/ucf_oneStage2.pth")
    parser.add_argument("--inter_attribute", type=str,
                        default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others')
    parser.add_argument("--single_attribute", type=str,
                        default='male-nonmale-asian-white-black-others')
    parser.add_argument("--test_data_name", type=str,
                        default='celebdf')
    parser.add_argument("--savepath", type=str,
                        default='/home/ubuntu/shahur/Final_Misleading/checkpoints/XceptionMis_Newthird_final_effic/ff++_XceptionMis_Newthird_final_M2/result/out_res')
    parser.add_argument("--model_structure", type=str, default='XceptionMis_Newthird_final_effic',
                        help="efficient,ucf_daw")

    opt = parser.parse_args()
    print(opt, '!!!!!!!!!!!')

    cuda = True if torch.cuda.is_available() else False

    # prepare the model (detector)
    if opt.model_structure == 'XceptionMis_Newthird_final_effic':
        model_class = DETECTOR['XceptionMis_Newthird_final_effic']
    
        # print(model_class)


    if opt.model_structure == 'XceptionMis_Newthird_final_effic':
        from transform import xception_default256_data_transforms as data_transforms
        model = model_class()
    if cuda:
        # model.cuda()
        model.to(device)

    ckpt = torch.load(opt.checkpoints,map_location=device)
    model.load_state_dict(ckpt, strict=True)
    print('loading from: ', opt.checkpoints)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    interattributes = opt.inter_attribute.split('-')
    singleattributes = opt.single_attribute.split('-')

    nonmale_dst = []
    male_dst = []
    black_dst = []
    white_dst = []
    others_dst = []
    asian_dst = []

    for eachatt in interattributes:
        print(opt.test_path)
        test_dataset = ImageDataset_Test(
            opt.test_path, eachatt, data_transforms['test'], opt.test_data_name)
        if 'nonmale,' in eachatt:
            nonmale_dst.append(test_dataset)
        else:
            male_dst.append(test_dataset)
        if ',black' in eachatt:
            black_dst.append(test_dataset)
        if ',white' in eachatt:
            white_dst.append(test_dataset)
        if ',others' in eachatt:
            others_dst.append(test_dataset)
        if ',asian' in eachatt:
            asian_dst.append(test_dataset)

        test_dataloader = DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=False)

        print("%s" % opt.test_path)
        print('Testing: ', eachatt)
        print('-' * 10)
        print('%d batches int total' % len(test_dataloader))

        corrects = 0.0
        predict = {}
        start_time = time.time()

        pred_list = []
        label_list = []
        face_list = []
        feature_list = []

        for i, data_dict in enumerate(test_dataloader):
            bSTime = time.time()
            model.eval()
            data, label = data_dict['image'], data_dict["label"]
            data_dict['image'], data_dict["label"] = data.to(
                device), label.to(device)

            with torch.no_grad():

                output = model(data_dict, inference=True)
                pred = output['cls'][:, 1]

                # print(pred.shape, '222222222')
                # print(pred)
                pred = pred.cpu().data.numpy().tolist()
                # print(pred)

                simp_label = label
                pred_list += pred
                label_list += label.cpu().data.numpy().tolist()

            bETime = time.time()
            print('#{} batch finished, eclipse time: {}'.format(i, bETime-bSTime))

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)

        # savepath = opt.savepath + '/' + opt.test_mode + '/' + eachatt

        # os.makedirs(opt.savepath + '/' + opt.test_mode + '/', exist_ok=True)
        savepath = opt.savepath + '/' + eachatt
        # os.makedirs(savepath, exist_ok=True)

        np.save(savepath+'labels.npy', label_list)
        np.save(savepath+'predictions.npy', pred_list)

    print()
    print('-' * 10)
