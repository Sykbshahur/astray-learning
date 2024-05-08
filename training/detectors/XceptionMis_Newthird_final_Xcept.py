import math
import torch.nn as nn
import logging
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from sklearn import metrics
from metrics.base_metrics_class import calculate_metrics_for_train
import numpy as np
from functools import reduce
from detectors import DETECTOR
from loss import LOSSFUNC
from networks import BACKBONE
from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='XceptionMis_Newthird_final_Xcept')
class XceptionMis_Newthird_final_Xcept(AbstractDetector):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super().__init__()

        self.backbone_xce = self.build_backbone()
        self.backbone_srm = self.build_backbone()

        self.loss_func = self.build_loss()

        self.srm_conv0 = SRMConv2d_simple(inc=3)

        self.sa = SimpleChannelAttention(in_chan=2, out_chan=1)

        # self.fuse1 = FeatureFusionModule(in_chan=728*2, out_chan=728)
        # self.fuse2 = FeatureFusionModule(in_chan=728*2, out_chan=728)
        # self.fuse3 = FeatureFusionModule(in_chan=1024*2, out_chan=1024)

        self.encoder_feat_dim = 4096
        self.half_fingerprint_dim = self.encoder_feat_dim//2
        self.specific_task_number = 6
        self.num_classes = 2

        self.finalblock_spe = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )
        self.finalblock_sha = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )
        self.finalhead_spe = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=self.specific_task_number
        )
        self.finalhead_sha = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=self.num_classes
        )
        self.loss_func = self.build_loss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    

    def build_backbone(self):
        # prepare the backbone
        backbone_class = BACKBONE['xception']
        backbone = backbone_class({'mode': 'original',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})
        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load(
            '/home/ubuntu/shahur/fairness_gen/pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone

    def build_loss(self):
        # prepare the loss function
        cls_loss_class = LOSSFUNC['cross_entropy']
        spe_loss_class = LOSSFUNC['cross_entropy']
        con_loss_class = LOSSFUNC['contrastive_regularization_dual']
        con_loss_func = con_loss_class(margin=3.0)
        cls_loss_func = cls_loss_class()
        spe_loss_func = spe_loss_class()
        loss_func = {
            'cls': cls_loss_func,
            'spe': spe_loss_func,
            'con': con_loss_func}
        return loss_func

    def texture_encoder(self, img) -> torch.tensor:
        texture = self.srm_conv0(img)

        texture = self.backbone_srm.fea_part1_0(texture) 
        texture = self.backbone_srm.fea_part1_1(texture) 
        texture = self.backbone_srm.fea_part2(texture)
        texture = self.backbone_srm.fea_part3(texture)
        texture = self.backbone_srm.fea_part4(texture)
        texture = self.backbone_srm.fea_part5(texture)
        return texture

    def features(self, img) -> torch.tensor:
        
        f = self.backbone_xce.features(img)
        return f
        

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:

    
        label = data_dict['label']
        label_spe = data_dict['label_spe']

        pred = pred_dict['cls']
        pred_spe = pred_dict['cls_spe']
       
        common_features = pred_dict['feat']
        specific_features = pred_dict['feat_spe']


        loss_sha = self.loss_func['cls'](pred, label)
        # 2. spe loss
        loss_spe = self.loss_func['spe'](pred_spe, label_spe)
        # 3. constrative loss
        loss_con = self.loss_func['con'](
            common_features, specific_features, label_spe)
        
        loss = loss_sha + 0.1* loss_spe + 0.05* loss_con      
        loss_dict = {'overall': loss,
                     'loss_sha': loss_sha,
                     'loss_spe': loss_spe,
                     'loss_con': loss_con}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy
        label = data_dict['label']
        label_spe = data_dict['label_spe']
        pred = pred_dict['cls']
        pred_spe = pred_dict['cls_spe']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'acc_spe': acc_spe, 'eer':eer,'ap':ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true,y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc':acc, 'auc':auc, 'eer':eer, 'ap':ap, 'pred':y_pred, 'label':y_true}

    def classifier(self,f_features, texture) -> torch.tensor:
        fusion_features = torch.concat((f_features, texture), dim = 1)
        f_spe = self.finalblock_spe(fusion_features)
        f_share = self.finalblock_sha(fusion_features)
        return f_spe, f_share
            

    def forward(self, data_dict: dict, inference=False) -> dict:
        img = data_dict['image']
        # get the features by backbone
        texture = self.texture_encoder(img)
        f_features = self.features(img)

        f_spe, f_share = self.classifier(f_features, texture)
   

        out_spe, spe_feat = self.finalhead_spe(f_spe)
        out_sha, sha_feat = self.finalhead_sha(f_share)

        # get the probability of the pred
        prob_sha = torch.softmax(out_sha, dim=1)[:, 1]
        prob_spe = torch.softmax(out_spe, dim=1)[:, 1]

        # build the prediction dict for each output
        pred_dict = {'cls': out_sha, 
                     'prob': prob_sha,
                     'feat': sha_feat,
                     'cls_spe': out_spe,
                     'prob_spe': prob_spe,
                     'feat_spe': spe_feat}
        if inference:
            self.prob.append(
                pred_dict['prob']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(out_sha, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)
        return pred_dict

# =========================================srm_conv================================================================
class SRMConv2d_simple(nn.Module):
    
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],#, filter1, filter1],
                   [filter2],#, filter2, filter2],
                   [filter3]]#, filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters

class SimpleChannelAttention(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SimpleChannelAttention, self).__init__()

        self.conv = nn.Conv2d(in_chan, out_chan, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048*2, out_chan=2048):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.sc = SimpleChannelAttention(2, 1)
        self.init_weight()

    def forward(self, srm, rgb):

        Attsrm = self.sc(srm)

        fuse = self.convblk(torch.cat((rgb, srm + srm * Attsrm), dim = 1)) 

        return fuse

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


