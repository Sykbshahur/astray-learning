import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


def swap_spe_features(type_list, value_list):
    type_list = type_list.cpu().numpy().tolist()
    # get index
    index_list = list(range(len(type_list)))

    # init a dict, where its key is the type and value is the index
    spe_dict = defaultdict(list)

    # do for-loop to get spe dict
    for i, one_type in enumerate(type_list):
        spe_dict[one_type].append(index_list[i])

    # shuffle the value list of each key
    for keys in spe_dict.keys():
        random.shuffle(spe_dict[keys])

    # generate a new index list for the value list
    new_index_list = []
    for one_type in type_list:
        value = spe_dict[one_type].pop()
        new_index_list.append(value)

    # swap the value_list by new_index_list
    value_list_new = value_list[new_index_list]

    return value_list_new


@LOSSFUNC.register_module(module_name="contrastive_regularization")
class ContrastiveLoss(AbstractLossClass):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def contrastive_loss(self, anchor, positive, negative):
        dist_pos = F.pairwise_distance(anchor, positive)
        dist_neg = F.pairwise_distance(anchor, negative)
        # Compute loss as the distance between anchor and negative minus the distance between anchor and positive
        loss = torch.mean(torch.clamp(
            dist_pos - dist_neg + self.margin, min=0.0))
        return loss

    def forward(self, common, specific, spe_label):
        # prepare
        bs = common.shape[0]
        # print(bs, common.shape, 'common.shape[0]')
        # 16 common.shape[0]
        fake_common, real_common, twofake_common = common.chunk(3)
        fake_spe, real_spe, twofake_spe = specific.chunk(3)
        fake_spe_label, real_spe_label, twofake_spe_label = spe_label.chunk(3)
        # common real
        idx_list = list(range(0, bs//3))
        random.shuffle(idx_list)
        fake_common_anchor = common[idx_list]
        # common fake
        idx_list = list(range(bs//3, 2*(bs//3)))
        random.shuffle(idx_list)
        real_common_anchor = common[idx_list]

        idx_list = list(range(2*(bs//3), bs))
        random.shuffle(idx_list)
        twofake_common_anchor = common[idx_list]

         # Compute the contrastive loss of common between real and fake
        loss_fakecommon = self.contrastive_loss(
            fake_common, fake_common_anchor, real_common_anchor)
        loss_realcommon = self.contrastive_loss(
            real_common, real_common_anchor, fake_common_anchor)
        loss_twofakecommon = self.contrastive_loss(
            twofake_common, twofake_common_anchor, real_common_anchor)
        loss_faketotwofakecommon = self.contrastive_loss(
            fake_common_anchor, twofake_common_anchor, real_common_anchor)
        


        # specific
        fake_spe_anchor = swap_spe_features(fake_spe_label, fake_spe)
        real_spe_anchor = swap_spe_features(real_spe_label, real_spe)
        twofake_spe_anchor = swap_spe_features(twofake_spe_label, twofake_spe)

        # Comupte the constrastive loss of specific between real and fake
        loss_fakespecific = self.contrastive_loss(
            fake_spe, fake_spe_anchor, real_spe_anchor)
        loss_realspecific = self.contrastive_loss(
            real_spe, real_spe_anchor, fake_spe_anchor)
        loss_twofakespecific = self.contrastive_loss(
            twofake_spe, twofake_spe_anchor, real_spe_anchor)
        loss_faketotwofakespecific = self.contrastive_loss(
            fake_spe_anchor, twofake_spe_anchor, real_spe_anchor)

        # Compute the final loss as the sum of all contrastive losses
        loss_common = loss_fakecommon + loss_realcommon + loss_twofakecommon + 0.1 * loss_faketotwofakecommon
        loss_spe = loss_fakespecific + loss_realspecific + loss_twofakespecific + 0.1 * loss_faketotwofakespecific
        loss = 0.5 * loss_common + 0.5 * loss_spe
        return loss
