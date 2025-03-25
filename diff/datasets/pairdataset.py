from typing import Any
import torch
import numpy as np
import random
import copy
import os.path as osp
    
from mmseg.registry import DATASETS
from mmengine.dataset import Compose
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PairDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'foreground'),
        palette=[[0, 0, 0], [128, 0, 0]])

    def __init__(self,
                 meta_list:str,
                 supp_pipeline,
                 vos=False,
                 shot:int=1,
                 **kwargs):
        super().__init__(**kwargs)
        _, self.clsdct = np.load(meta_list,allow_pickle=True)
        self.shot = shot
        self.vos = vos
        self.supp_pipeline = Compose(supp_pipeline)

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        null_data_info = copy.deepcopy(data_info)
        data_info = self.pre_pipeline(data_info,cls_chosen=None)
        que = self.pipeline(data_info)
        cls_chosen = que['data_samples'].cls_chosen
        sups = []
        file_class_chosen = self.clsdct[cls_chosen]
        num_file = len(file_class_chosen)
        if num_file<=self.shot:
            if not self.test_mode:
                return None
            else:
                err_que_path = que['data_samples'].img_path
                raise ValueError(f'{err_que_path} only have {num_file} samples in {self.shot} shot')
        support_idx_list = []
        for k in range(self.shot):
            sup_info = copy.deepcopy(null_data_info)
            sup_info.pop('sample_idx')
            sup_idx = random.randint(1,num_file)-1
            sup_seg_name = osp.join(self.data_prefix.get('seg_map_path',None),file_class_chosen[sup_idx]+self.seg_map_suffix)
            while((sup_seg_name == data_info['seg_map_path']) or sup_seg_name in support_idx_list):   # avoid conflict by name
                sup_idx = random.randint(1,num_file)-1
                sup_seg_name = osp.join(self.data_prefix.get('seg_map_path',None),file_class_chosen[sup_idx]+self.seg_map_suffix)
            support_idx_list.append(sup_seg_name)
            sup_info['seg_map_path'] = sup_seg_name
            sup_info['img_path'] = osp.join(self.data_prefix.get('img_path',None),file_class_chosen[sup_idx]+self.img_suffix)
            sup_info = self.pre_pipeline(sup_info,cls_chosen=cls_chosen)
            sup = self.supp_pipeline(sup_info)
            sups.append(sup)
            del sup_info
            assert sup['inputs'].shape[1:]==sup['data_samples'].gt_sem_seg.shape, print(f'wrong shape in {sup_seg_name}',\
                                                                                        sup['inputs'].shape,sup['data_samples'].gt_sem_seg.shape)
        for sup in sups:
            if sup['data_samples'].gt_sem_seg.data.sum()==0:
                return None
        return dict(que=que,sup=sups)

    def pre_pipeline(self,data_info,cls_chosen=None):
        data_info['cls_chosen'] = cls_chosen
        return data_info



        