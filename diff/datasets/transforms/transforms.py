import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
# from scipy.ndimage import gaussian_filter
import random as sysrandom
# from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS
from mmengine.logging import print_log
import mmengine.fileio as fileio
import os.path as osp

@TRANSFORMS.register_module()
class VOSMultiTransform(BaseTransform):
    def __init__(self):
        super().__init__()
    
    def transform(self, results: Dict):
        cls_chosen = results['cls_chosen']
        label = results['gt_seg_map']
        if len(label.shape)!=2:
            segpath = results['seg_map_path']
            label = label[:,:,0]
        label_class = np.unique(label).tolist()
        if 255 in label_class:
            label_class.remove(255)
        if 0 in label_class:
            label_class.remove(0)

        if cls_chosen==None:
            cls_idx = label_class[sysrandom.randint(1,len(label_class)) - 1 ]
            frame = results['seg_map_path'].split('/')[-2]
            results['cls_chosen'] = f'{frame}:{cls_idx}'
        else:
            info = cls_chosen.split(':')
            frame, cls_idx = info[0], int(info[1])
        target_pix = np.where(label == cls_idx)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        results['gt_seg_map'] = label
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str



@TRANSFORMS.register_module()
class VOSTransform(BaseTransform):
    def __init__(self,target=255):
        super().__init__()
        self.tgt_idx = target   
    
    def transform(self, results: Dict):
        cls_chosen = results['cls_chosen']
        label = results['gt_seg_map']
        if len(label.shape)!=2:
            segpath = results['seg_map_path']
            label = label[:,:,0]
        if cls_chosen==None:
            cls_chosen = results['seg_map_path'].split('/')[-2]
            results['cls_chosen'] = cls_chosen
        target_pix = np.where(label == self.tgt_idx)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        results['gt_seg_map'] = label
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class FSTransform(BaseTransform):

    def __init__(self, ignore_index=255) -> None:
        self.ignore_index = ignore_index
    
    def transform(self, results: dict) -> dict:
        cls_chosen = results['cls_chosen']
        label = results['gt_seg_map']
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        assert len(label_class) > 0

        if cls_chosen==None:    
            cls_chosen = label_class[sysrandom.randint(1,len(label_class))-1]
            results['cls_chosen'] = cls_chosen
        target_pix = np.where(label == cls_chosen)
        ignore_pix = np.where(label == self.ignore_index)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = self.ignore_index
        results['gt_seg_map'] = label
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


