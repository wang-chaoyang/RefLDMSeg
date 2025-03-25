from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch

from mmseg.models.data_preprocessor import SegDataPreProcessor

@MODELS.register_module()
class PairPreProcessor(SegDataPreProcessor):

    def __init__(self, allow_resize=False,**kwargs):
        super().__init__(**kwargs)
        self.allow_resize=allow_resize
    
    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        
        que_data = super().forward(data['que'], training)
        sup_data = []
        for _sup in data['sup']:
            x = super().forward(_sup, training=True)
            sup_data.append(x)
        sup_img = torch.stack([x['inputs'] for x in sup_data],dim=1)
        sup_seg = []
        for x in sup_data:
            _samples = x['data_samples']
            _seg = torch.stack([_x.gt_sem_seg.data for _x in _samples],dim=0)
            sup_seg.append(_seg)
        sup_seg = torch.stack(sup_seg,dim=1)
        return dict(inputs=dict(que=que_data['inputs'],sup=sup_img,sup_seg=sup_seg),data_samples=que_data['data_samples'])
    
    