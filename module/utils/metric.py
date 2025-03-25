import torch
from torch import Tensor
from mmseg.structures import SegDataSample
from mmseg.models.utils import resize
from mmengine.structures import PixelData

def postprocess_result(seg_logits: Tensor,
                       data_samples = None,
                       threshold = 0.3,
                       align_corners=False):

    batch_size, C, H, W = seg_logits.shape

    if data_samples is None:
        data_samples = [SegDataSample() for _ in range(batch_size)]
        only_prediction = True
    else:
        only_prediction = False

    post_logits = []
    for i in range(batch_size):
        if not only_prediction:
            img_meta = data_samples[i].metainfo
            if 'img_padding_size' not in img_meta:
                padding_size = img_meta.get('padding_size', [0] * 4)
            else:
                padding_size = img_meta['img_padding_size']
            padding_left, padding_right, padding_top, padding_bottom =\
                padding_size
            i_seg_logits = seg_logits[i:i + 1, :,
                                        padding_top:H - padding_bottom,
                                        padding_left:W - padding_right]

            flip = img_meta.get('flip', None)
            if flip:
                flip_direction = img_meta.get('flip_direction', None)
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    i_seg_logits = i_seg_logits.flip(dims=(3, ))
                else:
                    i_seg_logits = i_seg_logits.flip(dims=(2, ))

            i_seg_logits = resize(
                i_seg_logits,
                size=img_meta['ori_shape'],
                mode='bilinear',
                align_corners=align_corners,
                warning=False).squeeze(0)
        else:
            i_seg_logits = seg_logits[i]
        post_logits.append(i_seg_logits)

    return torch.stack(post_logits,dim=0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def inter_and_union(seg_map,gt_seg,map_tbl,cls_num,device,batch_data_samples):
    idxs = [s.cls_chosen if type(s.cls_chosen)==int else map_tbl[s.cls_chosen] for s in batch_data_samples]
    intersection_c, union_c = torch.zeros((cls_num,),device=device), torch.zeros((cls_num,),device=device)
    for i in range(len(idxs)):
        intersection_, union_, _ = intersectionAndUnion(seg_map[i], gt_seg[i], 2, 255)
        intersection_c[0] += intersection_[0]
        intersection_c[idxs[i]] += intersection_[1]
        union_c[0] += union_[0]
        union_c[idxs[i]] += union_[1]
    return intersection_c, union_c
    
def printer_miou(accelerator,intersection_meter,union_meter):
    # accelerator.print('counter:',intersection_meter.count)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    miou = torch.mean(iou_class[1:])*100
    accelerator.print('miou:{:.2f}'.format(miou))


