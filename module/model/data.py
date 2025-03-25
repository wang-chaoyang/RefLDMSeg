import torch
import torch.nn.functional as F
import math
from einops import rearrange


def get_condition(sx,sy,vis_encoder,vis_size,dtype,shot):
    with torch.no_grad():
        sx = F.interpolate(sx.float(),(vis_size,vis_size),mode='bilinear',align_corners=False)
        sx = sx.to(dtype=dtype)
        sup_feat = vis_encoder(sx).last_hidden_state[:,1:,:]
        vis_out_size = int(math.sqrt(sup_feat.shape[1]))
        sy[sy==255] = 0
        sy = F.interpolate(sy.float(),(vis_out_size,vis_out_size),mode='nearest')
        sy = sy.to(dtype=dtype)
        sy = rearrange(sy,'b c h w -> b (h w) c')
        proto = (sup_feat*sy).sum(dim=1)/(sy.sum(dim=1)+1e-6)
        prompt_embeds = rearrange(proto.unsqueeze(1),'(b k) q c -> b k q c',k=shot)
    return prompt_embeds    # b,c



def preprocess(batch,data_preprocessor,weight_dtype,device,istrain,**kwargs):
    batch = data_preprocessor(batch,training=istrain)
    inputs = batch['inputs']
    x = inputs['que'].to(dtype=weight_dtype,device=device)
    sx = inputs['sup'].to(dtype=weight_dtype,device=device)
    sy = inputs['sup_seg'].to(dtype=weight_dtype,device=device)
    shot = sx.shape[1]
    sx = rearrange(sx,'b k c h w -> (b k) c h w')
    sy = rearrange(sy,'b k c h w -> (b k) c h w')
    batch_data_samples = batch['data_samples']
    gt_sem_seg = [
        data_sample.gt_sem_seg.data for data_sample in batch_data_samples
    ]
    label = torch.stack(gt_sem_seg, dim=0).to(device=device)
    return x,label,sx,sy,shot,batch_data_samples

def create_pseu_mask(colormap,label,weight_dtype,device):

    assert len(label.shape)==4
    b,c,h,w = label.shape
    pseudomask = torch.ones((b,3,h,w),dtype=weight_dtype,device=device)*colormap[0][None,:,None,None]
    pseudomask = pseudomask.permute(0,2,3,1)
    pseudomask[label.squeeze(1)==1] = colormap[1]
    pseudomask = pseudomask.permute(0,3,1,2)
    return pseudomask
