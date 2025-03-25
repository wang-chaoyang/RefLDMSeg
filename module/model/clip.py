import torch
import torch.nn.functional as F
import alpha_clip
from types import MethodType
from einops import rearrange


def select_feature_alpha(self, x: torch.Tensor, alpha=None, return_attn=False):
    x = self.conv1(x)  
    x = x + self.conv1_alpha(alpha)
    x = x.reshape(x.shape[0], x.shape[1], -1)  
    x = x.permute(0, 2, 1)  
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)  
    if return_attn:
        x, attn_last = self.transformer(x, return_attn=True)
    else:
        x = self.transformer(x, return_attn=False)
    x = x.permute(1, 0, 2)  
    return x



def build_vision_tower(args,device):
    clipmodel, _ = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth=args.pretrain_vision, device=device)
    vis_encoder = clipmodel.visual
    setattr(vis_encoder,'forward',MethodType(select_feature_alpha,vis_encoder))
    return vis_encoder



def get_condition_unpool(sx,sy,vis_encoder,args,dtype,shot):
    masktype = args.mask
    assert masktype == 'm'
    alpha = True if "AC" in args.vision_tower else False
    vis_size = args.vis_size

    with torch.no_grad():
        dtype = sx.dtype
        raw_sy = torch.clone(sy)
        sy[sy==255] = 0
        sx = F.interpolate(sx.float(),(vis_size,vis_size),mode='bilinear',align_corners=False)
        if alpha:
            sx = sx.to(dtype=dtype).half()
            alpha_mask = F.interpolate(sy.float(),(vis_size,vis_size),mode='nearest')
            alpha_mask = alpha_mask.sub_(0.5).div_(0.26).half()
            sup_feat = vis_encoder(sx,alpha_mask)    
            sup_feat = sup_feat.to(dtype)
        else:
            sup_feat = vis_encoder(sx).to(dtype)
        
        b,l,d = sup_feat.shape

        attn_mask = F.interpolate(sy.float(),(16,16),mode='nearest')
        attn_mask = rearrange(attn_mask,'(b k) c h w -> b (k h w) c',k=shot).squeeze(2)
        sup_feat = sup_feat[:,1:,:]
        sup_feat = rearrange(sup_feat,'(b k) l d -> b (k l) d',k=shot)
    return (sup_feat,attn_mask.long().to(sx.device))    

