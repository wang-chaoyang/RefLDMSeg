import torch
import torch.nn.functional as F
import math
from einops import rearrange


def get_condition_unpool(sx,sy,vis_encoder,vis_size,dtype,shot):
    with torch.no_grad():
        sx = F.interpolate(sx.float(),(vis_size,vis_size),mode='bilinear',align_corners=False)
        sx = sx.to(dtype=dtype)
        sup_feat = vis_encoder(sx).last_hidden_state[:,1:,:]   
        vis_out_size = int(math.sqrt(sup_feat.shape[1]))
        sy[sy==255] = 0
        sy = F.interpolate(sy.float(),(vis_out_size,vis_out_size),mode='nearest')
        sy = sy.to(dtype=dtype)
        sy = rearrange(sy,'(b k) c h w -> b (k h w) c',k=shot)
        sy = sy.squeeze(2)
        sup_feat = rearrange(sup_feat,'(b k) q c -> b k q c',k=shot)
    return (sup_feat,sy)    



def cfg_pipeline_unpool(scheduler,unet,image_latents,encoder_hidden_states,blank_feat,symask, \
                        image_guidance_scale,guidance_scale):
    scheduler_is_in_sigma_space = hasattr(scheduler, "sigmas")
    latents = torch.randn_like(image_latents)
    latents *= scheduler.init_noise_sigma
    cls_free = image_guidance_scale >= 1.0 and guidance_scale>=1.0
    bsz = image_latents.shape[0]
    if len(blank_feat.shape)==4:
        shot = encoder_hidden_states.shape[1]
        blank_feat = blank_feat.repeat(bsz,shot,1,1)
    else:
        shot = encoder_hidden_states.shape[1] // blank_feat.shape[1]
        blank_feat = blank_feat.unsqueeze(0).repeat(shot,bsz,1,1)
        blank_feat = rearrange(blank_feat, 'k b q c -> b (k q) c')
    if cls_free:
        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        encoder_hidden_states = torch.cat([encoder_hidden_states, blank_feat, blank_feat], dim=0)
        symask = torch.concat([symask,symask,symask],dim=0)

    timesteps = scheduler.timesteps
    extra_step_kwargs = {}
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 3) if cls_free else latents
        scaled_latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

        noise_pred = unet(
            scaled_latent_model_input, t, encoder_hidden_states, encoder_attention_mask=symask, return_dict=False
        )[0]

        if scheduler_is_in_sigma_space:
            step_index = (scheduler.timesteps == t).nonzero()[0].item()
            sigma = scheduler.sigmas[step_index]
            noise_pred = latent_model_input - sigma * noise_pred

        if cls_free:
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

        if scheduler_is_in_sigma_space:
            noise_pred = (noise_pred - latents) / (-sigma)

        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    return latents