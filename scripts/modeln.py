import sys
sys.path.append('.')
import itertools
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DeepSpeedPlugin

from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)

from diffusers.optimization import get_scheduler
from module.utils.prepare_args import parse_args
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import MODELS
import diff
import mmseg
from module.utils.metric import postprocess_result, AverageMeter, printer_miou, inter_and_union
from module.utils.map_tbl import PASCAL,COCO
from module.utils.hook import *
from module.model.data import preprocess, create_pseu_mask
from module.model.prompter import cfg_pipeline_unpool
from module.model.clip import get_condition_unpool,build_vision_tower

logger = get_logger(__name__)

def val_fs(args,accelerator,scheduler,vae,unet,vis_encoder,val_dataloader,data_preprocessor,\
                  global_step,map_tbl,null_condition,igs=1.5,tgs=7):
    device = accelerator.device
    weight_dtype = null_condition.dtype

    cls_num = len(map_tbl)
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for iter,batch in enumerate(val_dataloader):
        if iter%200==0: accelerator.print(f'val iter:{iter}')
        with torch.no_grad():
            x,label,sx,sy,shot,batch_data_samples = preprocess(batch,data_preprocessor,weight_dtype,device,istrain=False)
            gt_seg = label.squeeze(1)
            
            image_latents = vae.encode(x).latent_dist.mode()
            prompt_embeds,symask = get_condition_unpool(sx,sy,vis_encoder,args,weight_dtype,shot)
            
            num_inference_steps=20
            bsz = image_latents.shape[0]
            scheduler.set_timesteps(num_inference_steps, device=device)
            model_pred = cfg_pipeline_unpool(
                scheduler,
                unet,
                image_latents,
                prompt_embeds,
                null_condition,
                symask,
                igs,
                tgs,
                )

            model_pred = model_pred.to(weight_dtype)
            seg_logits = vae.decode(model_pred/vae.config.scaling_factor, return_dict=False)[0]
 
            seg_logits = seg_logits-args.mask_alpha*x

            seg_map = postprocess_result(seg_logits.float(),batch_data_samples)
            seg_map = (seg_map[:,0,:,:]<seg_map[:,1,:,:]).long()

        intersection_c, union_c = inter_and_union(seg_map,gt_seg,map_tbl,cls_num,device,batch_data_samples)
        intersection = accelerator.gather_for_metrics(intersection_c.unsqueeze(0))
        union = accelerator.gather_for_metrics(union_c.unsqueeze(0))
        intersection_meter.update(intersection.sum(0),intersection.shape[0]), union_meter.update(union.sum(0),union.shape[0])

    accelerator.wait_for_everyone()
    printer_miou(accelerator,intersection_meter,union_meter)
    torch.cuda.empty_cache()



def main(args):
    cfg = Config.fromfile(args.config)
    cfg.train_dataloader.batch_size = args.train_batch_size
    train_loader_cfg=cfg.get('train_dataloader')
    data_preprocessor = MODELS.build(cfg.get('data_preprocessor'))
    train_dataloader = Runner.build_dataloader(train_loader_cfg)
    if args.val_dataset == 'pascal':
        val_loader_cfg=cfg.get('val_dataloader_pascal')
        map_tbl = PASCAL
    elif args.val_dataset == 'coco':
        val_loader_cfg=cfg.get('val_dataloader_coco')
        map_tbl = COCO
    else:
        raise ValueError('unregisted val dataset')
    
    val_loader_cfg['batch_size'] = args.val_batch_size
    val_dataloader = Runner.build_dataloader(val_loader_cfg)

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin if args.ds else None
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrain_model, subfolder="scheduler")

    assert args.pretrain_vae is not None
    vae = AutoencoderKL.from_pretrained(args.pretrain_vae, subfolder=None, revision=args.revision)
    vis_encoder = build_vision_tower(args,accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrain_model, subfolder="unet", revision=args.revision, \
                                                encoder_hid_dim=1024, low_cpu_mem_usage=False)

    vae.requires_grad_(False)
    vis_encoder.requires_grad_(False)

    in_channels = 8
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)


    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
   
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    vis_encoder.to(accelerator.device)

    module_to_optimize = unet
    params_to_optimize = module_to_optimize.parameters()

    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    scale_ratio = accelerator.num_processes if not args.ds else 1
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * scale_ratio,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps * scale_ratio,
    )

    module_to_optimize, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
            module_to_optimize, optimizer, train_dataloader, lr_scheduler, val_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("model", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    device = accelerator.device
    colormap = {0:[255,0,127.5],1:[0,255,127.5]}
    colormap = {k:torch.tensor(colormap[k],dtype=weight_dtype,device=device) for k in colormap.keys()}
    colormap = {k:(colormap[k]-127.5)/127.5 for k in colormap.keys()}

    null_sx = torch.zeros((1,3,224,224),device=device,dtype=weight_dtype)
    null_sy = torch.ones((1,1,224,224),device=device)
    null_condition,_ = get_condition_unpool(null_sx,null_sy,vis_encoder,args,weight_dtype,shot=1)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.only_val:
        loadweight(unet,args.output_dir)
        module_to_optimize.eval()
        val_fs(args,accelerator,noise_scheduler,vae,unet,vis_encoder,val_dataloader,data_preprocessor,\
                  global_step,map_tbl,null_condition)
    else:
        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                module_to_optimize.train()

                with accelerator.accumulate(unet):
                    x,label,sx,sy,shot,_ = preprocess(batch,data_preprocessor,weight_dtype,device,istrain=True)
                    assert shot==1
                    pseudomask = create_pseu_mask(colormap,label,weight_dtype,device)
                    maskiage = (1-args.mask_alpha)*pseudomask + args.mask_alpha*x
 
                    original_image_embeds = vae.encode(x).latent_dist.mode()
                    latent = vae.encode(maskiage).latent_dist.sample()
                    latent = latent * vae.config.scaling_factor

                    noise = torch.randn_like(latent)
                    bsz = latent.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                    timesteps = timesteps.long()
                    noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)

                    prompt_embeds,symask = get_condition_unpool(sx,sy,vis_encoder,args,weight_dtype,shot)

                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(bsz, device=device, generator=generator)
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                        prompt_embeds = torch.where(prompt_mask, null_condition, prompt_embeds)

                        image_mask_dtype = original_image_embeds.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        original_image_embeds = image_mask * original_image_embeds

                    concat_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latent, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                    model_pred = unet(concat_noisy_latents, timesteps, prompt_embeds,encoder_attention_mask=symask).sample
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        save_normal(accelerator,args,logger,global_step)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if global_step >= args.max_train_steps:
                    return

        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
