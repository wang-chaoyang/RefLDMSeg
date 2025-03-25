import os
import os.path as osp
from accelerate import Accelerator,DistributedType
import shutil
import torch

def loadweight(model, loadpath):

    print(f'load ckpt from {loadpath}')
    state_dict = torch.load(osp.join(loadpath,'pytorch_model/mp_rank_00_model_states.pt'))['module']
    if hasattr(model,'module'):
        model.module.load_state_dict(state_dict,strict=True)
    else:
        model.load_state_dict(state_dict,strict=True)


def save_normal(accelerator:Accelerator,args,logger,global_step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
        
    if accelerator.is_main_process:
        if args.checkpoints_total_limit is not None:
            checkpoints = os.listdir(args.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) > args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)
    