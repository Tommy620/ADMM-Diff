# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Following code is adapted from 
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''
# 这个代码是用来可视化迭代去噪bev的，可视化逻辑是可视化调度器5轮生成的bev图。
import argparse
import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import matplotlib.pyplot as plt
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model
from layout_diffusion.admm_denoiser_mynet_ import ADMMDenoiser
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample

logger = get_logger(__name__, log_level="INFO")

def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument('--bev_config', 
                        default="",
                        help='test config file path')
    
    parser.add_argument('--bev_checkpoint', 
                        default="",
                        help='checkpoint file')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        choices=[
            "CompVis/stable-diffusion-v1-4",
            # "stabilityai/stable-diffusion-2-1"
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="The checkpoint directory of admm_denoiser.",
    )


    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'sample' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    
    parser.add_argument(
        "--use_classifier_guidence",
        action='store_true',
        help="whether to use classifier guidence",
    )
    
    parser.add_argument(
        '--noise_timesteps', 
        type=int, 
        default=0, 
        help='The number of timesteps to add noise.')
    
    parser.add_argument(
        '--denoise_timesteps', 
        type=int, 
        default=5, 
        help='The number of timesteps to denoise.')
    
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=5, 
        help='The number of diffusion steps to run the admm_denoiser.')
    
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')


    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

    
def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)
    
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
        
    if args.launcher != 'none':
        init_dist(args.launcher, **bev_cfg.dist_params)
        
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
    bev_model = get_bev_model(args)
    if not args.use_classifier_guidence:
        bev_model.requires_grad_(False)
    
    admm_denoiser = ADMMDenoiser()
    admm_denoiser.from_pretrained(args.checkpoint_dir, subfolder="admm_denoiser")
    device = next(bev_model.parameters()).device
    admm_denoiser.to(device, dtype=torch.float32)
    admm_denoiser.requires_grad_(False) 
    admm_denoiser.eval()
    
    bev_cfg.data.test.test_mode = True
    bev_cfg.data.test.load_annos = True
    dataset = build_dataset(bev_cfg.data.test,
                            default_args={
                                        'pc_range': bev_cfg.point_cloud_range,
                                        'use_3d_bbox': bev_cfg.use_3d_bbox,
                                        'num_classes': bev_cfg.num_classes,
                                        'num_bboxes': bev_cfg.num_bboxes,
                                    })
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=bev_cfg.data.samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
        drop_last=True
    )
  
    save_path = os.path.join('../../test', args.bev_config.split('/')[-1].split('.')[-2]) #, args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1]
        
    evaluate(admm_denoiser=admm_denoiser,
             bev_model=bev_model,
             noise_scheduler=noise_scheduler,
             dataset=dataset,
             dataloader=dataloader,
             bev_cfg=bev_cfg,
             eval=args.eval,
             save_path=save_path,
             noise_timesteps=args.noise_timesteps,
             denoise_timesteps=args.denoise_timesteps,
             num_inference_steps=args.num_inference_steps,
             use_classifier_guidence=args.use_classifier_guidence)


def evaluate(admm_denoiser,
             bev_model,
             noise_scheduler,
             dataset,
             dataloader,
             bev_cfg,
             eval='bbox',
             save_path='',
             noise_timesteps=0,
             denoise_timesteps=0,
             num_inference_steps=0,
             use_classifier_guidence=False):
    
    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        loss = bev_model(return_loss=False, only_bev=False, given_bev=x_, return_eval_loss=True, **kwargs)
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient
    
    def get_condition(batch, use_cond=True):
        cond = {}
        if 'layout_obj_classes' in batch:
            cond['obj_class'] = torch.stack(batch['layout_obj_classes'].data[0])
        if 'layout_obj_bboxes' in batch:
            cond['obj_bbox'] = torch.stack(batch['layout_obj_bboxes'].data[0])
        if 'layout_obj_is_valid' in batch:
            cond['is_valid_obj'] = torch.stack(batch['layout_obj_is_valid'].data[0]) 
        if 'layout_obj_names' in batch:
            cond['obj_name'] = torch.stack(batch['layout_obj_names'].data[0])
        
        if not use_cond:
            if isinstance(admm_denoiser, ADMMDenoiser):
                if 'obj_class' in admm_denoiser.layout_encoder.used_condition_types:
                    cond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(admm_denoiser.layout_encoder.num_classes_for_layout_object - 1)
                    cond['obj_class'][:, 0] = admm_denoiser.layout_encoder.num_classes_for_layout_object - 2
                if 'obj_name' in admm_denoiser.layout_encoder.used_condition_types:
                    cond['obj_name'] = torch.stack(batch['default_obj_names'].data[0])
                if 'obj_bbox' in admm_denoiser.layout_encoder.used_condition_types:
                    cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
                    if admm_denoiser.layout_encoder.use_3d_bbox:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
                    else:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 1, 1])
                cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                cond['is_valid_obj'][:, 0] = 1.0 
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond[key] = value.to(latents.device)            
        return cond
    

    
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-mini', dataroot='/root/autodl-fs/nuscenes', verbose=True)

    vis_dict = {}

    for step, batch in enumerate(dataloader):
        vis_ls = []  # 初始BEV/加噪BEV/cond最后一轮输出/调度器最终计算输出

        TARGET_SAMPLE_TOKEN = batch['img_metas'][0].data[0][0]['sample_idx']

        print(f"Processing target sample: {TARGET_SAMPLE_TOKEN}")

        latents = bev_model(return_loss=False, only_bev=True, **batch).detach()

        latents = latents.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)

        latents = latents.permute(0, 3, 1, 2) #1,256,50,50
        # vis_ls.append(latents.clone().detach().cpu())  # bev模型输出的原始BEV # 要外围图解禁这个
        if noise_timesteps > 0: #5
            if noise_timesteps > 1000:
                latents = torch.randn_like(latents)
                latents = latents * noise_scheduler.init_noise_sigma
            else:
                noise = torch.randn_like(latents)
                noise_timesteps = torch.tensor(noise_timesteps).long()
                latents = noise_scheduler.add_noise(latents, noise, noise_timesteps) #1,256,50,50 #这个latents就是x_t
        # vis_ls.append(latents.clone().detach().cpu())  # 加噪后的完全噪声图 # 要外围图解禁这个
        if denoise_timesteps > 0:
            cond, uncond = get_condition(batch, use_cond=True), get_condition(batch, use_cond=False)

            # # DDIM
            noise_scheduler.config.num_train_timesteps=denoise_timesteps #5 原本调度器默认1000步去噪，现在定义为5步去噪完毕
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)


            for idx, t in enumerate(noise_scheduler.timesteps): #5，4，3，2，1
                t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)

                # 计算当前时间步的sqrt_alpha_t
                sqrt_alpha_t = torch.sqrt(noise_scheduler.alphas_cumprod[t])
                sqrt_alpha_t = sqrt_alpha_t.view(-1, 1, 1, 1).expand(latents.shape[0], 1, 1, 1).to(latents.device)

                # 两个调用共用相同的sqrt_alpha_t
                denoiser_kwargs = {
                    'timesteps': t_batch,
                    'sqrt_alpha_t': sqrt_alpha_t
                }
                # 要外围图解禁这个
                noise_pred_uncond = admm_denoiser(latents, **denoiser_kwargs, **uncond)
                noise_pred_cond = admm_denoiser(latents, **denoiser_kwargs, **cond)
                noise_pred = noise_pred_uncond + 2 * (noise_pred_cond - noise_pred_uncond)



                # 要外围图解禁这个
                if t == 1:
                    vis_ls.append(noise_pred.clone().detach().cpu())

                classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]
            # vis_ls.append(latents.clone().detach().cpu()) # 要外围图解禁这个
            # 可视化内部ADMM迭代结果
            vis_dir = f'/tmp/pycharm_project_55/BEVFormer/tools/vis_doc'
            # lidar_render_path = f'/tmp/pycharm_project_745/BEVFormer/tools/vis_doc/lidar_renders/{TARGET_SAMPLE_TOKEN}.png'
            print(f"\nVisualizing sample token:{TARGET_SAMPLE_TOKEN}")
            if vis_ls:
                collect_vis_ls(
                    vis_dict=vis_dict,
                    vis_ls=vis_ls,
                    token=TARGET_SAMPLE_TOKEN
                )


        if rank == 0:
            prog_bar.update()

    save_all_vis(vis_dict, save_path='/tmp/pycharm_project_55/BEVFormer/tools/vis_doc/admm_vis.pt')


def collect_vis_ls(vis_dict, vis_ls, token):
    vis_ls_cpu = [v.detach().cpu() for v in vis_ls]
    vis_dict[token] = vis_ls_cpu

def save_all_vis(vis_dict, save_path):
    torch.save(vis_dict, save_path)
    print(f"Saved {len(vis_dict)} samples to {save_path}")


def visualize_bev_features(feat_ls_1, save_dir=None, token = None, step_names=None):
    """
    可视化BEV特征图 - 展示去噪效果，使用灰度显示

    Args:
        vis_ls: 包含三个BEV特征图的列表 [原始BEV, 加噪BEV, 去噪BEV]
        save_dir: 保存目录
        step_names: 每个步骤的名称列表，默认为['original', 'noisy', 'denoised']
    """
    os.makedirs(save_dir, exist_ok=True)


    # 步骤1：先找出所有特征图的全局最小值和最大值 此代码的添加是为了统一colorbar到同一个范围，不然每一张图虽然是黑小白大，但每个图的min和max不同，仍然colorbar不同。
    # global_min = float('inf')
    # global_max = float('-inf')
    #
    # for f_l in feat_ls_list:
    #     for feat_in_f_l in f_l:
    #         if isinstance(feat_in_f_l, torch.Tensor):
    #             feat_in_f_l = feat_in_f_l.numpy()
    #
    #         # 在整个张量上找最大最小（包括所有通道）
    #         global_min = min(global_min, feat_in_f_l.min())
    #         global_max = max(global_max, feat_in_f_l.max())
    # if global_max - global_min < 1e-8:
    #     global_max = global_min + 1e-8
    # 步骤2：使用统一的vmin/vmax绘制所有子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 定义每行的名称
    row_names = ['x (primal var)', 'v (auxiliary)', 'Rv (sparse code)',
                 'w_sparse (thresholded)', 'z (auxiliary var)', 'u (Lagrange multiplier)']

    # ========== 新增：打印统计信息 ==========
    print("\n" + "=" * 100)
    print(
        f"{'Variable':<20} {'Iter':<6} {'Mean':<15} {'Median':<15} {'Variance':<15} {'Std':<15} {'Min':<15} {'Max':<15}{'Sparsity info':<20}")
    print("=" * 100)



    for lie, feat_in_f_l in enumerate(feat_ls_1):
        if isinstance(feat_in_f_l, torch.Tensor):
            feat_in_f_l = feat_in_f_l.numpy()




        # 归一化到[-1, 1]
        # feat_in_f_l_min, feat_in_f_l_max = feat_in_f_l.min(), feat_in_f_l.max()
        # if feat_in_f_l_max - feat_in_f_l_min >= 1e-8:
        #     feat_in_f_l_norm = 2 * (feat_in_f_l - feat_in_f_l_min) / (feat_in_f_l_max - feat_in_f_l_min) - 1

        #全局归一化
        # feat_in_f_l_norm = 2 * (feat_in_f_l - global_min) / (global_max - global_min) - 1

        # 负数置零，正数保留 (ReLU风格)
        # feat_in_f_l_relu = np.maximum(feat_in_f_l, 0)  # 所有小于0的变成0，大于0的保持不变

        #  取绝对值 bevdiffuser用的不是这种
        # feat_in_f_l_abs = np.abs(feat_in_f_l)

        # 通道维度平均 [1, C, H, W] -> [H, W]
        feat_in_f_l_mean = np.mean(feat_in_f_l[0], axis=0)  # 在通道维度上平均

        #上下翻转180度
        feat_in_f_l_mean = np.flipud(feat_in_f_l_mean)

        # 使用gray色图显示，更接近lidar俯视图的感觉
        im = axes[lie].imshow(feat_in_f_l_mean, cmap='gray', aspect='auto', interpolation='bicubic')
        axes[lie].set_title(f'Iter {lie+1}', fontsize=10)
        axes[lie].axis('off')

        # 添加colorbar，显示强度值范围

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)





    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{token}.png')

    print(f"Visualization saved to {save_path}")
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight') #之前是150，调成300看看效果
    plt.close()



def lidiar_render_gt_simple(dataset, sample_token, out_path=None):
    """
    最简单的GT-only LiDAR俯视图渲染
    """
    # 获取sample
    sample = dataset.get('sample', sample_token)

    # 创建只包含GT的EvalBoxes
    gt_annotations = EvalBoxes()
    bbox_gt_list = []

    for ann in sample['anns']:
        content = dataset.get('sample_annotation', ann)
        detection_name = category_to_detection_name(content['category_name'])
        if detection_name:
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=(0, 0),
                ego_translation=(0, 0, 0),
                num_pts=-1,
                detection_name=detection_name,
                detection_score=-1.0,
                attribute_name=''
            ))

    gt_annotations.add_boxes(sample_token, bbox_gt_list)

    # 空的预测
    pred_annotations = EvalBoxes()

    # 调用可视化函数，只显示GT
    visualize_sample(
        dataset,
        sample_token,
        gt_annotations,
        pred_annotations,
        savepath=out_path
    )


if __name__ == "__main__":
    test()



# 要可视化的时候，只需要修改内围还是外围； sample_token; 文件名称。即可。

