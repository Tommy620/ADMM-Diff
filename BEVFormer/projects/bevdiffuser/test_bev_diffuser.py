# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

"""
Following code is adapted from 
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

import argparse
import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import time

import distutils.version
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
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import (
    custom_encode_mask_results,
    collect_results_cpu,
)
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model
from layout_diffusion.admm_denoiser_mynet_ import ADMMDenoiser

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument("--bev_config", default="", help="test config file path")

    parser.add_argument("--bev_checkpoint", default="", help="checkpoint file")

    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="pytorch",
        help="job launcher",
    )

    parser.add_argument("--local_rank", type=int, default=0)

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
        action="store_true",
        help="whether to use classifier guidence",
    )

    parser.add_argument(
        "--noise_timesteps",
        type=int,
        default=0,
        help="The number of timesteps to add noise.",
    )

    parser.add_argument(
        "--denoise_timesteps",
        type=int,
        default=5,
        help="The number of timesteps to denoise.",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=5,
        help="The number of diffusion steps to run the admm_denoiser.",
    )

    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )

    # ============= Table 7 参数：BEV Feature Quality Analysis =============
    parser.add_argument(
        "--measure_feature_quality",
        action="store_true",
        default=False,
        help="Table 7: 启用 BEV 特征质量度量（与训练目标 clean BEV 比较 MSE/PSNR/cosine）",
    )  # 当你要复现获得table7的结果时，直接在训练指令中传入--measure_feature_quality 即可，只是个开关，无需赋值。
    # ======================================================================

    # ============= Table 8 参数 =============
    parser.add_argument(
        "--enable_timing",
        action="store_true",
        default=False,
        help="Table 8: 总开关，启用 z-update / total forward 的细粒度计时",
    )
    parser.add_argument(
        "--use_sparse_ops",
        action="store_true",
        default=False,
        help='Table 8: 启用 z-update 的稀疏掩码（对应 paper "+ sparse ops" 行）',
    )
    parser.add_argument(
        "--timing_warmup_batches",
        type=int,
        default=10,
        help="Table 8: 计时前的 warmup batch 数",
    )
    parser.add_argument(
        "--timing_num_samples", type=int, default=500, help="Table 8: 用于平均的样本数"
    )  # 当你要复现获得table8的结果时，在训练指令中传入134获得不使用稀疏归0，1234则使用，期待后者比前者快。不要table8时不用这些参数的。
    # ==========================================
    parser.add_argument(
        "--use_up_down_sample",
        action="store_true",
        default=False,
        help="训练tiny模型不用传入；训练base或者v2就要传入，无需参数",
    )
    parser.add_argument(
        "--num_admm_iters",
        type=int,
        default=4,
        help="Number of unrolled ADMM iterations used by ADMMDenoiser.",
    )
    # 实验性旋钮（默认静默）：测 sparse 权重时须与训练用同一 c。
    parser.add_argument(
        "--sparsity_coef",
        type=float,
        default=0.0,
        help="V3 固定相对软阈值系数 c：threshold = c * mean(|Rv|)。默认 0.0=不产生稀疏。",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    if args.launcher != "none":
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

    admm_denoiser = ADMMDenoiser(
        num_admm_iters=args.num_admm_iters, sparsity_coef=args.sparsity_coef
    )
    admm_denoiser.use_up_down_sample = args.use_up_down_sample
    admm_denoiser.from_pretrained(args.checkpoint_dir, subfolder="admm_denoiser")
    device = next(bev_model.parameters()).device
    admm_denoiser.to(device, dtype=torch.float32)
    admm_denoiser.requires_grad_(False)
    admm_denoiser.eval()

    # ============= Table 8：透传开关给 denoiser =============
    admm_denoiser.enable_timing = args.enable_timing
    admm_denoiser.use_sparse_ops = args.use_sparse_ops
    if args.enable_timing:
        print(f"[Table 8] enable_timing  = True")
        print(f"[Table 8] use_sparse_ops = {admm_denoiser.use_sparse_ops}")
    # ========================================================

    bev_cfg.data.test.test_mode = True
    bev_cfg.data.test.load_annos = True
    dataset = build_dataset(
        bev_cfg.data.test,
        default_args={
            "pc_range": bev_cfg.point_cloud_range,
            "use_3d_bbox": bev_cfg.use_3d_bbox,
            "num_classes": bev_cfg.num_classes,
            "num_bboxes": bev_cfg.num_bboxes,
        },
    )
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=bev_cfg.data.samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != "none"),
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )

    save_path = os.path.join(
        "../../test", args.bev_config.split("/")[-1].split(".")[-2]
    )  # , args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1]

    evaluate(
        admm_denoiser=admm_denoiser,
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
        measure_feature_quality=args.measure_feature_quality,
        use_classifier_guidence=args.use_classifier_guidence,
        enable_timing=args.enable_timing,
        timing_warmup_batches=args.timing_warmup_batches,
        timing_num_samples=args.timing_num_samples,
    )


def evaluate(
    admm_denoiser,
    bev_model,
    noise_scheduler,
    dataset,
    dataloader,
    bev_cfg,
    eval="bbox",
    save_path="",
    noise_timesteps=0,
    denoise_timesteps=0,
    num_inference_steps=0,
    measure_feature_quality=False,
    use_classifier_guidence=False,
    enable_timing=False,  # Table 8
    timing_warmup_batches=10,  # Table 8
    timing_num_samples=500,  # Table 8
):
    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, bev_cfg.bev_h_ * bev_cfg.bev_w_, bev_cfg._dim_)
        loss = bev_model(
            return_loss=False,
            only_bev=False,
            given_bev=x_,
            return_eval_loss=True,
            **kwargs,
        )
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient

    def get_condition(batch, use_cond=True):
        cond = {}
        if "layout_obj_classes" in batch:
            cond["obj_class"] = torch.stack(batch["layout_obj_classes"].data[0])
        if "layout_obj_bboxes" in batch:
            cond["obj_bbox"] = torch.stack(batch["layout_obj_bboxes"].data[0])
        if "layout_obj_is_valid" in batch:
            cond["is_valid_obj"] = torch.stack(batch["layout_obj_is_valid"].data[0])
        if "layout_obj_names" in batch:
            cond["obj_name"] = torch.stack(batch["layout_obj_names"].data[0])

        if not use_cond:
            if isinstance(admm_denoiser, ADMMDenoiser):
                if "obj_class" in admm_denoiser.layout_encoder.used_condition_types:
                    cond["obj_class"] = torch.ones_like(cond["obj_class"]).fill_(
                        admm_denoiser.layout_encoder.num_classes_for_layout_object - 1
                    )
                    cond["obj_class"][:, 0] = (
                        admm_denoiser.layout_encoder.num_classes_for_layout_object - 2
                    )
                if "obj_name" in admm_denoiser.layout_encoder.used_condition_types:
                    cond["obj_name"] = torch.stack(batch["default_obj_names"].data[0])
                if "obj_bbox" in admm_denoiser.layout_encoder.used_condition_types:
                    cond["obj_bbox"] = torch.zeros_like(cond["obj_bbox"])
                    if admm_denoiser.layout_encoder.use_3d_bbox:
                        cond["obj_bbox"][:, 0] = torch.FloatTensor(
                            [0, 0, 0, 1, 1, 1, 0, 0, 0]
                        )
                    else:
                        cond["obj_bbox"][:, 0] = torch.FloatTensor([0, 0, 1, 1])
                cond["is_valid_obj"] = torch.zeros_like(cond["is_valid_obj"])
                cond["is_valid_obj"][:, 0] = 1.0
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond[key] = value.to(latents.device)
        return cond

    det_res_path = f"{noise_timesteps}_{denoise_timesteps}_{num_inference_steps}"
    bbox_results = []
    mask_results = []
    have_mask = False

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    # 测时间用
    total_time = 0
    num_batches = len(dataloader)
    start_time = time.time()
    gamma_vals = admm_denoiser.gamma.detach().flatten().tolist()
    print("gamma value: " + ", ".join(f"{g:.4f}" for g in gamma_vals))

    # ============= Table 7：feature quality 累计量 =============
    # 我们采用"逐样本累计 + 最后求平均"的写法，避免在循环中存全部 tensor。
    fq_sum_mse = 0.0  # 累计的 per-sample MSE 总和
    fq_sum_psnr = 0.0  # 累计的 per-sample PSNR 总和（dB）
    fq_sum_cos = 0.0  # 累计的 per-sample cosine similarity 总和
    fq_count = 0  # 实际参与统计的样本数
    # =============================================================

    # ============= Table 8：计时控制状态（仅在 enable_timing 时生效）=============
    timing_started = False
    timing_sample_count = 0
    # ==========================================================================

    for step, batch in enumerate(dataloader):
        # ============= Table 8：到 warmup 阈值后开启计时 =============
        if enable_timing:
            if (not timing_started) and step >= timing_warmup_batches:
                admm_denoiser.reset_timing()
                timing_started = True
                print(f"[Table 8] Timing started at batch {step}")

            if timing_started and timing_sample_count >= timing_num_samples:
                print(
                    f"[Table 8] Reached {timing_num_samples} timed samples, stopping."
                )
                break
        # =================================================================

        batch_start_time = time.time()

        latents = bev_model(return_loss=False, only_bev=True, **batch).detach()

        latents = latents.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)

        latents = latents.permute(0, 3, 1, 2)  # 1,256,50,50

        clean_bev = latents.clone().detach() if measure_feature_quality else None

        if noise_timesteps > 0:  # 5
            if noise_timesteps > 1000:
                latents = torch.randn_like(latents)
                latents = latents * noise_scheduler.init_noise_sigma
            else:
                noise = torch.randn_like(latents)
                noise_timesteps = torch.tensor(noise_timesteps).long()
                latents = noise_scheduler.add_noise(
                    latents, noise, noise_timesteps
                )  # 1,256,50,50 #这个latents就是x_t

        if denoise_timesteps > 0:
            cond, uncond = get_condition(batch, use_cond=True), get_condition(
                batch, use_cond=False
            )

            # # DDIM
            noise_scheduler.config.num_train_timesteps = (
                denoise_timesteps  # 5 原本调度器默认1000步去噪，现在定义为5步去噪完毕
            )
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

            for _, t in enumerate(noise_scheduler.timesteps):  # 5，4，3，2，1
                t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)

                # 计算当前时间步的sqrt_alpha_t
                sqrt_alpha_t = torch.sqrt(noise_scheduler.alphas_cumprod[t])
                sqrt_alpha_t = (
                    sqrt_alpha_t.view(-1, 1, 1, 1)
                    .expand(latents.shape[0], 1, 1, 1)
                    .to(latents.device)
                )

                # 两个调用共用相同的sqrt_alpha_t
                denoiser_kwargs = {"timesteps": t_batch, "sqrt_alpha_t": sqrt_alpha_t}

                noise_pred_uncond, noise_pred_cond = admm_denoiser(
                    latents, **denoiser_kwargs, **uncond
                ), admm_denoiser(latents, **denoiser_kwargs, **cond)
                noise_pred = noise_pred_uncond + 2 * (
                    noise_pred_cond - noise_pred_uncond
                )
                # noise_pred = noise_pred_uncond

                classifier_gradient = (
                    get_classifier_gradient(latents, **batch)
                    if use_classifier_guidence
                    else None
                )
                latents = noise_scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    classifier_gradient=classifier_gradient,
                )[0]

        # ============= Table 7：BEV Feature Quality Analysis =============
        if measure_feature_quality and clean_bev is not None:
            # latents 此时是 ADMM-Diff 的 Stage 1 输出 \hat{x}_0，shape [B, 256, 50, 50]
            # clean_bev 是同一个 batch 的训练目标 x_0，shape 一致
            with torch.no_grad():
                pred = latents.float()
                gt = clean_bev.float()
                B = pred.shape[0]

                # 形状检查
                assert (
                    pred.shape == gt.shape
                ), f"[Table 7] pred shape {pred.shape} != gt shape {gt.shape}"

                # ---- per-sample MSE ----
                per_sample_mse = ((pred - gt) ** 2).reshape(B, -1).mean(dim=1)  # [B]

                # ---- per-sample PSNR ----
                # BEV 特征不是 [0, 1] 像素，所以用 GT 自己的最大绝对值作为 dynamic range
                # 这是 per-sample 独立计算，BEVDiffuser baseline 用同样的公式即可保持公平
                gt_max_abs = (
                    gt.reshape(B, -1).abs().max(dim=1).values.clamp(min=1e-8)
                )  # [B]
                eps = 1e-12
                per_sample_psnr = 10.0 * torch.log10(
                    (gt_max_abs**2) / (per_sample_mse + eps)
                )  # [B]

                # ---- per-sample cosine similarity ----
                # 把每个样本的特征展平成长向量再算 cosine
                pred_flat = pred.reshape(B, -1)
                gt_flat = gt.reshape(B, -1)
                per_sample_cos = torch.nn.functional.cosine_similarity(
                    pred_flat, gt_flat, dim=1
                )  # [B]

                # 累加（用 .item() 避免 GPU 内存堆积）
                fq_sum_mse += per_sample_mse.sum().item()
                fq_sum_psnr += per_sample_psnr.sum().item()
                fq_sum_cos += per_sample_cos.sum().item()
                fq_count += B
        # ===================================================================

        # get detection results
        latents = latents.permute(0, 2, 3, 1)
        latents = latents.reshape(-1, bev_cfg.bev_h_ * bev_cfg.bev_w_, bev_cfg._dim_)
        det_result = bev_model(
            return_loss=False, only_bev=False, given_bev=latents, rescale=True, **batch
        )

        if isinstance(det_result, dict):
            if "bbox_results" in det_result.keys():
                bbox_result = det_result["bbox_results"]
                batch_size = len(det_result["bbox_results"])
                bbox_results.extend(bbox_result)
            if (
                "mask_results" in det_result.keys()
                and det_result["mask_results"] is not None
            ):
                mask_result = custom_encode_mask_results(det_result["mask_results"])
                mask_results.extend(mask_result)
                have_mask = True
        else:
            batch_size = len(det_result)
            bbox_results.extend(det_result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

        # 计算并打印每个batch的时间
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        total_time += batch_time
        if rank == 0:  # 只在主进程打印
            print(
                f"Batch {step + 1}/{num_batches} 耗时: {batch_time:.4f} 秒, 平均每个样本: {batch_time / batch_size:.4f} 秒"
            )

        # ============= Table 8：累计已计时样本数（条件执行）=============
        if enable_timing and timing_started:
            timing_sample_count += batch_size
        # ================================================================

    end_time = time.time()
    total_elapsed_time = end_time - start_time

    if rank == 0:  # 只在主进程打印
        print("=" * 50)
        print(f"总测试时间: {total_elapsed_time:.4f} 秒")
        print(f"总batch数: {num_batches}")
        print(f"平均每个batch耗时: {total_elapsed_time / num_batches:.4f} 秒")
        print(f"平均每个样本耗时: {total_elapsed_time / (num_batches * batch_size):.4f} 秒")
        print(f"总forward时间(累加): {total_time:.4f} 秒")
        print("=" * 50)
        # ============= Table 7：报告 feature quality 平均值 =============
        if measure_feature_quality and fq_count > 0:
            avg_mse = fq_sum_mse / fq_count
            avg_psnr = fq_sum_psnr / fq_count
            avg_cos = fq_sum_cos / fq_count
            print("[Table 7] === BEV Feature Quality Result ===")
            print(f"[Table 7] num val samples  = {fq_count}")
            print(
                f"[Table 7] MSE  (lower)     = {avg_mse:.4f}     <-- 填入 Table 7 第 2 列"
            )
            print(
                f"[Table 7] PSNR (higher dB) = {avg_psnr:.2f}    <-- 填入 Table 7 第 3 列"
            )
            print(
                f"[Table 7] Cos  (higher)    = {avg_cos:.4f}     <-- 填入 Table 7 第 4 列"
            )
            print("=" * 50)

        # ============= Table 8：报告 z-update / total 平均耗时 =============
        if enable_timing:
            avg_z_ms, avg_total_ms = admm_denoiser.get_avg_timing()
            regularization_state = admm_denoiser.get_regularization_state()
            print("[Table 8] === Inference Efficiency Result ===")
            print(f"[Table 8] use_sparse_ops    = {admm_denoiser.use_sparse_ops}")
            print(f'[Table 8] rho value         = {regularization_state["rho"]:.4f}')
            print(
                f'[Table 8] sparsity_coef     = {regularization_state["sparsity_coef"]:.4f}'
            )
            print(
                f'[Table 8] sparsity          = {regularization_state["sparsity"]:.4f}'
            )
            print(
                f'[Table 8] num_admm_iters    = {regularization_state["num_admm_iters"]:.0f}'
            )
            print(f"[Table 8] timed samples     = {timing_sample_count}")
            print(f"[Table 8] avg z-update (ms) = {avg_z_ms:.2f}")
            print(f"[Table 8] avg total   (ms) = {avg_total_ms:.2f}")
            print("=" * 50)
        # ===================================================================

    bbox_results = collect_results_cpu(
        bbox_results, len(dataset), tmpdir=os.path.join(save_path, ".dist_test")
    )
    if have_mask:
        mask_results = collect_results_cpu(
            mask_results, len(dataset), tmpdir=os.path.join(save_path, ".dist_test")
        )
    else:
        mask_results = None

    det_results = (
        bbox_results
        if mask_results is None
        else {"bbox_results": bbox_results, "mask_results": mask_results}
    )

    key_score = {}
    if rank == 0:
        eval_kwargs = bev_cfg.get("evaluation", {}).copy()
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
        eval_kwargs["jsonfile_prefix"] = os.path.join(save_path, det_res_path)

        eval_results = dataset.evaluate(det_results, **eval_kwargs)
        for metric, score in eval_results.items():
            if "mAP" in metric or "NDS" in metric:
                key_score[metric] = score
    return key_score


if __name__ == "__main__":
    test()
