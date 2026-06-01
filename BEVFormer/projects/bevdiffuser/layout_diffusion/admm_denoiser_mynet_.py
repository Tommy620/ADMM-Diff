import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import distutils.version
from abc import abstractmethod
from scipy.io import loadmat
from os.path import join
from .layout_encoder import LayoutTransformerEncoder, xf_convert_module_to_f16
import os
import torch
import safetensors
import copy
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


def _inverse_softplus(value):
    value = torch.as_tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(value))


# 相较ADMM-Diff-ori，共 7 处实质性修改：

# 输出改为残差连接（最关键）

# 旧：final_output = self.out_layer(final_x0)
# 新：final_output = final_x0 + self.out_layer(final_x0)
# 含义：旧版输出完全由 out_layer 决定，而 out_layer 是 zero-module（初始权重为 0），会把 ADMM 求出的解析解 final_x0 整个抹掉，等于白算；新版让解析解走捷径直通，out_layer 只学残差修正。
# rho 强制为正

# 旧：rho = nn.Parameter(torch.tensor([0.1]))（可被优化成负数，数值不稳定）
# 新：rho = _inverse_softplus([0.1])，forward 用 F.softplus(rho).clamp(min=1e-6) 保证恒正。
# gamma 强制为正 + 改初值

# 旧：gamma = nn.Parameter(torch.tensor(0.0))（初值 0，软阈值不起作用，且可变负）
# 新：gamma = _inverse_softplus(1e-4)，forward 用 F.softplus(gamma) 恒正。
# 软阈值公式修正

# 旧：阈值直接用 self.gamma（|Rv| - gamma）
# 新：阈值用 threshold = gamma / rho（|Rv| - threshold）——这才是 ADMM 软阈值算子的理论正确形式。
# ADMM 迭代次数可配置

# 旧：for i in range(4) 写死 4 轮
# 新：for i in range(self.num_admm_iters)，新增 num_admm_iters（默认 4，不传时行为不变，可用于速度/精度折中）。
# 去掉 up/down sample 模块构建

# 旧：__init__ 里无条件构建 downsample_blocks/upsample_blocks（各 2 层）
# 新：整段注释掉。tiny 本来就不用（use_up_down_sample=False），无影响；但注意：新版若传 --use_up_down_sample（base/v2）会因找不到这些 block 而报错。
# 稀疏度记录方式 + 新增诊断接口

# 旧：每步 print 稀疏收敛
# 新：存到 self.last_sparsity，并新增 get_regularization_state() / _positive_regularizers() 供训练日志读取 rho/gamma/threshold/sparsity。
# 一句话总结改动方向：修了三个数值/结构上的硬伤——把 rho/gamma 约束为正、修正软阈值为 gamma/rho、让输出走残差（避免 zero-module 抹掉解析解），并把迭代数变成可调。前三项直接关系到"
# 长 iters 训练能否稳定成长"，正是你上一轮关心的问题。


# Ori-Version
# class ADMMDenoiser(nn.Module): #总共有三种图：精细图，稀疏图和残缺图
#     def __init__(
#         self,
#         in_channels: int = 256,
#         out_channels: int = 256,
#         kernel_size: int = 5

#     ):
#         """
#         Args:

#         """
#         super(ADMMDenoiser, self).__init__()

#         self.downsample_blocks = nn.ModuleList([])
#         self.upsample_blocks = nn.ModuleList([])
#         self.use_up_down_sample = False

#         for _ in range(2):  # num_pre_downsample = 0 # 2 for base and v2
#             self.downsample_blocks.append(Downsample(
#                 in_channels, True, dims=2, out_channels=in_channels
#             ))  # in_channels=256; conv_resample=True; dims=2
#             self.upsample_blocks.append(Upsample(
#                 out_channels, True, dims=2, out_channels=out_channels
#             ))  # out_channels=256


#         self.model_channels = in_channels
#         time_embed_dim = 1024
#         self.time_embed = nn.Sequential(
#             linear(256, time_embed_dim),
#             SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )
#         self.layout_encoder = LayoutTransformerEncoder(used_condition_types=['obj_class', 'obj_bbox', 'is_valid_obj'],
#                 layout_length=300,
#                 num_classes_for_layout_object=12,
#                 mask_size_for_layout_object=0,
#                 hidden_dim=256,
#                 output_dim=1024, # model_channels x 4
#                 num_layers=6,
#                 num_heads=8,
#                 use_final_ln=True,
#                 use_positional_embedding=False,
#                 resolution_to_attention=[12, 25, 50], #[ 8, 16, 32 ],
#                 use_key_padding_mask=False,
#                 use_3d_bbox=True)

#         self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True) #约束强度，作用于s.t. 归零式之前


#         self.R_struct = RStructTransform(
#             in_channels=256,
#             hidden_channels=512
#         )

#         self.R_struct_inverse = RStructInverseTransform(
#             in_channels=512,
#             out_channels=256
#         )

#         self.gamma = nn.Parameter(torch.tensor(0.0))
#         # self.gamma = torch.tensor(0.0) # ablation study for " gamma stay as 0 "
#         self.out_layer = Out_Layer()

#         #for making first fig, after that, you can note these codes.
#         self.x_history = []  # 记录内部迭代的x
#         self.z_history = []  # 记录内部迭代的z
#         self.reset_history()

#         # ============= Table 8 实验所需新增字段（不启用时全部空转）=============
#         # enable_timing: 总开关，外部不传 --enable_timing 时为 False，
#         #                所有 cuda.synchronize 和 perf_counter 调用都会跳过
#         # use_sparse_ops: Table 8 中 "+ sparse ops" 行的开关
#         self.enable_timing = False
#         self.use_sparse_ops = False

#         # 计时累计器（仅在 enable_timing=True 时被写入）
#         self.z_update_time_ms = 0.0
#         self.total_forward_time_ms = 0.0
#         self.timing_count = 0
#         # =====================================================================

#     # for making first fig, after that, you can note these codes.
#     def reset_history(self):
#         """重置历史记录"""
#         self.x_history = []
#         self.z_history = []

#     def forward(self, x, timesteps, sqrt_alpha_t=None, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, obj_name=None, last_scheduler_iter = False, **kwargs): #模型输入x：已经稀疏化后的图像（Fx） 所以这个模型本质上就是学会从Fx_gt到x_gt的映射，学会如何从稀疏到稠密。

#         # for making first fig, after that, you can note these codes.
#         # 只在最后一次外部迭代时记录（由外部控制）

#         if last_scheduler_iter:
#             self.reset_history()

#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

#         layout_outputs = self.layout_encoder(
#             obj_class=obj_class,
#             obj_bbox=obj_bbox,
#             obj_mask=obj_mask,
#             is_valid_obj=is_valid_obj,
#             obj_name=obj_name
#         )
#         xf_proj, xf_out = layout_outputs["xf_proj"], layout_outputs["xf_out"]
#         # xf_proj (1,1024) xf_out(1,256,300)
#         emb_combined = emb + xf_proj.to(emb)


#         # 计算 α_t
#         alpha_bar_t = sqrt_alpha_t ** 2  # α_t

#         # ============= Table 8 计时：forward 总时间起点（条件执行）=============
#         if self.enable_timing:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             forward_start = time.perf_counter()
#         # =====================================================================

#         h = x.type(torch.float32)
#         if self.use_up_down_sample:
#             for module in self.downsample_blocks:
#                 h = module(h)

#         # ========== ADMM初始化 ==========
#         # x_0 初始化（第一个闭式解）
#         x_0 = (2 * sqrt_alpha_t * h) / (2 * alpha_bar_t + self.rho)
#         # x_0 (1,256,50,50)
#         z = torch.zeros_like(x_0)  # 辅助变量
#         u = torch.zeros_like(x_0)  # 拉格朗日乘子
#         sparsity_list = []
#         # ========== ADMM迭代 ==========
#         for i in range(4):
#             # 1. x-子问题：数据保真项更新（闭式解）
#             # x^{k+1} = argmin_x 数据保真项 + (ρ/2)||x - z^k + u^k/ρ||²
#             x_0 = (2 * sqrt_alpha_t * h - u + self.rho * z) / (2 * alpha_bar_t + self.rho) #1,256,50,50

#             # 2. z-子问题：先验项更新（神经网络近似）
#             # z^{k+1} = argmin_z 先验项(z) + (ρ/2)||x^{k+1} - z + u^k/ρ||²
#             # 用神经网络近似这个优化问题
#             v = x_0 + u / (self.rho + 1e-8)

#             # ============= Table 8 计时：z-update 起点（条件执行）=============
#             if self.enable_timing:
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 z_start = time.perf_counter()
#             # =================================================================


#             # 闭式解：z = Rᵀ[soft_threshold(Rv, γ/ρ)]
#             # 步骤1: R变换
#             # ===== R_struct forward =====
#             Rv = self.R_struct(v, emb_combined, layout_outputs)

#             # ===== soft threshold =====
#             w_sparse = torch.sign(Rv) * torch.clamp(
#                 torch.abs(Rv) - self.gamma,
#                 min=0
#             )

#             # ============= Table 8: sparse ops 开关（条件执行）=============
#             # 启用时把 |w_sparse| < 1e-6 的位置显式置零并构造紧凑张量
#             if self.use_sparse_ops:
#                 sparse_mask = (torch.abs(w_sparse) >= 1e-6)
#                 w_sparse = (w_sparse * sparse_mask).contiguous()
#             # ================================================================

#             # ===== inverse transform =====
#             z = self.R_struct_inverse(w_sparse, emb_combined, layout_outputs) # 1, 256, 50, 50

#             # ============= Table 8 计时：z-update 终点（条件执行）=============
#             if self.enable_timing:
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 self.z_update_time_ms += (time.perf_counter() - z_start) * 1000.0
#             # =================================================================


#             # 3. u-子问题：乘子更新
#             # u^{k+1} = u^k + ρ(x^{k+1} - z^{k+1})
#             u = u + self.rho * (x_0 - z)

#             sparsity = (torch.abs(w_sparse) < 1e-6).float().mean().item()
#             sparsity_list.append(sparsity)

#             # for making first fig, after that, you can note these codes.
#             # ==== 记录中间结果（只在需要记录时）====
#             if last_scheduler_iter:
#                 # detach()并转移到CPU保存
#                 self.x_history.append(x_0.detach().cpu())
#                 self.z_history.append(z.detach().cpu())

#         if len(sparsity_list) > 1:
#             print(f"[Sparsity Convergence] {sparsity_list[0] * 100:.1f}% → {sparsity_list[-1] * 100:.1f}%")


#         # ========== 最终输出 ==========
#         # 经过ADMM迭代后的x_0估计
#         final_x0 = x_0

#         final_output = self.out_layer(final_x0)

#         # ============= Table 8 计时：forward 总时间终点（条件执行）=============
#         if self.enable_timing:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             self.total_forward_time_ms += (time.perf_counter() - forward_start) * 1000.0
#             self.timing_count += 1
#         # =====================================================================

#         if self.use_up_down_sample:
#             for module in self.upsample_blocks:
#                 final_output = module(final_output)

#         return (final_output, list(self.x_history), list(self.z_history)) if last_scheduler_iter else final_output


#     def save_pretrained(self, save_directory):
#         if os.path.isfile(save_directory):
#             print(f"Provided path ({save_directory}) should be a directory, not a file")
#             return

#         os.makedirs(save_directory, exist_ok=True)
#         weights_name = SAFETENSORS_WEIGHTS_NAME
#         safetensors.torch.save_file(self.state_dict(), os.path.join(save_directory, weights_name), metadata={"format": "pt"})
#     # 跑test_bev_diffuser的时候就用这个，按照文件夹范式加载ckpt，训练时方便用下面的。
#     def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
#         weights_name = SAFETENSORS_WEIGHTS_NAME
#         if os.path.isfile(pretrained_model_name_or_path):
#             checkpoint_file = pretrained_model_name_or_path
#         elif os.path.isdir(pretrained_model_name_or_path):
#             if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
#                 checkpoint_file = os.path.join(pretrained_model_name_or_path, weights_name)
#             elif subfolder is not None and os.path.isfile(
#             os.path.join(pretrained_model_name_or_path, subfolder, weights_name)):
#                 checkpoint_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
#         else:
#             print(f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.")
#             return
#         state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
#         try:
#             self.load_state_dict(state_dict, strict=True)
#             print('successfully load the entire model')
#         except:
#             print('not successfully load the entire model, try to load part of model')
#             self.load_state_dict(state_dict, strict=False)

#     # ============= Table 8 计时：辅助方法 =============
#     def reset_timing(self):
#         self.z_update_time_ms = 0.0
#         self.total_forward_time_ms = 0.0
#         self.timing_count = 0

#     def get_avg_timing(self):
#         if self.timing_count == 0:
#             return 0.0, 0.0
#         return (self.z_update_time_ms / self.timing_count,
#                 self.total_forward_time_ms / self.timing_count)
#     # =====================================================


# V1-Version
# class ADMMDenoiser(nn.Module): #总共有三种图：精细图，稀疏图和残缺图
#     def __init__(
#         self,
#         in_channels: int = 256,
#         out_channels: int = 256,
#         kernel_size: int = 5,
#         num_admm_iters: int = 4
#
#     ):
#         """
#         Args:
#
#         """
#         super(ADMMDenoiser, self).__init__()
#
#         # self.downsample_blocks = nn.ModuleList([])
#         # self.upsample_blocks = nn.ModuleList([])
#         self.use_up_down_sample = False
#         self.num_admm_iters = num_admm_iters
#
#         # for _ in range(2):  # num_pre_downsample = 0 # 2 for base and v2
#         #     self.downsample_blocks.append(Downsample(
#         #         in_channels, True, dims=2, out_channels=in_channels
#         #     ))  # in_channels=256; conv_resample=True; dims=2
#         #     self.upsample_blocks.append(Upsample(
#         #         out_channels, True, dims=2, out_channels=out_channels
#         #     ))  # out_channels=256
#
#
#
#         self.model_channels = in_channels
#         time_embed_dim = 1024
#         self.time_embed = nn.Sequential(
#             linear(256, time_embed_dim),
#             SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )
#         self.layout_encoder = LayoutTransformerEncoder(used_condition_types=['obj_class', 'obj_bbox', 'is_valid_obj'],
#                 layout_length=300,
#                 num_classes_for_layout_object=12,
#                 mask_size_for_layout_object=0,
#                 hidden_dim=256,
#                 output_dim=1024, # model_channels x 4
#                 num_layers=6,
#                 num_heads=8,
#                 use_final_ln=True,
#                 use_positional_embedding=False,
#                 resolution_to_attention=[12, 25, 50], #[ 8, 16, 32 ],
#                 use_key_padding_mask=False,
#                 use_3d_bbox=True)
#
#         self.rho = nn.Parameter(_inverse_softplus([0.1]), requires_grad=True) # 约束强度，forward 中用 softplus 保证为正
#
#
#
#         self.R_struct = RStructTransform(
#             in_channels=256,
#             hidden_channels=512
#         )
#
#         self.R_struct_inverse = RStructInverseTransform(
#             in_channels=512,
#             out_channels=256
#         )
#
#         self.gamma = nn.Parameter(_inverse_softplus(1e-4))
#         # self.gamma = torch.tensor(0.0) # ablation study for " gamma stay as 0 "
#         self.out_layer = Out_Layer()
#         self.last_sparsity = 0.0
#
#         #for making first fig, after that, you can note these codes.
#         self.x_history = []  # 记录内部迭代的x
#         self.z_history = []  # 记录内部迭代的z
#         self.reset_history()
#
#         # ============= Table 8 实验所需新增字段（不启用时全部空转）=============
#         # enable_timing: 总开关，外部不传 --enable_timing 时为 False，
#         #                所有 cuda.synchronize 和 perf_counter 调用都会跳过
#         # use_sparse_ops: Table 8 中 "+ sparse ops" 行的开关
#         self.enable_timing = False
#         self.use_sparse_ops = False
#
#         # 计时累计器（仅在 enable_timing=True 时被写入）
#         self.z_update_time_ms = 0.0
#         self.total_forward_time_ms = 0.0
#         self.timing_count = 0
#         # =====================================================================
#
#     # for making first fig, after that, you can note these codes.
#     def reset_history(self):
#         """重置历史记录"""
#         self.x_history = []
#         self.z_history = []
#
#     def _positive_regularizers(self):
#         rho = F.softplus(self.rho).clamp(min=1e-6)
#         gamma = F.softplus(self.gamma)
#         threshold = gamma / rho
#         return rho, gamma, threshold
#
#     def get_regularization_state(self):
#         rho, gamma, threshold = self._positive_regularizers()
#         return {
#             "rho": rho.detach().item(),
#             "gamma": gamma.detach().item(),
#             "threshold": threshold.detach().item(),
#             "sparsity": float(self.last_sparsity),
#             "num_admm_iters": float(self.num_admm_iters),
#         }
#
#     def forward(self, x, timesteps, sqrt_alpha_t=None, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, obj_name=None, last_scheduler_iter = False, **kwargs): #模型输入x：已经稀疏化后的图像（Fx） 所以这个模型本质上就是学会从Fx_gt到x_gt的映射，学会如何从稀疏到稠密。
#
#         # for making first fig, after that, you can note these codes.
#         # 只在最后一次外部迭代时记录（由外部控制）
#
#         if last_scheduler_iter:
#             self.reset_history()
#
#         emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
#
#         layout_outputs = self.layout_encoder(
#             obj_class=obj_class,
#             obj_bbox=obj_bbox,
#             obj_mask=obj_mask,
#             is_valid_obj=is_valid_obj,
#             obj_name=obj_name
#         )
#         xf_proj, xf_out = layout_outputs["xf_proj"], layout_outputs["xf_out"]
#         # xf_proj (1,1024) xf_out(1,256,300)
#         emb_combined = emb + xf_proj.to(emb)
#
#
#         # 计算 α_t
#         alpha_bar_t = sqrt_alpha_t ** 2  # α_t
#         rho, gamma, threshold = self._positive_regularizers()
#
#         # ============= Table 8 计时：forward 总时间起点（条件执行）=============
#         if self.enable_timing:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             forward_start = time.perf_counter()
#         # =====================================================================
#
#         h = x.type(torch.float32)
#         if self.use_up_down_sample:
#             for module in self.downsample_blocks:
#                 h = module(h)
#
#         # ========== ADMM初始化 ==========
#         # x_0 初始化（第一个闭式解）
#         x_0 = (2 * sqrt_alpha_t * h) / (2 * alpha_bar_t + rho)
#         # x_0 (1,256,50,50)
#         z = torch.zeros_like(x_0)  # 辅助变量
#         u = torch.zeros_like(x_0)  # 拉格朗日乘子
#         sparsity_list = []
#         # ========== ADMM迭代 ==========
#         for i in range(self.num_admm_iters):
#             # 1. x-子问题：数据保真项更新（闭式解）
#             # x^{k+1} = argmin_x 数据保真项 + (ρ/2)||x - z^k + u^k/ρ||²
#             x_0 = (2 * sqrt_alpha_t * h - u + rho * z) / (2 * alpha_bar_t + rho) #1,256,50,50
#
#             # 2. z-子问题：先验项更新（神经网络近似）
#             # z^{k+1} = argmin_z 先验项(z) + (ρ/2)||x^{k+1} - z + u^k/ρ||²
#             # 用神经网络近似这个优化问题
#             v = x_0 + u / (rho + 1e-8)
#
#             # ============= Table 8 计时：z-update 起点（条件执行）=============
#             if self.enable_timing:
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 z_start = time.perf_counter()
#             # =================================================================
#
#
#             # 闭式解：z = Rᵀ[soft_threshold(Rv, γ/ρ)]
#             # 步骤1: R变换
#             # ===== R_struct forward =====
#             Rv = self.R_struct(v, emb_combined, layout_outputs)
#
#             # ===== soft threshold =====
#             w_sparse = torch.sign(Rv) * torch.clamp(
#                 torch.abs(Rv) - threshold,
#                 min=0
#             )
#
#             # ============= Table 8: sparse ops 开关（条件执行）=============
#             # 启用时把 |w_sparse| < 1e-6 的位置显式置零并构造紧凑张量
#             if self.use_sparse_ops:
#                 sparse_mask = (torch.abs(w_sparse) >= 1e-6)
#                 w_sparse = (w_sparse * sparse_mask).contiguous()
#             # ================================================================
#
#             # ===== inverse transform =====
#             z = self.R_struct_inverse(w_sparse, emb_combined, layout_outputs) # 1, 256, 50, 50
#
#             # ============= Table 8 计时：z-update 终点（条件执行）=============
#             if self.enable_timing:
#                 if torch.cuda.is_available():
#                     torch.cuda.synchronize()
#                 self.z_update_time_ms += (time.perf_counter() - z_start) * 1000.0
#             # =================================================================
#
#
#             # 3. u-子问题：乘子更新
#             # u^{k+1} = u^k + ρ(x^{k+1} - z^{k+1})
#             u = u + rho * (x_0 - z)
#
#             sparsity = (torch.abs(w_sparse) < 1e-6).float().mean().item()
#             sparsity_list.append(sparsity)
#
#             # for making first fig, after that, you can note these codes.
#             # ==== 记录中间结果（只在需要记录时）====
#             if last_scheduler_iter:
#                 # detach()并转移到CPU保存
#                 self.x_history.append(x_0.detach().cpu())
#                 self.z_history.append(z.detach().cpu())
#
#         if sparsity_list:
#             self.last_sparsity = sparsity_list[-1]
#
#
#         # ========== 最终输出 ==========
#         # 经过ADMM迭代后的x_0估计
#         final_x0 = x_0
#
#         final_output = final_x0 + self.out_layer(final_x0)
#
#         # ============= Table 8 计时：forward 总时间终点（条件执行）=============
#         if self.enable_timing:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             self.total_forward_time_ms += (time.perf_counter() - forward_start) * 1000.0
#             self.timing_count += 1
#         # =====================================================================
#
#         if self.use_up_down_sample:
#             for module in self.upsample_blocks:
#                 final_output = module(final_output)
#
#         return (final_output, list(self.x_history), list(self.z_history)) if last_scheduler_iter else final_output
#
#
#     def save_pretrained(self, save_directory):
#         if os.path.isfile(save_directory):
#             print(f"Provided path ({save_directory}) should be a directory, not a file")
#             return
#
#         os.makedirs(save_directory, exist_ok=True)
#         weights_name = SAFETENSORS_WEIGHTS_NAME
#         safetensors.torch.save_file(self.state_dict(), os.path.join(save_directory, weights_name), metadata={"format": "pt"})
#     # 跑test_bev_diffuser的时候就用这个，按照文件夹范式加载ckpt，训练时方便用下面的。
#     def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
#         weights_name = SAFETENSORS_WEIGHTS_NAME
#         if os.path.isfile(pretrained_model_name_or_path):
#             checkpoint_file = pretrained_model_name_or_path
#         elif os.path.isdir(pretrained_model_name_or_path):
#             if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
#                 checkpoint_file = os.path.join(pretrained_model_name_or_path, weights_name)
#             elif subfolder is not None and os.path.isfile(
#             os.path.join(pretrained_model_name_or_path, subfolder, weights_name)):
#                 checkpoint_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
#         else:
#             print(f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.")
#             return
#         state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
#         try:
#             self.load_state_dict(state_dict, strict=True)
#             print('successfully load the entire model')
#         except:
#             print('not successfully load the entire model, try to load part of model')
#             self.load_state_dict(state_dict, strict=False)
#
#     # ============= Table 8 计时：辅助方法 =============
#     def reset_timing(self):
#         self.z_update_time_ms = 0.0
#         self.total_forward_time_ms = 0.0
#         self.timing_count = 0
#
#     def get_avg_timing(self):
#         if self.timing_count == 0:
#             return 0.0, 0.0
#         return (self.z_update_time_ms / self.timing_count,
#                 self.total_forward_time_ms / self.timing_count)
#     # =====================================================


# V2-Version
# 相较 V1 的改动（V1 已整体注释于上方，处理方式与 ori 一致），共 3 处：
#   1) 撤销 V1 的残差输出，回退到 ori：final_output = self.out_layer(final_x0)
#      —— V1 用 final_x0 + out_layer(final_x0)，会把"未收敛、被数据项粗暴缩放过"的解析解
#         final_x0 当默认输出，逼网络去纠偏一个偏置很大的先验，破坏了 ori 里 zero-init
#         输出层"从零稳定学习"的动力学（实测 V1 < ori，这是回退的关键一步）。
#   2) prior 网络按迭代解绑（deep unrolling，主增容杠杆）：
#      R_struct / R_struct_inverse 由"一套权重在 4 次迭代里复用"改为"4 套独立权重"(nn.ModuleList)，
#      第 i 次迭代用第 i 套。参数量约 4 倍；但前向次数仍是 4 次、每次网络规模不变，
#      故 FLOPs / 推理时间基本不变，只是参数 / 梯度 / 优化器状态显存增大。
#   3) rho、gamma 由"全局单标量"改为"按迭代各一组可学习标量"（形状 (K,)），
#      保留 V1 的 softplus 正性参数化（数值安全，ADMM 要求 rho>0，且 rho 在分母里不能取负）；
#      阈值回退到 ori 形式 threshold_i = gamma_i（去掉 V1 的 gamma/rho —— gamma_i 既已按迭代
#      独立可学，再除以 rho 只是冗余耦合）。
#   其余（layout_encoder、time_embed、Out_Layer 的 zero-init、save/load、计时与 history、
#   forward 接口）全部与 V1 保持一致。
class ADMMDenoiser(nn.Module):  # 总共有三种图：精细图，稀疏图和残缺图
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: int = 5,
        num_admm_iters: int = 4,
    ):
        """
        Args:

        """
        super(ADMMDenoiser, self).__init__()

        # self.downsample_blocks = nn.ModuleList([])
        # self.upsample_blocks = nn.ModuleList([])
        self.use_up_down_sample = False
        self.num_admm_iters = num_admm_iters

        self.model_channels = in_channels
        time_embed_dim = 1024
        self.time_embed = nn.Sequential(
            linear(256, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.layout_encoder = LayoutTransformerEncoder(
            used_condition_types=["obj_class", "obj_bbox", "is_valid_obj"],
            layout_length=300,
            num_classes_for_layout_object=12,
            mask_size_for_layout_object=0,
            hidden_dim=256,
            output_dim=1024,  # model_channels x 4
            num_layers=6,
            num_heads=8,
            use_final_ln=True,
            use_positional_embedding=False,
            resolution_to_attention=[12, 25, 50],  # [ 8, 16, 32 ],
            use_key_padding_mask=False,
            use_3d_bbox=True,
        )

        # 改动3：按迭代各配一组 rho/gamma（可学习标量，softplus 保证为正）。
        # 形状均为 (num_admm_iters,)，第 i 次迭代取 rho[i] / gamma[i]。
        self.rho = nn.Parameter(_inverse_softplus(torch.full((num_admm_iters,), 0.1)))
        self.gamma = nn.Parameter(
            _inverse_softplus(torch.full((num_admm_iters,), 1e-4))
        )

        # 改动2：prior 网络按迭代解绑，4 次迭代各用一套独立权重（deep unrolling）。
        self.R_struct = nn.ModuleList(
            [
                RStructTransform(in_channels=256, hidden_channels=512)
                for _ in range(num_admm_iters)
            ]
        )
        self.R_struct_inverse = nn.ModuleList(
            [
                RStructInverseTransform(in_channels=512, out_channels=256)
                for _ in range(num_admm_iters)
            ]
        )

        self.out_layer = Out_Layer()
        self.last_sparsity = 0.0

        # for making first fig, after that, you can note these codes.
        self.x_history = []  # 记录内部迭代的x
        self.z_history = []  # 记录内部迭代的z
        self.reset_history()

        # ============= Table 8 实验所需新增字段（不启用时全部空转）=============
        self.enable_timing = False
        self.use_sparse_ops = False
        self.z_update_time_ms = 0.0
        self.total_forward_time_ms = 0.0
        self.timing_count = 0
        # =====================================================================

    # for making first fig, after that, you can note these codes.
    def reset_history(self):
        """重置历史记录"""
        self.x_history = []
        self.z_history = []

    def _positive_regularizers(self):
        # 返回按迭代的 (K,) 向量
        rho = F.softplus(self.rho).clamp(min=1e-6)
        gamma = F.softplus(self.gamma)
        threshold = gamma  # 阈值回退到 ori 形式：threshold_i = gamma_i
        return rho, gamma, threshold

    def get_regularization_state(self):
        rho, gamma, threshold = self._positive_regularizers()
        state = {}
        for i in range(self.num_admm_iters):
            state[f"rho_{i}"] = rho[i].detach().item()
            state[f"gamma_{i}"] = gamma[i].detach().item()
            state[f"threshold_{i}"] = threshold[i].detach().item()
        state["sparsity"] = float(self.last_sparsity)
        state["num_admm_iters"] = float(self.num_admm_iters)
        return state

    def forward(
        self,
        x,
        timesteps,
        sqrt_alpha_t=None,
        obj_class=None,
        obj_bbox=None,
        obj_mask=None,
        is_valid_obj=None,
        obj_name=None,
        last_scheduler_iter=False,
        **kwargs,
    ):  # 模型输入x：已经稀疏化后的图像（Fx） 所以这个模型本质上就是学会从Fx_gt到x_gt的映射，学会如何从稀疏到稠密。
        # for making first fig, after that, you can note these codes.
        if last_scheduler_iter:
            self.reset_history()

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        layout_outputs = self.layout_encoder(
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj,
            obj_name=obj_name,
        )
        xf_proj, xf_out = layout_outputs["xf_proj"], layout_outputs["xf_out"]
        emb_combined = emb + xf_proj.to(emb)

        # 计算 α_t
        alpha_bar_t = sqrt_alpha_t**2  # α_t
        rho, gamma, threshold = self._positive_regularizers()  # 均为 (K,)

        # ============= Table 8 计时：forward 总时间起点（条件执行）=============
        if self.enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
        # =====================================================================

        h = x.type(torch.float32)
        if self.use_up_down_sample:
            for module in self.downsample_blocks:
                h = module(h)

        # ========== ADMM初始化 ==========
        # x_0 初始化（第一个闭式解），用第 0 次迭代的 rho
        x_0 = (2 * sqrt_alpha_t * h) / (2 * alpha_bar_t + rho[0])
        z = torch.zeros_like(x_0)  # 辅助变量
        u = torch.zeros_like(x_0)  # 拉格朗日乘子
        sparsity_list = []
        # ========== ADMM迭代（每次迭代用各自的 rho_i/gamma_i 与各自的 prior 权重）==========
        for i in range(self.num_admm_iters):
            rho_i = rho[i]
            threshold_i = threshold[i]

            # 1. x-子问题：数据保真项更新（闭式解）
            x_0 = (2 * sqrt_alpha_t * h - u + rho_i * z) / (2 * alpha_bar_t + rho_i)

            # 2. z-子问题：先验项更新（神经网络近似的近端算子）
            v = x_0 + u / (rho_i + 1e-8)

            # ============= Table 8 计时：z-update 起点（条件执行）=============
            if self.enable_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                z_start = time.perf_counter()
            # =================================================================

            # 闭式解：z = Rᵀ[soft_threshold(Rv, threshold_i)]，第 i 套权重
            Rv = self.R_struct[i](v, emb_combined, layout_outputs)

            # ===== soft threshold =====
            w_sparse = torch.sign(Rv) * torch.clamp(torch.abs(Rv) - threshold_i, min=0)

            # ============= Table 8: sparse ops 开关（条件执行）=============
            if self.use_sparse_ops:
                sparse_mask = torch.abs(w_sparse) >= 1e-6
                w_sparse = (w_sparse * sparse_mask).contiguous()
            # ================================================================

            # ===== inverse transform =====
            z = self.R_struct_inverse[i](w_sparse, emb_combined, layout_outputs)

            # ============= Table 8 计时：z-update 终点（条件执行）=============
            if self.enable_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.z_update_time_ms += (time.perf_counter() - z_start) * 1000.0
            # =================================================================

            # 3. u-子问题：乘子更新
            u = u + rho_i * (x_0 - z)

            sparsity = (torch.abs(w_sparse) < 1e-6).float().mean().item()
            sparsity_list.append(sparsity)

            # for making first fig, after that, you can note these codes.
            if last_scheduler_iter:
                self.x_history.append(x_0.detach().cpu())
                self.z_history.append(z.detach().cpu())

        if sparsity_list:
            self.last_sparsity = sparsity_list[-1]

        # ========== 末轮 x 闭式更新 ==========
        # 必要性：prior 解绑后（改动2），循环里 x_0 是在每轮"开头"用上一轮的 z/u 算的，
        # 因此最后一轮的 R_struct[K-1]/R_struct_inverse[K-1] 产出的 z/u 不会进入 final_x0=x_0，
        # 这套权重对 loss 无贡献，DDP 会报 "parameters not used in producing loss"。
        # 这里补做一次 x 闭式解（纯解析、无网络），让最后一轮的 z/u 进入最终解，所有 prior 权重均参与 loss。
        rho_last = rho[self.num_admm_iters - 1]
        x_0 = (2 * sqrt_alpha_t * h - u + rho_last * z) / (2 * alpha_bar_t + rho_last)

        # ========== 最终输出（改动1：回退到 ori，不再加残差）==========
        final_x0 = x_0
        final_output = self.out_layer(final_x0)

        # ============= Table 8 计时：forward 总时间终点（条件执行）=============
        if self.enable_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.total_forward_time_ms += (time.perf_counter() - forward_start) * 1000.0
            self.timing_count += 1
        # =====================================================================

        if self.use_up_down_sample:
            for module in self.upsample_blocks:
                final_output = module(final_output)

        return (
            (final_output, list(self.x_history), list(self.z_history))
            if last_scheduler_iter
            else final_output
        )

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        weights_name = SAFETENSORS_WEIGHTS_NAME
        safetensors.torch.save_file(
            self.state_dict(),
            os.path.join(save_directory, weights_name),
            metadata={"format": "pt"},
        )

    # 跑test_bev_diffuser的时候就用这个，按照文件夹范式加载ckpt，训练时方便用下面的。
    def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
        weights_name = SAFETENSORS_WEIGHTS_NAME
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(
                os.path.join(pretrained_model_name_or_path, weights_name)
            ):
                checkpoint_file = os.path.join(
                    pretrained_model_name_or_path, weights_name
                )
            elif subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            ):
                checkpoint_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, weights_name
                )
        else:
            print(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
            return
        state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
        try:
            self.load_state_dict(state_dict, strict=True)
            print("successfully load the entire model")
        except:
            print("not successfully load the entire model, try to load part of model")
            self.load_state_dict(state_dict, strict=False)

    # ============= Table 8 计时：辅助方法 =============
    def reset_timing(self):
        self.z_update_time_ms = 0.0
        self.total_forward_time_ms = 0.0
        self.timing_count = 0

    def get_avg_timing(self):
        if self.timing_count == 0:
            return 0.0, 0.0
        return (
            self.z_update_time_ms / self.timing_count,
            self.total_forward_time_ms / self.timing_count,
        )

    # =====================================================


class SiLU(nn.Module):  # export-friendly version of SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, dic):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        out = x + h

        return out


class ObjectAwareCrossAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,  # 256
        num_heads=1,  # 8
        num_head_channels=-1,  # 32
        use_checkpoint=False,  # F
        encoder_channels=None,  # 256
        return_attention_embeddings=False,  # F
        ds=None,  # 1
        resolution=None,  # 50
        type=None,  #'input'
        use_positional_embedding=True,  # T
        use_key_padding_mask=False,  # F
        channels_scale_for_positional_embedding=1.0,  # 1
        norm_first=False,  # F
    ):
        super().__init__()
        self.norm_for_obj_embedding = None
        self.norm_first = norm_first
        self.channels_scale_for_positional_embedding = (
            channels_scale_for_positional_embedding
        )
        self.use_key_padding_mask = use_key_padding_mask
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings

        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_positional_embedding = use_positional_embedding
        assert self.use_positional_embedding

        self.use_checkpoint = use_checkpoint

        self.qkv_projector = conv_nd(1, channels, 3 * channels, 1)
        self.norm_for_qkv = normalization(channels)

        if encoder_channels is not None:  # 256
            self.encoder_channels = encoder_channels
            self.layout_content_embedding_projector = conv_nd(
                1, encoder_channels, channels * 2, 1
            )
            self.layout_position_embedding_projector = conv_nd(
                1, encoder_channels, int(channels), 1
            )

            self.norm_for_obj_class_embedding = normalization(encoder_channels)
            self.norm_for_layout_positional_embedding = normalization(int(channels))
            self.norm_for_image_patch_positional_embedding = normalization(
                int(channels)
            )

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, layout):
        """
        :param x: (N, C, H, W)
        :param cond_kwargs['xf_out']: (N, C, L2)
        :return:
            extra_output: N x L2 x 3 x ds x ds
        """

        cond_kwargs = layout

        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)  # N x C x (HxW)

        qkv = self.qkv_projector(self.norm_for_qkv(x))  # N x 3C x L1, 其中L1=H*W
        bs, C, L1, L2 = (
            qkv.shape[0],
            self.channels,
            qkv.shape[2],
            cond_kwargs["obj_bbox_embedding"].shape[-1],
        )

        # positional embedding for image patch

        image_patch_positional_embedding = self.layout_position_embedding_projector(
            cond_kwargs[
                "image_patch_bbox_embedding_for_resolution{}".format(self.resolution)
            ]
        )  # N x C * channels_scale_for_positional_embedding x L1, 其中L1=H*W
        image_patch_positional_embedding = (
            self.norm_for_image_patch_positional_embedding(
                image_patch_positional_embedding
            )
        )  # (N, C * channels_scale_for_positional_embedding, L1)
        image_patch_positional_embedding = image_patch_positional_embedding.reshape(
            bs * self.num_heads,
            int(C * self.channels_scale_for_positional_embedding) // self.num_heads,
            L1,
        )  # (N * num_heads, C * channels_scale_for_positional_embedding // num_heads, L1)

        # content embedding for image patch
        (
            q_image_patch_content_embedding,
            k_image_patch_content_embedding,
            v_image_patch_content_embedding,
        ) = qkv.split(
            C, dim=1
        )  # 3 x (N , C, L1)
        q_image_patch_content_embedding = q_image_patch_content_embedding.reshape(
            bs * self.num_heads, C // self.num_heads, L1
        )  # (N // num_heads, C // num_heads, L1)
        k_image_patch_content_embedding = k_image_patch_content_embedding.reshape(
            bs * self.num_heads, C // self.num_heads, L1
        )  # (N // num_heads, C // num_heads, L1)
        v_image_patch_content_embedding = v_image_patch_content_embedding.reshape(
            bs * self.num_heads, C // self.num_heads, L1
        )  # (N // num_heads, C // num_heads, L1)

        # embedding for image patch
        q_image_patch = torch.cat(
            [q_image_patch_content_embedding, image_patch_positional_embedding], dim=1
        )  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1)
        k_image_patch = torch.cat(
            [k_image_patch_content_embedding, image_patch_positional_embedding], dim=1
        )  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1)
        v_image_patch = (
            v_image_patch_content_embedding  # (N // num_heads, C // num_heads, L1)
        )

        # positional embedding for layout

        layout_positional_embedding = self.layout_position_embedding_projector(
            cond_kwargs["obj_bbox_embedding"]
        )  # N x C*channels_scale_for_positional_embedding x L2
        layout_positional_embedding = self.norm_for_layout_positional_embedding(
            layout_positional_embedding
        )  # (N, C * channels_scale_for_positional_embedding, L2)
        layout_positional_embedding = layout_positional_embedding.reshape(
            bs * self.num_heads,
            int(C * self.channels_scale_for_positional_embedding) // self.num_heads,
            L2,
        )  # (N // num_heads, channels_scale_for_positional_embedding * C // num_heads, L2)

        # content embedding for layout

        layout_content_embedding = (
            cond_kwargs["xf_out"]
            + self.norm_for_obj_class_embedding(cond_kwargs["obj_class_embedding"])
        ) / 2
        (
            k_layout_content_embedding,
            v_layout_content_embedding,
        ) = self.layout_content_embedding_projector(layout_content_embedding).split(
            C, dim=1
        )  # 2 x (N x C x L2)
        k_layout_content_embedding = k_layout_content_embedding.reshape(
            bs * self.num_heads, C // self.num_heads, L2
        )  # (N // num_heads, C // num_heads, L2)
        v_layout_content_embedding = v_layout_content_embedding.reshape(
            bs * self.num_heads, C // self.num_heads, L2
        )  # (N // num_heads, C // num_heads, L2)

        # embedding for layout
        k_layout = torch.cat(
            [k_layout_content_embedding, layout_positional_embedding], dim=1
        )  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L2)
        v_layout = v_layout_content_embedding  # (N // num_heads, C // num_heads, L2)

        #  mix embedding for cross attention
        k_mix = torch.cat(
            [k_image_patch, k_layout], dim=2
        )  # (N // num_heads, (1+channels_scale_for_positional_embedding) * C // num_heads, L1+L2)
        v_mix = torch.cat(
            [v_image_patch, v_layout], dim=2
        )  # (N // num_heads, 1 * C // num_heads, L1+L2)

        scale = 1 / math.sqrt(
            math.sqrt(
                int((1 + self.channels_scale_for_positional_embedding) * C)
                // self.num_heads
            )
        )
        attn_output_weights = torch.einsum(
            "bct,bcs->bts", q_image_patch * scale, k_mix * scale
        )  # More stable with f16 than dividing afterwards, (N x num_heads, L1, L1+L2)

        attn_output_weights = attn_output_weights.view(bs, self.num_heads, L1, L1 + L2)

        attn_output_weights = attn_output_weights.view(bs * self.num_heads, L1, L1 + L2)

        attn_output_weights = torch.softmax(attn_output_weights.float(), dim=-1).type(
            attn_output_weights.dtype
        )  # (N x num_heads, L1, L1+L2)

        attn_output = torch.einsum(
            "bts,bcs->bct", attn_output_weights, v_mix
        )  # (N x num_heads, C // num_heads, L1)
        attn_output = attn_output.reshape(bs, C, L1)  # (N, C, L1)

        #
        h = self.proj_out(attn_output)

        output = (x + h).reshape(b, c, *spatial)

        return output


class Out_Layer(nn.Module):
    def __init__(self):
        super(Out_Layer, self).__init__()
        self.seq = nn.Sequential(
            normalization(256),
            SiLU(),
            zero_module(conv_nd(2, 256, 256, 3, padding=1)),
        )

    def forward(self, x):
        return self.seq(x)


class RStructTransform(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.res1 = ResBlock(
            256,
            1024,
            0,
            out_channels=256,
            dims=2,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )
        self.res2 = ResBlock(
            256,
            1024,
            0,
            out_channels=256,
            dims=2,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )

        self.attn = ObjectAwareCrossAttention(
            256,
            use_checkpoint=False,
            num_heads=8,
            num_head_channels=32,
            encoder_channels=256,
            ds=1,
            resolution=50,
            type="input",
            use_positional_embedding=True,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
        )

        self.proj = nn.Conv2d(in_channels, hidden_channels, 1)

    def forward(self, x, embed, cond):
        x = self.res1(x, embed)
        x = self.attn(x, cond)
        x = self.res2(x, embed)
        x = self.proj(x)
        return x


class RStructInverseTransform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels, 1)

        self.attn = ObjectAwareCrossAttention(
            256,
            use_checkpoint=False,
            num_heads=8,
            num_head_channels=32,
            encoder_channels=256,
            ds=1,
            resolution=50,
            type="input",
            use_positional_embedding=True,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
        )

        self.res1 = ResBlock(
            256,
            1024,
            0,
            out_channels=256,
            dims=2,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )
        self.res2 = ResBlock(
            256,
            1024,
            0,
            out_channels=256,
            dims=2,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )

    def forward(self, x, embed, cond):
        x = self.proj(x)
        x = self.res2(x, embed)
        x = self.attn(x, cond)
        x = self.res1(x, embed)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, out_size=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.out_size = out_size
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.out_size is None:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(
                    x, (x.shape[2], self.out_size, self.out_size), mode="nearest"
                )
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=self.out_size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)
