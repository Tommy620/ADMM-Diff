# debug_import.py
# import sys
# print("Python path:")
# for p in sys.path:
#     print(f"  {p}")
#
# import mmdet.core.bbox.assigners as assigners_module
#
# original_register = assigners_module.BBOX_ASSIGNERS._register_module
#
# def traced_register(name=None, module=None, force=False):
#     import traceback
#     print(f"\nRegistering: {name}")
#     print("Stack trace:")
#     for line in traceback.format_stack()[:-1]:
#         if 'register_module' in line or 'site-packages' not in line:
#             print(line.strip())
#     return original_register(name=name, module=module, force=force)
#
# assigners_module.BBOX_ASSIGNERS._register_module = traced_register
# # 测试导入
# print("\n尝试导入...")
# try:
#     from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
#     print("导入成功")
# except Exception as e:
#     print(f"导入失败: {e}")
#
import torch

a = torch.eye(3, device='cuda')
print(torch.linalg.inv(a))