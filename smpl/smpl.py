#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : smpl.py
#   Author      : YunYang1994
#   Created date: 2021-01-05 16:11:14
#   Description :
#
#================================================================

import cv2
import util
import pickle
import numpy as np


smpl = pickle._Unpickler(open("/Users/yang/models/basicmodel_f_lbs_10_207_0_v1.0.0.pkl", "rb"), encoding='latin1')
smpl = smpl.load()      # smpl 是一个字典，关键的 key 如下：

# J_regressor: (24, 6890), 与 vertices (6890, 3) 相乘边得到 joints 位置 (24, 3)
# f: (13776, 3)，faces，我们说到 mesh 除了有 vertices 组成，还有一个 triangle list，每个 triangle 由三个 vertices index 组成
# kintree_table: (2, 24)，一般取第一行，这就是上面提到的每个点的父节点
# weights: (6890, 24), blend weights, 定义了顶点受每个 joint 旋转矩阵影响的权重
# shapedirs: (6890, 3, 10), 表示体型参数到 shape blend shape 的映射关系
# posedirs: (6890, 3, 207), 表示姿势参数到 pose blend shape 的映射关系
# v_template: (6890, 3), 人体基模版的 vertices

#============================ Part 1: 显示基模版 v_template 下的人体, 并设置 betas 和 poses 参数, 见论文 figure 3(a)
# render(smpl['v_template'], smpl['f'])

# 设置身材参数 betas 和姿态参数 poses
betas = np.random.rand(10) * 0.03
poses = np.random.rand(72) * 0.20

#============================ Part 2: 计算受到 betas 和 pose 参数影响下的 T-pose, 见论文 figure 3(b)

# 根据 betas 调整 T-pose, 计算 vertices
v_shaped = smpl['shapedirs'].dot(betas) + smpl['v_template']
# util.render(v_shaped, smpl['f'])


#============================ Part 3: 根据 poses 调整臀部的位置, 见论文 figure 3(c)

J = smpl['J_regressor'].dot(v_shaped)     # 计算 T-pose 下 joints 位置
v_posed = v_shaped + smpl['posedirs'].dot(util.posemap(poses))   # 计算受 pose 影响下调整臀部之后的 vertices
# util.render(v_posed, smpl['f'])

#============================ Part 4: 计算受到 betas 和 pose 参数影响下的 transformed-pose, 见论文 figure 3(d)

# 将关节点的轴角 (axial-angle) 形状为 [24, 3]
poses = poses.reshape((-1,3))

# 定义 SMPL 的关节树, 开始骨骼绑定操作
id_to_col = {smpl['kintree_table'][1,i] : i for i in range(smpl['kintree_table'].shape[1])}
parent = {i : id_to_col[smpl['kintree_table'][0,i]] for i in range(1, smpl['kintree_table'].shape[1])}

# parent: {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
# parent 定义了 SMPL 的关节树, 如 1 的父节点为 0, 4 的父节点为 1, 等等
# 如下所示:
#---------------------------------------- 关节树 --------------------------------------#
# Pelvis(0)
# |-- L_Hip(1)
#     |-- L_Knee(4)
#         |-- L_Ankle(7)
#             |-- L_Foot(10)
# |-- R_Hip(2)
#     |-- R_Knee(5)
#         |-- R_Ankle(8)
#             |-- R_Foot(11)
# |-- Spine1(3)
#     |-- Spine2(6)
#         |-- Spine3(9)
#             |-- Neck(12)
#                 |-- Head(15)
#             |-- L_Collar(13)
#                 |-- L_Shoulder(16)
#                 |-- L_Elbow(18)
#                     |-- L_Wrist(20)
#                         |-- L_Hand(22)
#             |-- R_Collar(14)
#                 |-- R_Shoulder(17)
#                 |-- R_Elbow(19)
#                     |-- R_Wrist(21)
#                         |-- R_Hand(23)
#
# 此时我们需要根据 joints 的旋转矩阵计算出各个 joint 的位置， 由于 betas 参数已
# 经给定，上述早已计算出 T-pose 下的 joints 的位置，因此我们设定每个 joint 的变
# 换矩阵如下：
#
#                       | 旋转矩阵 | 关节点位置 |
#                   G = --------------------------
#                       |    0     |     1      |
#
# G 的形状为 4x4 ，和 slam 里相机位姿矩阵的概念是一致的, 下面分别计算各个 joint

rodrigues = lambda x: cv2.Rodrigues(x)[0]
Gs = np.zeros([24,4,4])

# 首先计算根结点 (0) 的世界坐标变换, 或者说是根结点相对世界坐标系的位姿
G = np.zeros([4,4])
G[:3, :3] = rodrigues(poses[0])     # 旋转矩阵
G[:3, 3] = J[0]                     # 关节点位置
G[3, 3] = 1                         # 齐次矩阵，1
Gs[0] = G

# 计算子节点 (1~24) 的世界坐标变换
for i in range(1,24):
    # 首先计算子节点相对父节点坐标系的位姿
    G = np.zeros([4,4])
    G[:3, :3] = rodrigues(poses[i]) # 子节点相对父节点的旋转矩阵
    G[:3, 3]  = J[i] - J[parent[i]] # 子节点位置减去父节点位置，便得到相对位置
    G[3, 3] = 1
    # 然后计算子节点相对世界坐标系的位姿
    Gs[i] = Gs[parent[i]].dot(G) # 乘上其父节点的变换矩阵

# 获得 poses 影响下的 joints 位置, shape = (24, 3)
Jtr = Gs[:, :3, 3]

# 开始蒙皮操作，计算 vertices 坐标

# 计算相对 rest pose 时的位姿矩阵，值得注意的是 rest pose 时的旋转矩阵为单位矩
# 阵，因此旋转矩阵不变，因而只需要减去 rest pose 时的关节点位置就行

for i in range(24):
    R = Gs[i][:3, :3]                           # 相对 rest pose 的旋转矩阵
    t = Gs[i][:3, 3] - R.dot(J[i])              # 相对 rest pose 的节点位置
    Gs[i][:3, 3] = t

# 计算每个节点的旋转矩阵对每个顶点的权重, (6890, 4, 4)
T = smpl['weights'].dot(Gs.reshape([24,16])).reshape([-1,4,4])

# 将 v_posed 变成齐次坐标矩阵 (6890, 4)
v_posed_homo = np.vstack((v_posed.T, np.ones([1, v_posed.shape[0]])))

# 权重 T 与顶点 v_posed_homo 相乘
v = np.matmul(T, v_posed_homo.T.reshape([-1, 4, 1]))
v = v.reshape([-1, 4])[:,:3]    # 由于是齐次矩阵，取前3列
util.render(v, smpl['f'])
