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

#============================ Part 1.  导入 smpl 模型数据:

smpl = pickle._Unpickler(open("/Users/yang/models/basicmodel_f_lbs_10_207_0_v1.0.0.pkl", "rb"), encoding='latin1')
smpl = smpl.load()

#============================ Part 2: 显示基模版 v_template 下的人体, 并设置 betas 和 poses 参数
# render(smpl['v_template'], smpl['f'])

# 设置身材参数 betas 和姿态参数 poses
betas = np.random.rand(10) * 0.03
poses = np.random.rand(72) * 0.20

#============================ Part 3: 计算受到 betas 和 pose 参数影响下的 T-pose

# 根据 betas 调整 T-pose, 计算 vertices
v_shaped = smpl['shapedirs'].dot(betas) + smpl['v_template']
# util.render(v_shaped, smpl['f'])

# 根据 poses 臀部的位置， 计算 vertices
J = smpl['J_regressor'].dot(v_shaped)     # 计算 T-pose 下 joints 位置
v_posed = v_shaped + smpl['posedirs'].dot(util.posemap(poses))
# util.render(v_posed, smpl['f'])

#============================ Part 4: 计算受到 betas 和 pose 参数影响下的 transformed-pose

# 根据 poses 调整人体姿态，计算 vertices
poses = poses.reshape((-1,3))     # 将关节点的轴角 (axial-angle) 形状为 [24, 3]

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

# 此时我们需要根据 joints 的旋转矩阵计算出各个 joint 的位置， 由于 betas 参数已
# 经给定，上述早已计算出 T-pose 下的 joints 的位置，因此我们设定每个 joint 的变
# 换矩阵如下：

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
G[3, 3] = 1                         # 齐次坐标，1
Gs[0] = G

for i in range(1,24):
    # 首先计算子节点相对父节点坐标系的位姿
    G = np.zeros([4,4])
    G[:3, :3] = rodrigues(poses[i])
    G[:3, 3]  = J[i] - J[parent[i]] # 子节点位置减去父节点位置，便得到相对位置
    G[3, 3] = 1
    # 然后计算子节点相对世界坐标系的位姿
    Gs[i] = Gs[parent[i]].dot(G) # 乘上其父节点的变换矩阵

# 获得 poses 影响下的 joints 位置, shape = (24, 3)
Jtr = Gs[:, :3, 3]
# 开始蒙皮操作，计算 vertices 坐标
pack = lambda x : np.hstack([np.zeros((4, 3)), x.reshape((4,1))])
Gs = [Gs[i]-pack(Gs[i].dot(np.array([*J[i], 0]))) for i in range(24)]

Gs = np.dstack(Gs)
T = Gs.dot(smpl['weights'].T)
rest_shape_h = np.vstack((v_posed.T, np.ones([1, v_posed.shape[0]])))

v =(T[:,0,:] * rest_shape_h[0, :].reshape((1, -1)) +
    T[:,1,:] * rest_shape_h[1, :].reshape((1, -1)) +
    T[:,2,:] * rest_shape_h[2, :].reshape((1, -1)) +
    T[:,3,:] * rest_shape_h[3, :].reshape((1, -1))).T

v = v[:,:3]
util.render(v, smpl['f'])


from smpl_webuser.serialization import load_model

## Load SMPL model (here we load the female model)
m = load_model( "/Users/yang/models/basicmodel_f_lbs_10_207_0_v1.0.0.pkl" )

## Assign random pose and shape parameters
m.pose[:]  = poses.reshape(-1)
m.betas[:] = betas
util.render(np.array(m), smpl['f'])




