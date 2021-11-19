#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : util.py
#   Author      : YunYang1994
#   Created date: 2021-01-05 17:37:13
#   Description :
#
#================================================================


import cv2
import trimesh
import pyrender
import numpy as np

def posemap(p):
    p = p.ravel()[3:]   # 跳过根结点
    return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()

def render(vertices, faces, joints=None):
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.6]
    tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)

    # adding body meshs
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)


    # adding body joints
    if joints is not None:
        sm = trimesh.creation.uv_sphere(radius=0.007)
        sm.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

    # adding obeserving camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 5])
    scene.add(camera, pose=camera_pose)

    pyrender.Viewer(scene, use_raymond_lighting=True)


