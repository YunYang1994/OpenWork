#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : transform.py
#   Author      : YunYang1994
#   Created date: 2021-09-16 20:24:27
#   Description :
#
#================================================================

import math
import numpy as np
from .render import render_colors

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,           0,             0],
                 [0, math.cos(x),  -math.sin(x)],
                 [0, math.sin(x),   math.cos(x)]])
    # y
    Ry=np.array([[ math.cos(y), 0, math.sin(y)],
                 [           0, 1,           0],
                 [-math.sin(y), 0, math.cos(y)]])
    # z
    Rz=np.array([[math.cos(z), -math.sin(z), 0],
                 [math.sin(z),  math.cos(z), 0],
                 [          0,            0, 1]])

    R=Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def similarity_transform(vertices, s, R, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3].
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype = np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


def orthographic_project(vertices):
    ''' scaled orthographic projection(just delete z)
        assumes: variations in depth over the object is small relative to the mean distance from camera to object
        x -> x*f/z, y -> x*f/z, z -> f.
        for point i,j. zi~=zj. so just delete z
        ** often used in face
        Homo: P = [[1,0,0,0], [0,1,0,0], [0,0,1,0]]
    Args:
        vertices: [nver, 3]
    Returns:
        projected_vertices: [nver, 3] if isKeepZ=True. [nver, 2] if isKeepZ=False.
    '''
    return vertices.copy()

def perspective_project(vertices, fovy, aspect_ratio = 1., near = 0.1, far = 1000.):
    ''' perspective projection.
    Args:
        vertices: [nver, 3]
        fovy: vertical angular field of view. degree.
        aspect_ratio : width / height of field of view
        near : depth of near clipping plane
        far : depth of far clipping plane
    Returns:
        projected_vertices: [nver, 3]
    '''
    fovy = np.deg2rad(fovy)
    top = near*np.tan(fovy)
    bottom = -top
    right = top*aspect_ratio
    left = -right

    #-- homo
    P = np.array([[near/right, 0, 0, 0],
                 [0, near/top, 0, 0],
                 [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                 [0, 0, -1, 0]])
    vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1)))) # [nver, 4]

    projected_vertices = vertices_homo.dot(P.T)
    projected_vertices = projected_vertices/projected_vertices[:,3:]
    projected_vertices = projected_vertices[:,:3]

    projected_vertices[:,2] = -projected_vertices[:,2]
    return projected_vertices

def to_image(vertices, h, w, is_perspective = False):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]
    '''
    image_vertices = vertices.copy()
    if is_perspective:
        # if perspective, since the projected vertices are normalized to [-1, 1]. so change it to image size first.
        image_vertices[:,0] = image_vertices[:,0]*w/2
        image_vertices[:,1] = image_vertices[:,1]*h/2
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] + w/2
    image_vertices[:,1] = image_vertices[:,1] + h/2
    # flip vertices along y-axis.
    image_vertices[:,1] = h - image_vertices[:,1] - 1  # 在人脸坐标系里，y 轴朝上为正，而图像坐标系是 y 轴朝下为正，因此要反转一下
    return image_vertices


def from_image(vertices, h, w):
    ''' change vertices to image coord system
    3d system: XYZ, center(0, 0, 0)
    2d image: x(u), y(v). center(w/2, h/2), flip y-axis.
    Args:
        vertices: [nver, 3]
        h: height of the rendering
        w : width of the rendering
    Returns:
        projected_vertices: [nver, 3]
    '''
    image_vertices = vertices.copy()
    # move to center of image
    image_vertices[:,0] = image_vertices[:,0] - w/2
    # flip vertices along y-axis.

    image_vertices[:,1] =  h - 1 - image_vertices[:,1]
    image_vertices[:,1] = image_vertices[:, 1] - h / 2
    return image_vertices


def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz


def normalize(x):
    epsilon = 1e-12
    norm = np.sqrt(np.sum(x**2, axis = 0))
    norm = np.maximum(norm, epsilon)
    return x/norm


def lookat_camera(camera):
    """ 'look at' transformation: from world space to camera space
    Xcam = R(X - C)
    Homo: [[R, -RC], [0, 1]]
    Args:
        camera
    Returns:
      transformed_vertices: [nver, 3]
    """

    eye = np.array(camera['eye']).astype(np.float32)
    at = np.array(camera['at']).astype(np.float32)

    z_axis = -normalize(at - eye)
    x_axis = normalize(np.cross(camera['up'], z_axis))
    y_axis = np.cross(z_axis, x_axis)

    # [[R, -RC], [0, 1]]
    R = np.stack((x_axis, y_axis, z_axis))#, axis = 0) # 3 x 3
    t = -R.dot(eye) # 详见 slam 14 讲里的 46 页

    angle = matrix2angle(R)
    return angle, t


def process_uv(uv_coords, h = 256, w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(h - 1)
    uv_coords[:,1] = h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords


def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T; x = x.T
    assert(x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert(n >= 4)

    #--- 1. normalization
    # 2d points
    mean = np.mean(x, 1) # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3,3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1) # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3,:] - X
    average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])

    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine


def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t



