#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : main.py
#   Author      : YunYang1994
#   Created date: 2021-09-17 15:27:19
#   Description :
#
#================================================================

import cv2
import dlib
import mesh
import model
import imageio
import numpy as np

camera = {}
bfm = model.MorphabelModel('model/BFM.mat')

#-- 1. 查看均值人脸

# 世界坐标（即人脸坐标系）下的网格
vertices = bfm.data['shapeMU'].reshape(53215, 3) + bfm.data['expMU'].reshape(53215, 3)
colors = bfm.data['texMU'].reshape(53215, 3) / 255.
triangles = bfm.data['tri'].reshape(105840, 3)

# BFM 模型的人脸坐标系 xyz 朝向如下
#     |Y
#     |
#     |__ __ __ X
#    /
#   / Z

# 保存人脸，用 meshLab 打开
mesh.io.write_obj_with_colors("mean_face.obj", vertices, triangles, colors)

#-- 2. 分别看看论文里 shape 的第一个主成分系数为 -5σ 和 +5σ 的效果
exp_para = np.zeros((29, 1))
shape_para = np.zeros((199, 1))
gray_colors = np.ones_like(colors) * (155,155,155) / 255.

for s in [-5, +5]:
    shape_para[0] = s
    vertices_5σ = bfm.generate_vertices(shape_para, exp_para)
    mesh.io.write_obj_with_colors("shape_{}σ_face.obj".format(s), vertices_5σ, triangles, gray_colors)

#-- 3. 在正交投影下，将人脸投影到图片上

### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size

# 相机位置，即外参
camera['eye']          = [0, -200, 200] # the XYZ world space position of the camera.
camera['at']           = [0, 0, 0] # a position along the center of the camera's gaze.
camera['up']           = [0, 1, 0] # up direction: 哪个轴朝上，既可以是 Y 轴也可以是 X、Z 轴

angle, t = mesh.transform.lookat_camera(camera)
print("angle: ", angle, " t: ", t)


# 这里的相机坐标系的 xyz 朝向与人脸坐标系相同
#     |Y
#     |
#     |__ __ __ X
#    /
#   / Z
#
# 上面是加州大学伯克利分校的课程 PPT 里说的「标准的相机坐标系」
# 详见 https://cs184.eecs.berkeley.edu/sp21/lecture/4-64/transforms

# 所以上面计算的 t 是负值，如果 Z 朝向里面的话那就是正值:
#
#            |Y / Z
#            | /
# X __ __ __ |/

# 不过你也可以不同，但要注意:
# 我们是不能通过旋转来改变坐标系的正交性的, 例如不可能变成
#
#  |Y / Z
#  | /
#  |/__ __ __ X


transformed_vertices = bfm.transform(vertices, s, angle, t) # world space to camera space
mesh.io.write_obj_with_colors("mean_face_up_45.obj", transformed_vertices, triangles, colors)

h = 256; w = 256

# using stantard camera & orth projection
projected_vertices = mesh.transform.orthographic_project(transformed_vertices)
image_vertices = mesh.transform.to_image(projected_vertices, h, w) # camera space to image space

color_image = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
imageio.imsave("data/mean_face_up_45.jpg", color_image)

#-- 3. depth map
z = image_vertices[:,2:]
z = (z - np.min(z)) / (np.max(z) - np.min(z))

depth_image = mesh.render.render_colors(image_vertices, triangles, z, h, w, c=1)
imageio.imsave("data/depth_image.jpg", depth_image)

#-- 4. uv coordinates in 'DenseReg: Fully convolutional dense shape regression in-the-wild'. for dense correspondences
uv_coords = bfm.load_uv_coords("model/BFM_UV.mat")
uv_coords_image = mesh.render.render_colors(image_vertices, triangles, uv_coords, h, w, c=2) # two channels: u, v
# add one channel for show
uv_coords_image = np.concatenate((np.zeros((h, w, 1)), uv_coords_image), 2)
imageio.imsave("data/uv_coords_image.jpg", uv_coords_image)

#-- 5. uv texture map
uv_coords = mesh.transform.process_uv(uv_coords, h, w)
uv_texture_map = mesh.render.render_colors(uv_coords, triangles, colors, h, w, c=3)
imageio.imsave("data/uv_texture_map.jpg", uv_texture_map)

#-- 6. pncc in 'Face Alignment Across Large Poses: A 3D Solution'. for dense correspondences
pncc = bfm.load_pncc_code("model/pncc_code.mat")
pncc_image = mesh.render.render_colors(image_vertices, triangles, pncc, h, w, c=3)
imageio.imsave("data/pncc_image.jpg", pncc_image)

#-- 7. fitting: 2D image points and corresponding 3D vertex indices-->  parameters(pose, shape, expression)
## only use 68 key points to fit

# reading image data
image = cv2.imread("./data/jay.jpg")
h, w, c = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

rects = detector(gray, 1)
shape = predictor(gray, rects[0])

# get the face bounding box and landmarks
rects = [(rects[0].tl_corner().x, rects[0].tl_corner().y), (rects[0].br_corner().x, rects[0].br_corner().y)]
landmarks = np.zeros((68, 2))

for i, p in enumerate(shape.parts()):
    landmarks[i] = [p.x, p.y]
    image = cv2.circle(image, (p.x, p.y), radius=2, color=(0, 0, 255), thickness=2)

# we need to change the coordinates of landmarks since the center is the origin of image space.
x = mesh.transform.from_image(landmarks, h, w)
X_ind = bfm.kpt_ind

# fitting process
fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter=200, isShow=False)
fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)

# world space to camera space
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
mesh.io.write_obj_with_colors("fitted_face.obj", transformed_vertices, bfm.triangles, gray_colors)

# camera space to image space
image_vertices = mesh.transform.to_image(transformed_vertices, h, w)
fitted_kpts = mesh.transform.to_image(transformed_vertices[X_ind], h, w)

for kpts in fitted_kpts:
    image = cv2.circle(image, (int(kpts[0]), int(kpts[1])), 2, (0,255,0), 2)

fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
fitted_image = fitted_image*255

alpha = 0.5
rows, cols, dims = np.where(fitted_image != np.array([0,0,0]))
image[rows, cols, dims] = image[rows, cols, dims]*alpha + (1.0-alpha)*fitted_image[rows, cols, dims]

cv2.imwrite('data/jay_fit.png', image)
