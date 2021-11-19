#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2021 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : morphabel_model.py
#   Author      : YunYang1994
#   Created date: 2021-09-16 17:27:26
#   Description :
#
#================================================================

import mesh
import numpy as np
import scipy.io as sio

from . import fit

class  MorphabelModel(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1]. *
            'shapePC': [3*nver, n_shape_para]. *
            'shapeEV': [n_shape_para, 1]. ~
            'expMU': [3*nver, 1]. ~                       # the mean shape
            'expPC': [3*nver, n_exp_para]. ~              # the principal components
            'expEV': [n_exp_para, 1]. ~                   # standard deviation of each principal components
            'texMU': [3*nver, 1]. ~
            'texPC': [3*nver, n_tex_para]. ~
            'texEV': [n_tex_para, 1]. ~
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """
    def __init__(self, model_path, model_type = 'BFM'):
        super( MorphabelModel, self).__init__()
        if model_type=='BFM':
            self.data = self.load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()

        # fixed attributes
        self.nver = self.data['shapePC'].shape[0]/3
        self.ntri = self.data['tri'].shape[0]
        self.n_shape_para = self.data['shapePC'].shape[1]
        self.n_exp_para = self.data['expPC'].shape[1]
        self.n_tex_para = self.data['texMU'].shape[1]

        self.kpt_ind = self.data['kpt_ind']
        self.triangles = self.data['tri']
        self.full_triangles = np.vstack((self.data['tri'], self.data['tri_mouth']))

    def load_BFM(self, model_path):
        ''' load BFM 3DMM model
        Args:
            model_path: path to BFM model.
        Returns:
            data: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
                'shapeMU': [3*nver, 1]
                'shapePC': [3*nver, 199]
                'shapeEV': [199, 1]
                'expMU': [3*nver, 1]
                'expPC': [3*nver, 29]
                'expEV': [29, 1]
                'texMU': [3*nver, 1]
                'texPC': [3*nver, 199]
                'texEV': [199, 1]
                'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
                'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
                'kpt_ind': [68,] (start from 1)
        PS:
            You can change codes according to your own saved data.
            Just make sure the model has corresponding attributes.
        '''
        C = sio.loadmat(model_path)
        data = C['model'][0,0]

        # change dtype from double(np.float64) to np.float32,
        # since big matrix process(espetially matrix dot) is too slow in python.

        data['shapeMU'] = data['shapeMU'].astype(np.float32) # the mean shape, μ
        data['shapePC'] = data['shapePC'].astype(np.float32) # the principal components, Us
        data['shapeEV'] = data['shapeEV'].astype(np.float32) # standard deviation of each principal components, σs

        data['expMU'] = data['expMU'].astype(np.float32)
        data['expEV'] = data['expEV'].astype(np.float32)
        data['expPC'] = data['expPC'].astype(np.float32)

        # matlab start with 1. change to 0 in python.
        data['tri'] = data['tri'].T.copy(order = 'C').astype(np.int32) - 1
        data['tri_mouth'] = data['tri_mouth'].T.copy(order = 'C').astype(np.int32) - 1

        # kpt ind
        data['kpt_ind'] = (np.squeeze(data['kpt_ind']) - 1).astype(np.int32)
        return data


    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1)
        Returns:
            vertices: (nver, 3)
        '''

        # s(α) = μs + Us diag(σs)α,  详见 paper 公式 (3): A 3D Face Model for Pose and Illumination Invariant Face Recognition
        shape_vertices = self.data['shapeMU'] + self.data['shapePC'].dot(self.data['shapeEV']*shape_para)
        exp_vertices = self.data['expMU'] + self.data['expPC'].dot(self.data['expEV']*exp_para)

        vertices = shape_vertices + exp_vertices
        vertices = np.reshape(vertices, [int(3), int(len(vertices)/3)], 'F').T
        return vertices

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''

        # t(β) = μt + Ut diag(σt)β
        colors = self.data['texMU'] + self.data['texPC'].dot(self.data['texEV']*tex_para)
        colors = np.reshape(colors, [int(3), int(len(colors)/3)], 'F').T/255.
        return colors

    def transform(self, vertices, s, angles, t3d):
        R = mesh.transform.angle2matrix(angles)
        return mesh.transform.similarity_transform(vertices, s, R, t3d)

    def fit(self, x, X_ind, max_iter = 4, isShow = False):
        ''' fit 3dmm & pose parameters
        Args:
            x: (n, 2) image points
            X_ind: (n,) corresponding Model vertex indices
            max_iter: iteration
            isShow: whether to reserve middle results for show
        Returns:
            fitted_sp: (n_sp, 1). shape parameters
            fitted_ep: (n_ep, 1). exp parameters
            s, angles, t
        '''
        if isShow:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points_for_show(x, X_ind, self.data, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = np.zeros((R.shape[0], 3))
            for i in range(R.shape[0]):
                angles[i] = mesh.transform.matrix2angle(R[i])
        else:
            fitted_sp, fitted_ep, s, R, t = fit.fit_points(x, X_ind, self.data, n_sp = self.n_shape_para, n_ep = self.n_exp_para, max_iter = max_iter)
            angles = mesh.transform.matrix2angle(R)

        fitted_sp = fitted_sp / self.data['shapeEV']
        fitted_ep = fitted_ep / self.data['expEV']

        return fitted_sp, fitted_ep, s, angles, t



    def load_uv_coords(self, path = 'BFM_UV.mat'):
        ''' load uv coords of BFM
        Args:
            path: path to data.
        Returns:
            uv_coords: [nver, 2]. range: 0-1
        '''
        C = sio.loadmat(path)
        uv_coords = C['UV'].copy(order = 'C')
        return uv_coords


    def load_pncc_code(self, path = 'pncc_code.mat'):
        ''' load pncc code of BFM
        PNCC code: Defined in 'Face Alignment Across Large Poses: A 3D Solution Xiangyu'
        download at http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm.
        Args:
            path: path to data.
        Returns:
            pncc_code: [nver, 3]
        '''
        C = sio.loadmat(path)
        pncc_code = C['vertex_code'].T
        return pncc_code
