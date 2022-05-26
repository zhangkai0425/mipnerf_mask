import smtpd
import sys
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import shutil

import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt

import time
from plyfile import PlyData, PlyElement
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesAtlas
from pytorch3d.io.obj_io import load_obj, save_obj
import open3d as o3d

from data import MVSDataset
from rasterize_off_line import OffLineRasterizerMachine

class Params(object):
    def __init__(self):
        super(Params, self).__init__()
        self.device = torch.device('cuda')
        self.batch_size = 6
        # input mesh obj dir.
        self.in_path = '/data/guangyu/dataset/GIGAMVS/atlas/erxiaomen_low/512X500/1.obj'
        # foreground mask output folder.
        self.save_path = '/data/guangyu/zhangkai/premask/result/erxiaomen'
        # images and cams base dir.
        self._input_data_rootFld = '/data/guangyu/dataset'
        self._datasetName = 'erxiaomen'
        self.light_condition = '3'
        self.light_condition_list = ['3']  # ['0','1','2','3','4','5','6']
        self.modelName = "9"
        self.datasetFolder = os.path.join(self._input_data_rootFld, 'GIGAMVS')
        self.imgNamePattern = "erxiaomen_low/images/00000#.{}".format('jpg')
        self.poseNamePattern = "erxiaomen_low/cams/00000#_cam.txt"
        # view list for creating corresponding masks.
        self.all_view_list = [i for i in range(25)]

        # original image size.
        self.img_h = 1576
        self.img_w = 2400
        # global down-sample factor, e.g. 1200x1600 --> 600x800 when setting 'compress_ratio_total=2'.
        self.compress_ratio_total = 1

        # below keep fixed.
        self.render_image_size = (
            self.img_h // self.compress_ratio_total,
            self.img_w // self.compress_ratio_total)  # the rendered output size
        self.compress_ratio_h = 1
        self.compress_ratio_w = 1
        self.image_size_single = torch.FloatTensor([[[int(
            self.img_w / (self.compress_ratio_w * self.compress_ratio_total)), int(
            self.img_h / (self.compress_ratio_h * self.compress_ratio_total))]]])  # the size of the input image
        self.faces_per_pixel = 1
        self.blur_radius = 2e-6
        self.z_axis_transform_rate = 1.0

class Masker(object):
    def __init__(self, params):
        super(Masker, self).__init__()
        self.params = params
        self.device = self.params.device
        self.in_path = self.params.in_path

        self.rasterizer_ol = OffLineRasterizerMachine(self.params)
        self.rasterizer_ol.to(self.device)

        self.MVSDataset = MVSDataset(self.params)
        self.cameraPoses = self.MVSDataset.cameraPO4s  # (N_v, 4, 4)
        self.images_ori = self.MVSDataset.imgs_all  # (N_v, H, W, 3)
        self.cameraPositions = self.MVSDataset.cameraTs_new  # (N_v, 3)

        verts, faces, aux = load_obj(f=self.in_path, load_textures=True, create_texture_atlas=False,
                                     device=torch.device("cpu"))
        verts_idx, normals_idx, textures_idx, materials_idx = faces
        self.mesh = Meshes(verts=verts[None, ...], faces=verts_idx[None, ...])

    def extend_attribute(self, attribute, num_view):
        F, FV, C = attribute.shape
        return attribute[None, ...].expand(num_view, -1, -1, -1).reshape(-1, FV, C)

    def forward(self, visualize=False):
        with torch.no_grad():
            batch_size = self.params.batch_size
            save_path = self.params.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            masks = []
            for batch_i in tqdm(range(0, self.cameraPositions.shape[0], batch_size)):
                cameraPoses_batch = self.cameraPoses[batch_i: batch_i + batch_size, ...]
                cameraPositions_batch = self.cameraPositions[batch_i: batch_i + batch_size, ...]
                images_ori_batch = self.images_ori[batch_i: batch_i + batch_size, ...]
                num_batch_view = cameraPositions_batch.shape[0]

                mesh = self.mesh.to(self.device).extend(num_batch_view)

                mask = self.rasterizer_ol(
                    mesh=mesh,
                    matrix=cameraPoses_batch.to(self.device),
                    img_size=self.params.image_size_single.to(self.device),
                    camera_position=cameraPositions_batch.to(self.device)
                )
                for v in range(mask.shape[0]):
                    if visualize:
                        plt.imsave(os.path.join(save_path, '{}.png'.format(batch_i+v)), mask[v, ..., 0].cpu().numpy())
                masks.append(mask[..., 0].cpu())
            masks = torch.cat(masks, dim=0).numpy()
            print(masks.shape)
            np.save(os.path.join(save_path, 'masks.npy'), masks)

if __name__ == "__main__":
    params = Params()
    masker = Masker(params=params)
    masker.forward(visualize=True)


