import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import numpy as np
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.structures.utils import list_to_packed
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,
    SfMOrthographicCameras,
    OrthographicCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    PointsRasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizer,
    #TexturedSoftPhongShader,
    SoftPhongShader,
    SoftGouraudShader
)

from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.points.rasterizer import PointFragments
from pytorch3d.renderer.mesh.utils import _clip_barycentric_coordinates, _interpolate_zbuf
from pytorch3d.renderer.mesh.shading import flat_shading, gouraud_shading, _apply_lighting
#from pytorch3d.renderer.mesh.texturing import interpolate_face_attributes
from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.renderer.blending import BlendParams
# add path for demo utils functions
import sys
import os

import time
import scipy.misc
import matplotlib.pyplot as plt
import pdb
from math import *

class OffLineFragmentMachine():
    def __init__(self, params):
        super(OffLineFragmentMachine, self).__init__()
        self.params = params

    def generate_attribute(self, fragment):
        self.pix_to_face = fragment.pix_to_face
        self.zbuf = fragment.zbuf
        self.dists = fragment.dists
        self.bary_coords = fragment.bary_coords

class OffLineRasterizerMachine(nn.Module):
    def __init__(self, params):
        super(OffLineRasterizerMachine, self).__init__()
        self.params = params

        self.device = self.params.device
        self.R = torch.FloatTensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])[None, ...]
        self.T = torch.FloatTensor([[0, 0, 0]])
        self.focal_length = torch.FloatTensor([[1, 1]])
        self.principal_point = torch.FloatTensor([[0.0, 0.0]])

        self.fragment = OffLineFragmentMachine(self.params)

        self.cameras = SfMOrthographicCameras(device=self.device,
                                              focal_length=self.focal_length,
                                              principal_point=self.principal_point,
                                              R=self.R,
                                              T=self.T)  # defined in NDC space.

        self.raster_settings = RasterizationSettings(
            image_size=self.params.render_image_size,
            blur_radius=self.params.blur_radius,
            faces_per_pixel=self.params.faces_per_pixel,
        )

        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings
        )

    def forward(self, mesh, matrix, img_size, camera_position):

        verts = mesh.verts_packed()  # (V, 3) --> (x, y, z) coordinates of each vertex
        faces = mesh.faces_packed()  # (F, 3) --> indices of the 3 vertices in 'verts' which form the triangular face.

        self.faces_verts = verts[faces]  # (F,3,3)

        mesh = self.transform_points_mvs(representation=mesh, matrix=matrix, img_size=img_size)
        self.matrix = matrix  # camera projection matrix P0.

        self.update_rasterizer()

        fragments = self.rasterizer(mesh)

        # sample fake textures (i.e. embedding index for each triangle facet.)
        # embedding_id_texels = mesh.sample_textures(fragments)[..., 0].type(torch.cuda.FloatTensor).to(self.device)

        self.bary_coords = fragments.bary_coords
        if self.params.blur_radius > 0.0:
            # TODO: potentially move barycentric clipping to the rasterizer
            # if no downstream functions requires unclipped values.
            # This will avoid unnecssary re-interpolation of the z buffer.

            # self.dists = fragments.dists
            clipped_bary_coords = _clip_barycentric_coordinates(fragments.bary_coords)

            # clipped_bary_coords = fragments.bary_coords
            clipped_zbuf = _interpolate_zbuf(
                fragments.pix_to_face, clipped_bary_coords, mesh
            )

            fragments = Fragments(
                bary_coords=clipped_bary_coords,
                # bary_coords=bary_coords,
                zbuf=clipped_zbuf,
                dists=fragments.dists,
                pix_to_face=fragments.pix_to_face,
            )

        mask = (fragments.pix_to_face >= 0.0)
        return mask

    def update_rasterizer(self):

        render_image_size = self.params.render_image_size
        self.render_image_size = render_image_size
        self.raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=self.params.blur_radius,
            faces_per_pixel=self.params.faces_per_pixel,
        )

        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings
        )

    def transform_points_mvs(self, representation, matrix, img_size):
        '''
        use camera matrix to transform the original global mesh verts
        (keep the dimension of z axis, that means the transformed camera is a SfMOrthographicCameras)
        args:
            mesh: pytorch3d mesh object
            matrix: cameras matrix
                shape:(N_views, 4, 4)
                type: torch.FloatTensor
            img_size: the image size of output image
                shape: (1,1,2)
                type: torch.FloatTensor

        output:
            pytorch3d mesh object
        '''
        mesh = representation
        verts = mesh.verts_padded() # shape: (N_v, max_num_verts, 3)
        verts_old_packed = mesh.verts_packed() # shape: (V, 3)

        N, P, _ = verts.shape
        ones = torch.ones(N, P, 1, dtype=verts.dtype, device=verts.device)
        verts = torch.cat([verts, ones], dim=2) # shape: (N_v, max_num_verts, 4)

        verts = torch.matmul(matrix[:, None, :, :], verts[..., None]) # shape: (N_v, max_num_verts, 4, 1)

        verts = torch.cat((verts[:, :, 0:2, 0] / verts[:, :, 2:3, 0],
                           verts[:, :, 2:3, 0] / self.params.z_axis_transform_rate),
                          dim=2)  # shape: (N_v, max_num_verts, 3)

        W = img_size[0][0][0].item()
        H = img_size[0][0][1].item()

        verts_ndc = torch.zeros(verts.shape, dtype=verts.dtype, device=verts.device)
        verts_ndc[..., 0] = - W / H + (2.0 * W / H * verts[..., 0] + W / H) / W
        verts_ndc[..., 1] = -1.0 + (2.0 * verts[..., 1] + 1.0) / H
        verts_ndc[..., 2] = verts[..., 2]

        return mesh.offset_verts(verts_ndc.view(-1, 3) - verts_old_packed)
