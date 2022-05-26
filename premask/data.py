import open3d as o3d
import pdb
import sys
import random
import os
import imageio
sys.path.append("../")
#########################################################
# from utils.base_util import interpolate_cameras
#########################################################
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from PIL import Image

class MVSDataset(Dataset):
    def __init__(self, params):
        super(MVSDataset, self).__init__()
        self.params = params
        self.init_data()

    def init_data(self):
        self.imgs_all = readAllImages(
            datasetFolder=self.params.datasetFolder,
            datasetName=self.params._datasetName,
            imgNamePattern=self.params.imgNamePattern,
            viewList=self.params.all_view_list,
            model=self.params.modelName,
            light_condition=self.params.light_condition,
        )

        self.cameraPOs, self.cameraPO4s, \
        self.cameraRTO4s, self.cameraKO4s = readCameraP0s_np_all(
            datasetFolder=self.params.datasetFolder,
            datasetName=self.params._datasetName,
            poseNamePatternModels=self.params.poseNamePattern,
            model=self.params.modelName,
            viewList=self.params.all_view_list)

        self.imgs_all, self.cameraPOs, self.cameraPO4s = resize_image_and_matrix(
            images=self.imgs_all, projection_M=self.cameraPOs, compress_ratio=self.params.compress_ratio_total
        )

        self.cameraTs_new = cameraPs2Ts(self.cameraPOs)
        self.cameraPO4s = torch.from_numpy(self.cameraPO4s).type(torch.FloatTensor)         # (N_v, 4, 4)
        # TODO: resize RT and K correspondingly.
        self.cameraRTO4s = torch.from_numpy(self.cameraRTO4s).type(torch.FloatTensor)       # (N_v, 3, 4)
        self.cameraKO4s = torch.from_numpy(self.cameraKO4s).type(torch.FloatTensor)         # (N_v, 3, 3)
        self.cameraTs_new = torch.from_numpy(self.cameraTs_new).type(torch.FloatTensor)     # (N_v, 3)
        self.imgs_all = torch.from_numpy(self.imgs_all).type(torch.FloatTensor)             # (N_v, H, W, 3)

    def create_novel_view(self, cameraRTO4s, cameraKO4s):
        cameraP0s_interpolate, cameraPOs_interpolate, cameraKOs_interpolate, \
        cameraRT4s_interpolate, cameraTs_interpolate = interpolate_cameras(cameraRTO4s[None, ...],
                                                                           cameraKO4s[None, ...],
                                                                           self.params.inter_choose,
                                                                           self.params.zoomin_rate,
                                                                           self.params.interpolate_novel_view_num,
                                                                           direction=self.params.interpolate_direction,
                                                                           zoomin_flag=self.params.inter_zoomin)
        return cameraP0s_interpolate[0], cameraTs_interpolate[0]

def readAllImages(datasetFolder, datasetName, imgNamePattern, viewList, model, light_condition):
    if datasetName == 'DTU':
        imgNamePattern_replace = imgNamePattern.replace('$', str(model)).replace('&', light_condition)
    else:
        imgNamePattern_replace = imgNamePattern
    print("debug heres::",viewList)
    images_list = readImages(
        datasetFolder=datasetFolder,
        imgNamePattern=imgNamePattern_replace,
        viewList=viewList,
        datasetName = datasetName,
        return_list=False)
    return images_list

def readImages(datasetFolder, imgNamePattern, viewList, datasetName, return_list=False):
    imgs_list = []
    for i, viewIndx in enumerate(viewList):
        if datasetName == 'DTU':
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx+1)))
        else:
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx)))
        # img = scipy.misc.imread(imgPath)  # read as np array
        img = np.array(imageio.imread(imgPath))
        # img = img / 256.0 - 0.5
        imgs_list.append(img)
        # print('loaded img ' + imgPath)
    return imgs_list if return_list else np.stack(imgs_list)

def readCameraP0s_np_all(datasetFolder, datasetName, poseNamePatternModels, model, viewList):
    cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(datasetFolder,
                                                           datasetName,
                                                           poseNamePatternModels,
                                                           viewList,
                                                           )
    ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
    cameraP0s = np.concatenate((cameraPOs, ones), axis=1)
    return (cameraPOs, cameraP0s, cameraRTOs, cameraKOs)

def readCameraPOs_as_np(datasetFolder, datasetName, poseNamePattern, viewList):
    cameraPOs = np.empty((len(viewList), 3, 4), dtype=np.float64)
    cameraRTOs = np.empty((len(viewList), 3, 4), dtype=np.float64)
    cameraKOs = np.empty((len(viewList), 3, 3), dtype=np.float64)
    for _i, _view in enumerate(viewList):
        _cameraPO, _cameraRT, _cameraK = readCameraP0_as_np_tanks(
            cameraPO_file=os.path.join(datasetFolder, poseNamePattern.replace('#', '{:03}'.format(_view))),
            datasetName=datasetName
        )
        cameraPOs[_i] = _cameraPO
        cameraRTOs[_i] = _cameraRT
        cameraKOs[_i] = _cameraK
    return cameraPOs, cameraRTOs, cameraKOs

def readCameraP0_as_np_tanks(cameraPO_file, datasetName, ):
    with open(cameraPO_file) as f:
        lines = f.readlines()
    cameraRTO = np.empty((3, 4)).astype(np.float64)
    cameraRTO[0, :] = np.array(lines[1].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[1, :] = np.array(lines[2].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[2, :] = np.array(lines[3].rstrip().split(' ')[:4], dtype=np.float64)

    cameraKO = np.empty((3, 3)).astype(np.float64)
    cameraKO[0, :] = np.array(lines[7].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[1, :] = np.array(lines[8].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[2, :] = np.array(lines[9].rstrip().split(' ')[:3], dtype=np.float64)

    cameraPO = np.dot(cameraKO, cameraRTO)
    return cameraPO, cameraRTO, cameraKO

def cameraPs2Ts(cameraPOs):
    """
    convert multiple POs to Ts.
    ----------
    input:
        cameraPOs: list / numpy
    output:
        cameraTs: list / numpy
    """
    if type(cameraPOs) is list:
        N = len(cameraPOs)
    else:
        N = cameraPOs.shape[0]
    cameraT_list = []
    for _cameraPO in cameraPOs:
        cameraT_list.append(__cameraP2T__(_cameraPO))

    return cameraT_list if type(cameraPOs) is list else np.stack(cameraT_list)

def __cameraP2T__(cameraPO):
    """
    cameraPO: (3,4)
    return camera center in the world coords: cameraT (3,0)
    >>> P = np.array([[798.693916, -2438.153488, 1568.674338, -542599.034996], \
                  [-44.838945, 1433.912029, 2576.399630, -1176685.647358], \
                  [-0.840873, -0.344537, 0.417405, 382.793511]])
    >>> t = np.array([555.64348632032, 191.10837560939, 360.02470478273])
    >>> np.allclose(__cameraP2T__(P), t)
    True
    """
    homo4D = np.array([np.linalg.det(cameraPO[:, [1, 2, 3]]), -1 * np.linalg.det(cameraPO[:, [0, 2, 3]]),
                       np.linalg.det(cameraPO[:, [0, 1, 3]]), -1 * np.linalg.det(cameraPO[:, [0, 1, 2]])])
    # print('homo4D', homo4D)
    cameraT = homo4D[:3] / homo4D[3]
    return cameraT

def resize_image_and_matrix(images,
                            projection_M,
                            return_list=False,
                            compress_ratio=1.0):
    '''
    compress image and garantee the camera position is not changing
    :param images:  all images of one model

    :param projection_M:  camera matrix
        shape: (N_views, 3, 4)
    :param return_list: bool
        if False return the numpy array
    '''
    resized_h = images[0].shape[0] // compress_ratio
    resized_w = images[0].shape[1] // compress_ratio

    compress_w_new, compress_h_new = compress_ratio, compress_ratio
    transform_matrix = np.array([[[1 / compress_w_new, 0, 0], [0, 1 / compress_h_new, 0], [0, 0, 1]]])
    projection_M_new = np.matmul(transform_matrix, projection_M)

    cameraTs = cameraPs2Ts(projection_M)
    cameraTs_new = cameraPs2Ts(projection_M_new)
    trans_vector = (cameraTs - cameraTs_new)[:, :, None]
    identical_matrix = np.repeat(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), cameraTs.shape[0], axis=0)
    bottom_matrix = np.repeat(np.array([[[0, 0, 0, 1]]]), cameraTs.shape[0], axis=0)
    transform_matrix2 = np.concatenate((identical_matrix, trans_vector), axis=2)
    transform_matrix2 = np.concatenate((transform_matrix2, bottom_matrix), axis=1)
    projection_M_new_f = np.concatenate((projection_M_new, bottom_matrix), axis=1)

    projection_M_new = np.matmul(transform_matrix2, projection_M_new_f)
    cameraPOs = projection_M_new[:, :3, :]
    ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=projection_M_new.shape[0], axis=0)
    cameraP04s = np.concatenate((cameraPOs, ones), axis=1)

    image_resized_list = []
    for i in range(images.shape[0]):
        # image_resized = scipy.misc.imresize(images[i], size=(resized_h, resized_w), interp='bicubic')
        image_resized = np.array(Image.fromarray(images[i].astype(np.uint8)).resize((resized_w, resized_h), Image.BICUBIC))
        image_resized = image_resized / 256.0 - 0.5
        image_resized_list.append(image_resized)
    images_resized = image_resized_list if return_list else np.stack(image_resized_list)
    return images_resized, cameraPOs, cameraP04s

if __name__ == "__main__":
    params = Params()
    mvs = MVSDataset(params=params)