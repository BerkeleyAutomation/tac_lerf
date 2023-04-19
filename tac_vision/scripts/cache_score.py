from PIL import Image
from tacvis.dataset import PREPROC_IMG
from tacvis.capture import DataCaptureUR5,l_p
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import os.path as osp
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from scipy import ndimage
import torch
from skimage.util import view_as_windows
from autolab_core import RigidTransform
import cv2
from typing import List
from tacvis.lightning_modules import ContrastiveModule, RotationModule
from sklearn.manifold import TSNE
from matplotlib.backend_bases import MouseButton
from scipy.spatial import geometric_slerp
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tacvis.dataset import PairedDatasetZoomedOut,collate_paired
from scipy import ndimage
import glob

model_dir = ""
data_dir = ""
tac_size = [128,128]
def preprocess_tac(tac_list):
    '''
    input: list of HxWxC np arrays
    output: Nx3xWxW torch tensor, where N == len(self.rotations)*len(tac_list)
    '''
    tac_batch = [PREPROC_IMG(tac) for tac in tac_list] #cwh
    tac_batch = torch.stack(tac_batch).cuda() #N*c*w*h
    hpad = int(np.clip(max(tac_batch.shape[2],tac_batch.shape[3])-tac_batch.shape[2],0,np.inf)/2)
    wpad = int(np.clip(max(tac_batch.shape[2],tac_batch.shape[3])-tac_batch.shape[3],0,np.inf)/2)
    tac_batch = TF.pad(tac_batch,[wpad,hpad])
    tac_batch = TF.rotate(tac_batch,90)
    tac_batch = TF.resize(tac_batch,tac_size)
    # tac_batches = [TF.rotate(tac_batch,rot) for rot in self.rotations]
    # tac_batches = torch.cat(tac_batches) #num_rot*N dim
    return tac_batch

model = ContrastiveModule.load_from_checkpoint(model_dir,strict=True).eval().cuda()
tac_files = 