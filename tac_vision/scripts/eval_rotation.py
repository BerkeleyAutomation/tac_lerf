import os
import torch
import torch.multiprocessing
import wandb
import yaml
from torch.utils.data import DataLoader
from tacvis.dataset import PairedDatasetRotation, collate_paired_rotation
from tacvis.models import  rank_distance, compute_similarity_heatmap
from tacvis.losses import batch_contrastive_loss,l2_norm_loss
import torchvision
import numpy as np
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from tacvis.lightning_modules import RotationModule
import shutil
import matplotlib.pyplot as plt

yaml_path = '/home/ravenhuang/tac_vis/tac_vision/config/train_rotation.yaml'

with open('config/test_ur5.yaml', 'r') as stream:
    params = yaml.safe_load(stream)
rot_dir = glob.glob(params["encoder"]["rot_model_dir"]+'*')
rotation_net =  RotationModule.load_from_checkpoint(rot_dir[0],strict=True).eval().cuda()

if __name__ == "__main__":
    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)
    dataset = PairedDatasetRotation(params)
    
    batch_size = params["batch_size"]
    eval_batch = params['eval_batch']

    val_loader = DataLoader(dataset, batch_size=eval_batch, num_workers=params['num_cores'],
                     pin_memory=True, drop_last=False,shuffle=False)
    
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            rgb_batch = data["rgb"].cuda()
            tac_batch = data["tac"].cuda()
            rgb_tac = torch.cat([rgb_batch,tac_batch],dim=1)
            pred = rotation_net.enc(rgb_tac)
            # rgb_feat = rotation_net.rgb_enc(rgb_batch)
            # tac_feat = rotation_net.tac_enc(tac_batch)
            # pred = rotation_net.MLP(torch.cat([rgb_feat,tac_feat],dim=1))
            # import pdb;pdb.set_trace()
            plt.imshow(rgb_batch[0].cpu().numpy().transpose(1,2,0))
            plt.savefig(f'rgb_{i}.png')
            plt.imshow(tac_batch[0].cpu().numpy().transpose(1,2,0))
            plt.savefig(f'tac_{i}.png')
            print(f"results:pred{torch.argmax(pred,1)},{torch.softmax(pred,1)}")

