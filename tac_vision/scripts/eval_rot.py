import os
import torch
import torch.multiprocessing
import wandb
import yaml
from torch.utils.data import DataLoader
from tacvis.dataset import PairedDatasetRotation, collate_paired
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

yaml_path = '/home/jkerr/tac_vision/config/train_rotation.yaml'

with open('config/test_ur5.yaml', 'r') as stream:
    params = yaml.safe_load(stream)
rot_dir = glob.glob(params["encoder"]["rot_model_dir"]+'*')
rotation_net =  RotationModule.load_from_checkpoint(rot_dir[0],strict=False).eval().cuda()

if __name__ == "__main__":
    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)
    dataset = PairedDatasetRotation(params)
    
    train_len = int(0.5 * len(dataset))
    val_len = int(len(dataset) - train_len)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])

    batch_size = params["batch_size"]
    eval_batch = params['eval_batch']
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=params["num_cores"], shuffle=True,
                    pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=eval_batch, num_workers=params['num_cores'],
                     pin_memory=True, drop_last=False,shuffle=True)
    
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            rgb_batch = data["rgb"].cuda()
            tac_batch = data["tac"].cuda()
            rgb_feat = rotation_net.rgb_enc(rgb_batch)
            tac_feat = rotation_net.tac_enc(tac_batch)
            pred = rotation_net.MLP(torch.cat([rgb_feat,tac_feat],dim=1))
            import pdb;pdb.set_trace()
            plt.imshow(rgb_batch[0].cpu().numpy().transpose(1,2,0))
            plt.savefig(f'rgb_{i}.png')
            plt.imshow(tac_batch[0].cpu().numpy().transpose(1,2,0))
            plt.savefig(f'tac_{i}.png')
            print(f"results:gt{torch.argmax(data['label'],1)},pred{torch.argmax(pred,1)}")
