import os
import torch
import torch.multiprocessing
import wandb
import yaml
from torch.utils.data import DataLoader
from tacvis.dataset import PairedDataset, PairedDatasetZoomedOut, collate_paired
from tacvis.models import  rank_distance, compute_similarity_heatmap
from tacvis.losses import batch_contrastive_loss,l2_norm_loss
import torchvision
import numpy as np
import glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from tacvis.lightning_modules import ContrastiveModule
import shutil

yaml_path = '/home/jkerr/tac_vision/config/train_contrastive.yaml'

if __name__ == "__main__":
    with open(yaml_path, 'r') as stream:
        params = yaml.safe_load(stream)
    assert params['batch_size']%params['repeat_rotations']==0, "batch_size must be divisible by repeat_rotations"
    dataset = PairedDatasetZoomedOut(params)
    
    train_len = int(params["train_test_split"] * len(dataset))
    val_len = int(len(dataset) - train_len)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])

    batch_size = params["batch_size"]
    eval_batch = params['eval_batch']
    repeat = params['repeat_rotations']
    train_loader = DataLoader(train_data, batch_size=batch_size//repeat, num_workers=params["num_cores"], shuffle=True,
                    pin_memory=True, drop_last=False, collate_fn=collate_paired)
    val_loader = DataLoader(val_data, batch_size=eval_batch//repeat, num_workers=params['num_cores'],
                     pin_memory=True, drop_last=False,shuffle=True,collate_fn=collate_paired)
    train_module = ContrastiveModule(params)
    # Model directory
    run_id = np.random.randint(0,1000000)
    wandb_logger = WandbLogger(config=params,entity='luv',project='tacvis-contrastive',name=f'contrastive{run_id}')
    name = wandb_logger.experiment.name
    output_dir = f"/raid/jkerr/tac_vision/output/{name}"
    os.makedirs(output_dir,exist_ok=True)
    shutil.copy(yaml_path, output_dir + '/params.yaml')
    
    
    checkpoint_callback = ModelCheckpoint(monitor="av_rank", mode="min",dirpath=f"{output_dir}/models")
    #save the yaml file into the output folder too:
    shutil.copy(yaml_path, output_dir + '/params.yaml')
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator='gpu',
        devices=params["devices"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=params["epochs"],
        auto_lr_find=True,
        precision=16,
        amp_backend="native",
        strategy=DDPStrategy(find_unused_parameters=False),
        check_val_every_n_epoch=params['eval_every'],
        log_every_n_steps=1,
        gradient_clip_val=params['grad_clip'],
        accumulate_grad_batches = params['accumulate_gradients']
    )
    trainer.fit(train_module, train_loader, val_loader)
