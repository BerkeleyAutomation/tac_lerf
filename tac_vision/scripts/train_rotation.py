import os
import torch
# import torch.multiprocessing
import wandb
import yaml
from torch.utils.data import DataLoader
from tacvis.dataset import PairedDatasetRotation, collate_paired, collate_paired_rotation
from tacvis.models import rank_distance, compute_similarity_heatmap
from tacvis.losses import batch_contrastive_loss, l2_norm_loss
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

import argparse
import datetime

yaml_path = 'config/train_rotation.yaml'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=yaml_path)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)
    dataset = PairedDatasetRotation(params)
    train_len = int(params["train_test_split"] * len(dataset))
    val_len = int(len(dataset) - train_len)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])
    print("Train data len", len(train_data))
    batch_size = params["batch_size"]
    eval_batch = params['eval_batch']
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              num_workers=params["num_cores"],
                              shuffle=True, pin_memory=True,
                              drop_last=False, collate_fn=collate_paired_rotation)
    # for batch in train_loader:
    #     rgb = batch['rgb']
    #     tac = batch['tac']
    #     print(rgb.shape)
    #     for i in range(rgb.shape[0]):
    #         print("batch ", i)
    #         print(batch['label'][i, ...])
    #         fig, ax = plt.subplots(1, 2)
    #         ax[0].imshow(rgb[i, ...].numpy().transpose(1, 2, 0))
    #         ax[1].imshow(tac[i, ...].numpy().transpose(1, 2, 0))
    #         plt.show()

    val_loader = DataLoader(val_data, batch_size=eval_batch, num_workers=params['num_cores'],
                            pin_memory=True, drop_last=False, shuffle=False,
                            collate_fn=collate_paired_rotation)
    train_module = RotationModule(params)


    # TODO (albert) remove
    # torch.autograd.set_detect_anomaly(True)
    # for i, data in enumerate(train_loader):
    #     loss = train_module.training_step(data)
    #     print(loss)
    #     loss.backward()
    #
    #     1/0

    # checkpoint = glob.glob('/raid/jkerr/tac_vision/output/contrastive624078/models/*.ckpt')[0]
    # train_module.tac_enc.load_state_dict(torch.load(checkpoint)['state_dict'],strict=True)
    # train_module.rgb_enc.load_state_dict(torch.load(checkpoint)['state_dict'],strict=True)

    # Model directory
    run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    exper_name = params['exper_name']
    wandb_logger = WandbLogger(config=params, entity='luv',
                               project='tacvis-contrastive',
                               name=f'{exper_name}_{run_id}')
    name = wandb_logger.experiment.name
    output_dir = f"output/{name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(yaml_path, output_dir + '/params.yaml')

    # wandb_logger.watch(train_module.tac_enc,log='gradients',log_freq=100)
    # wandb_logger.watch(train_module.rgb_enc,log='gradients',log_freq=100)
    wandb_logger.watch(train_module.enc, log='gradients', log_freq=100)
    # wandb_logger.watch(train_module.MLP,log='gradients',log_freq=100)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min",
                                          dirpath=f"{output_dir}/models")
    # save the yaml file into the output folder too:
    shutil.copy(yaml_path, output_dir + '/params.yaml')
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator='gpu',
        devices=params["devices"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=params["epochs"],
        auto_lr_find=True,
        precision=32,
        amp_backend="native",
        strategy=DDPStrategy(find_unused_parameters=False),
        check_val_every_n_epoch=params['eval_every'],
        log_every_n_steps=1,
        gradient_clip_val=params['grad_clip'],
        accumulate_grad_batches=params['accumulate_gradients'],
        detect_anomaly=False
    )
    trainer.fit(train_module, train_loader, val_loader)
