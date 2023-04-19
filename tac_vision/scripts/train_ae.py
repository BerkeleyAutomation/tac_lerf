from math import gamma
import os
import torch
import torch.multiprocessing
import tqdm
import wandb
import yaml
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tacvis.dataset import PairedDataset, PairedDatasetAE, PairedDatasetZoomedOut
from tacvis.models import AutoEncoder, DEVICE, UNetAE,Encoder
import matplotlib.pyplot as plt
import torchvision
from itertools import chain


def train_ae(epochs, model,dataset, optimizer, scheduler, params, output_dir, type):
    train_len = int(params["train_test_split"] * len(dataset))
    val_len = int(len(dataset) - train_len)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])

    batch_size = params["batch_size"]
    eval_batch = params['eval_batch']
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=params["num_cores"], shuffle=True,
                    pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=eval_batch, num_workers=params["num_cores"],
                     pin_memory=True, drop_last=False)

    for epoch in epochs:
        model.train()
        epoch_loss = 0

        n=0
        for data in train_loader:
            n+=1
            optimizer.zero_grad()
            if 'rgb' in type:
                batch = data["rgb"].to(DEVICE)
            else:
                batch = data['tac'].to(DEVICE)
            loss,rec,feat = model.loss(batch)
            
            loss.backward()
            epoch_loss += loss.detach().cpu()
            optimizer.step()
                
        wandb.log({'train_loss':epoch_loss/n})
        
        scheduler.step()
        if epoch%params['save_every']==0:
            model.encoder.save(f'{output_dir}/model_encoder_{epoch}.pt')
            model.decoder.save(f'{output_dir}/model_decoder_{epoch}.pt')
            model.save(f'{output_dir}/model_{epoch}.pt')
            
        
        if epoch%params['eval_every']==0:
            model.eval()
            val_loss=0
            with torch.no_grad():
                n=0
                for data in val_loader:
                    n+=1
                    if 'rgb' in type:
                        batch = data["rgb"].to(DEVICE)
                    else:
                        batch = data['tac'].to(DEVICE)
                    loss,rec,feat = model.loss(batch)
                    val_loss+=loss.detach().cpu()

                    if epoch%10 ==0 and n==1:
                        recon_images = wandb.Image(rec)
                        wandb.log({'recon_images': recon_images})
                        gt_images = wandb.Image(batch)
                        wandb.log({'gt_images': gt_images})

                wandb.log({'val_loss':val_loss/n})  #/n_val_ims
        
        print("train,val",epoch_loss/n,val_loss/n)


if __name__ == "__main__":
    wandb.init(mode="disabled")

    with open('config/trainae.yaml', 'r') as stream:
        params = yaml.safe_load(stream)
    
    # run = wandb.init(project="tacvis",name="ae" + "_" + params['note'])
    run = wandb.init(project="tacvis")
    NAME = run.name
    wandb.config.update(params)
    

    dataset = PairedDatasetZoomedOut(params)
    
    if params["load"]:
        encoder = Encoder(params)
        encoder.load(f'{params["encoder_dir"]}')
    else:
        encoder = None
    
    # ae = UNetAE(params,DEVICE,n_classes=3, encoder = encoder)
    ae = AutoEncoder(params,DEVICE,encoder = encoder)
    #This must be using the same encoder as the loaded ones otherwise it would error
    ae.to(DEVICE)
    wandb.watch(ae, criterion=None, log="gradients", log_freq=10, idx=None,log_graph=(False))
    
    # Model directory
    output_dir = f"output_ae/{NAME}_{params['type']}"
    os.makedirs(output_dir,exist_ok=True)
    epochs = tqdm.trange(params["epochs"], desc='Epoch', leave=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), 
                        lr=params["initial_lr"],weight_decay=params["weight_decay"])
    scheduler = ExponentialLR(optimizer, gamma=params["lr_decay"])
    
    train_ae(epochs, ae, dataset, optimizer, scheduler, params, output_dir, params["type"])
