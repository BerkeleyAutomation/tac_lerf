import pytorch_lightning as pl
from tacvis.models import MLP, Encoder, RotationEncoder, compute_similarity_heatmap, rank_distance
from torch import optim
from itertools import chain
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tacvis.losses import batch_contrastive_loss, rot_loss, l2_norm_loss
import glob
import numpy as np
from PIL import Image
import random
import torchvision
import torchvision.transforms.functional as TF
from tacvis.vit import SimpleViT


class ContrastiveModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.tac_enc = Encoder(**params)
        self.rgb_enc = Encoder(**params)
        if self.params['freeze_img_enc']:
            for p in self.rgb_enc.parameters():
                p.requires_grad = False
            for p in self.rgb_enc.resnet.fc.parameters():
                p.requires_grad = True
        if params['use_temp']:
            self.temperature = torch.nn.Parameter(
                torch.tensor(params['init_temperature'], requires_grad=True))
        else:
            self.temperature = torch.tensor(params['init_temperature'])

    def training_step(self, data):

        rgb_batch = data["rgb"]
        tac_batch = data["tac"]
        rgb_feat = self.rgb_enc(rgb_batch)
        tac_feat = self.tac_enc(tac_batch)
        loss = batch_contrastive_loss(rgb_feat, tac_feat, self.temperature)
        self.log("Train contrastive loss", loss, sync_dist=True)
        return loss

    def validation_step(self, data, batch_id):
        rgb_batch = data["rgb"]
        tac_batch = data["tac"]
        rgb_feat = self.rgb_enc(rgb_batch)
        tac_feat = self.tac_enc(tac_batch)
        loss = batch_contrastive_loss(rgb_feat, tac_feat, self.temperature)
        val_loss = loss
        ranks = rank_distance(rgb_feat, tac_feat)
        gold_square = torch.ones([3, 15, 15], device=rgb_feat.device, dtype=torch.float32)
        batch_rank = 0
        imgs = []
        top3 = 0
        top1 = 0
        for b in range(data['rgb'].shape[0]):
            matches = [rgb_batch[b, ...]]
            # this is the rank of the correct label, normalized to 0 to 1 based on batch size
            batch_rank += torch.where(ranks[b, :] == b)[0] / data['rgb'].shape[0]
            for i in range(3):
                match = tac_batch[ranks[b, i], :, :, :].clone()
                # mark the gold images
                if b == ranks[b, i]:
                    match[:, :15, :15] = gold_square
                    top3 += 1
                    if i == 0: top1 += 1
                matches.append(match)
            for i in range(-3, 0, 1):
                # show worst 3 next
                match = tac_batch[ranks[b, i], :, :, :].clone()
                matches.append(match)
            grid = torchvision.utils.make_grid(matches)
            imgs.append(grid)
        batch_rank /= data['rgb'].shape[0]
        return {"val_loss": val_loss.item(), 'batch_rank': batch_rank.item(), 'ranking_grid': imgs,
                'top3_count': top3, 'top1_count': top1, 'n_val_ims': data['rgb'].shape[0]}

    def validation_epoch_end(self, outputs):
        output_dict = {}
        for o in outputs:
            for k in o:
                if k in output_dict:
                    output_dict[k].append(o[k])
                else:
                    output_dict[k] = [o[k]]
        test_imgs = glob.glob(f'{self.params["heatmap_dir"]}/*/image_global.jpg')
        img_name = random.choice(test_imgs)
        tac_name = img_name.replace('global', 'tac_0')
        img = np.asarray(Image.open(img_name))
        tac = np.asarray(Image.open(tac_name))
        img = np.mean(img, axis=2).astype(np.uint8)
        img = np.stack([img, img, img], axis=2)
        heatmaps, _, _ = compute_similarity_heatmap(img, [tac], self.params['heatmap_scales'], [0],
                                                    self.rgb_enc,
                                                    self.tac_enc, self.params['tac_size'],
                                                    stride=img.shape[1] // 50)
        # make sure the heatmaps have 0->1 scaling
        heatmap_ims = []
        for scale in range(heatmaps.shape[1]):
            h = heatmaps[0, scale, :, :]
            h = h + np.min(h) + .001
            h /= (np.max(h) + .001)
            heatmap_ims.append(h)
        self.logger.log_image(key=f'Heatmaps', images=heatmap_ims,
                              caption=[self.params['heatmap_scales'][i] for i in
                                       range(len(self.params['heatmap_scales']))])
        self.logger.log_image(key=f'Image', images=[img])
        self.logger.log_image(key="Tactile", images=[tac])
        self.logger.log_image(key='Alignments',
                              images=list(chain.from_iterable(output_dict['ranking_grid'])))
        self.log('av_rank', np.mean(output_dict['batch_rank']), sync_dist=True)
        self.log('val_loss', np.mean(output_dict['val_loss']), sync_dist=True)
        self.log("top3_count", np.sum(output_dict['top3_count']) / np.sum(output_dict['n_val_ims']),
                 sync_dist=True)
        self.log("top1_count", np.sum(output_dict['top1_count']) / np.sum(output_dict['n_val_ims']),
                 sync_dist=True)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(chain(trainable_params),
                               lr=self.params["initial_lr"],
                               weight_decay=self.params["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma=self.params['lr_decay'])
        return [optimizer], [scheduler]


class RotationModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.continuous = params['continuous']
        self.rotation_list = torch.tensor(params["rotation_list"])
        # self.tac_enc = Encoder(**params)
        # self.rgb_enc = Encoder(**params)

        self.enc = RotationEncoder(**params)
        self.enc.resnet.stem[0] = torch.nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2),
                                                  padding=(1, 1), bias=False)

        if self.continuous:
            self.enc.resnet.fc = MLP(784, 1, params["mlp_layers"])
        else:
            self.enc.resnet.fc = MLP(784, len(params["rotation_list"]), params["mlp_layers"])
        if self.params['freeze_img_enc']:
            for p in self.rgb_enc.parameters():
                p.requires_grad = False
            for p in self.rgb_enc.resnet.fc.parameters():
                p.requires_grad = True
        if params['use_temp']:
            self.temperature = torch.nn.Parameter(
                torch.tensor(params['init_temperature'], requires_grad=True))
        else:
            self.temperature = torch.tensor(params['init_temperature'])

    def training_step(self, data):
        rgb_batch = data["rgb"]
        tac_batch = data["tac"]

        # rgb_feat = self.rgb_enc(rgb_batch)
        # tac_feat = self.tac_enc(tac_batch)
        # pred = self.MLP(torch.cat([rgb_feat,tac_feat],dim=1))

        rgb_tac = torch.cat([rgb_batch, tac_batch], dim=1)
        pred = self.enc(rgb_tac)

        # rgb_tac = torch.cat([rgb_batch,tac_batch],dim=2)
        # pred = self.enc(rgb_tac)

        # print((pred - data['label']).shape)

        # print(pred)

        # loss = l2_norm_loss(pred - data["label"])
        if self.continuous:
            loss = torch.square(pred - data['label']).mean()
        else:
            loss = rot_loss(pred, data["label"])
        # print(loss)
        # print(torch.norm(pred - data['label'], dim=1))
        self.log("Train_loss", loss, sync_dist=True)
        # import pdb;pdb.set_trace()
        return loss

    def rotation_probs(self, rgb_batch, tac_batch):
        # takes in Bx3xhxw torch tensors, returns a list or rotations along with their probabilities
        rgb_tac = torch.cat([rgb_batch, tac_batch], dim=1)
        pred = torch.softmax(self.enc(rgb_tac), 1)
        return pred

    def cont_rotation(self, rgb_batch, tac_batch):
        rgb_tac = torch.cat([rgb_batch, tac_batch], dim=1)
        pred = self.enc(rgb_tac)
        return pred

    def validation_step(self, data, batch_id):
        rgb_batch = data["rgb"]
        tac_batch = data["tac"]
        # rgb_feat = self.rgb_enc(rgb_batch)
        # tac_feat = self.tac_enc(tac_batch)
        # pred = self.MLP(torch.cat([rgb_feat,tac_feat],dim=1))

        rgb_tac = torch.cat([rgb_batch, tac_batch], dim=1)
        pred = self.enc(rgb_tac)

        # rgb_tac = torch.cat([rgb_batch,tac_batch],dim=2)
        # pred = self.enc(rgb_tac)

        # loss = rot_loss(pred, data["label"])
        if self.continuous:
            loss = torch.square(pred - data['label']).mean()
        else:
            loss = rot_loss(pred, data["label"])
        val_loss = loss

        if self.continuous:
            pred_out = pred
            label_out = data['label']
        else:
            _, pred_out = torch.max(pred, 1)
            _, label_out = torch.max(data["label"], 1)

        return {"val_loss": val_loss.item(), 'rgb_batch': rgb_batch, 'tac_batch': tac_batch,
                'pred': pred_out, 'label': label_out}

    def validation_epoch_end(self, outputs):
        output_dict = {}
        for o in outputs:
            for k in o:
                if k in output_dict:
                    output_dict[k].append(o[k])
                else:
                    output_dict[k] = [o[k]]

        imgs = []
        cap = []
        refer_rgb = output_dict["rgb_batch"][0]
        refer_tac = output_dict["tac_batch"][0]
        if self.continuous:
            pred_rot = output_dict['pred'][0] * 180
            gt_rot = output_dict['label'][0] * 180
        else:
            pred_rot = torch.gather(self.rotation_list, 0, output_dict["pred"][0].cpu())
            gt_rot = torch.gather(self.rotation_list, 0, output_dict["label"][0].cpu())

        for i in range(refer_rgb.shape[0]):
            match = []
            original_tac = TF.rotate(refer_tac[i], -gt_rot[i].item())
            pred_tac = TF.rotate(refer_tac[i], -pred_rot[i].item())
            match.append(refer_rgb[i])
            match.append(original_tac)
            match.append(refer_tac[i])
            match.append(pred_tac)
            grid = torchvision.utils.make_grid(match)
            imgs.append(grid)
            cap.append(f'gt:{gt_rot[i]},pred:{pred_rot[i]}')

        self.logger.log_image(key='val_imgs', images=imgs, caption=cap)
        self.log('val_loss', np.mean(output_dict['val_loss']), sync_dist=True)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.Adam(chain(trainable_params),
                               lr=self.params["initial_lr"],
                               weight_decay=self.params["weight_decay"])
        scheduler = ExponentialLR(optimizer, gamma=self.params['lr_decay'])
        return [optimizer], [scheduler]
